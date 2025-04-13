import json
import torch
import torch.nn as nn
from typing import List, Tuple
from src.model_din import DinAir  # 假设你已有 DinAir 定义

# ---------- 数据预加载 ----------
def load_resources(data_dir: str = "small_10", model_path: str = "models/din_air.pt"):
    """加载所有资源并返回一个字典"""

    with open(f'{data_dir}/appid_dic.json', 'r') as f:
        appid_read = {int(k): v for k, v in json.load(f).items()}

    with open(f'{data_dir}/appid_user_dic.json', 'r') as f:
        userid_read = {int(k): v for k, v in json.load(f).items()}

    with open(f'{data_dir}/gid2pos_map.json', 'r') as f:
        gid2pos = {int(k): v for k, v in json.load(f).items()}

    with open(f'{data_dir}/game_tags_mapping.json', 'r') as f:
        mapping_data = json.load(f)
        game_cat_map = {
            int(k): torch.zeros(442, dtype=torch.int).scatter(0, torch.tensor(v, dtype=torch.long), 1)
            for k, v in mapping_data.items()
        }

    model = DinAir(
        item_num=18369,
        cat_num=442,
        num_feature_size=4,
        hidden_size=64,
        num_heads=2
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    print("所有资源加载完成。")
    return {
        "appid_read": appid_read,
        "userid_read": userid_read,
        "gid2pos": gid2pos,
        "game_cat_map": game_cat_map,
        "model": model
    }


# ---------- 推理函数 ----------
def inference(model: nn.Module, game_history: List[int], candidate_game_id: int,
              numerical_features: List[float], game_cat_map: dict) -> float:
    model.eval()
    with torch.no_grad():
        candidate_game_id_tensor = torch.tensor([candidate_game_id], dtype=torch.long)
        candidate_game_cats = game_cat_map.get(candidate_game_id, torch.zeros(442, dtype=torch.int)).unsqueeze(0).float()
        item_num_features = torch.tensor(numerical_features, dtype=torch.float).unsqueeze(0)

        if len(game_history) == 0:
            hist_item_ids = torch.zeros((1, 1), dtype=torch.long)
            history_cats = torch.zeros((1, 1, 442), dtype=torch.float)
            seq_mask = torch.ones((1, 1), dtype=torch.bool)
        else:
            hist_item_ids = torch.tensor(game_history, dtype=torch.long).unsqueeze(0)
            history_cats = torch.stack([
                game_cat_map.get(game_id, torch.zeros(442, dtype=torch.int)) for game_id in game_history
            ]).unsqueeze(0).float()
            seq_mask = torch.zeros(hist_item_ids.shape, dtype=torch.bool)

        output = model(
            candidate_game_id_tensor,
            candidate_game_cats,
            item_num_features,
            hist_item_ids,
            history_cats,
            seq_mask
        )
        return torch.sigmoid(output.squeeze()).item()


# ---------- 主推荐函数 ----------
def recommend_for_user(userid: int, resources: dict, topk: int = 10) -> List[Tuple[int, float]]:
    appid_read = resources["appid_read"]
    userid_read = resources["userid_read"]
    gid2pos = resources["gid2pos"]
    game_cat_map = resources["game_cat_map"]
    model = resources["model"]

    if userid not in userid_read:
        raise ValueError(f"用户 ID {userid} 不存在。")

    user_history = userid_read[userid]
    game_history = [gid2pos[appid] for appid in user_history if appid in gid2pos]
    user_history_set = set(user_history)

    recs = []
    for appid in appid_read:
        if appid not in user_history_set and appid in gid2pos:
            pos_id = gid2pos[appid]
            num_features = appid_read[appid]
            score = inference(model, game_history, pos_id, num_features, game_cat_map)
            recs.append((appid, score))

    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:topk]



if __name__ == "__main__":
    resources = load_resources()  # 默认从 small_10/ 目录加载
    user_id = 181
    try:
        top_games = recommend_for_user(user_id, resources, topk=10)
        print("Rec Results:")
        for appid, score in top_games:
            print(f"AppID: {appid}, Prob: {score:.4f}")
    except ValueError as e:
        print(f"Error: {e}")
