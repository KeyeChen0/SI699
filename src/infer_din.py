import torch
import torch.nn as nn
import json
from model_din import DinAir  # 假设 model_din.py 中已经定义了 DinAir 模型

# 加载 game_cat_map
game_cat_map = {}
with open('small_10/game_tags_mapping.json', 'r') as f:
    mapping_data = json.load(f)
    for key, value in mapping_data.items():
        game_cat_map[int(key)] = torch.zeros(442, dtype=torch.int).scatter(0, torch.tensor(value, dtype=torch.long), 1)
print("Game category map loaded.")

def inference(model, game_history, candidate_game_id, numerical_features):
    """
    推理函数：
    - model: 模型实例
    - game_history: List[int]，用户历史游戏 ID
    - candidate_game_id: int，候选游戏 ID
    - numerical_features: List[float]，数值特征（例如：[num_feature1, num_feature2, num_feature3, num_feature4]）
    
    返回值：预测的数值概率（经过 sigmoid 函数映射）
    """
    model.eval()
    with torch.no_grad():
        # 候选游戏ID转换为tensor，并从game_cat_map中获取对应的游戏类别向量
        candidate_game_id_tensor = torch.tensor([candidate_game_id], dtype=torch.long)
        candidate_game_cats = game_cat_map.get(candidate_game_id, torch.zeros(442, dtype=torch.int)).unsqueeze(0).float()
        
        # 数值特征 tensor，shape: [1, 特征数]
        item_num_features = torch.tensor(numerical_features, dtype=torch.float).unsqueeze(0)
        
        # 将用户历史游戏ID转换为 tensor，shape: [1, T]
        hist_item_ids = torch.tensor(game_history, dtype=torch.long).unsqueeze(0)
        
        # 为每个历史游戏从game_cat_map中获取类别向量，stack后 shape: [1, T, 442]
        history_cats = torch.stack([
            game_cat_map.get(game_id, torch.zeros(442, dtype=torch.int)) for game_id in game_history
        ]).unsqueeze(0).float()
        
        # 序列掩码，由于这里未做padding，因此全为False，shape: [1, T]
        seq_mask = torch.zeros(hist_item_ids.shape, dtype=torch.bool)
        
        # 模型 forward 的输入顺序需与训练时保持一致：
        # candidate_game_id_tensor, candidate_game_cats, item_num_features, hist_item_ids, history_cats, seq_mask
        output = model(
            candidate_game_id_tensor,
            candidate_game_cats,
            item_num_features,
            hist_item_ids,
            history_cats,
            seq_mask
        )
        # 输出经过 sigmoid 映射为概率
        prob = torch.sigmoid(output.squeeze()).item()
    return prob

if __name__ == "__main__":
    # 模拟构造输入数据
    
    # 1. 实例化模型（参数同训练时设置）
    model = DinAir(
        item_num=18369,
        cat_num=442,
        num_feature_size=4,
        hidden_size=64,
        num_heads=2
    )
    params_path = "models/din_air.pt"
    # model.load_state_dict(torch.load(params_path))
    model.load_state_dict(torch.load(params_path, map_location=torch.device('cpu')))

    
    # 注意：此处未加载预训练权重，实际使用时可加载保存的模型 checkpoint
    
    # 2. 构造模拟的用户历史行为（游戏 ID 列表）
    game_history = [101, 202, 303, 404, 505]
    
    # 3. 候选游戏 ID
    candidate_game_id = 606
    
    # 4. 数值特征（示例）
    numerical_features = [0.1, 0.2, 0.3, 0.4]
    
    # 调用推理函数，得到预测概率
    probability = inference(model, game_history, candidate_game_id, numerical_features)
    print("预测概率：", probability)