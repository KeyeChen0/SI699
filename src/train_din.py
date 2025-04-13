import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import ast
from model_din import Din2025, DinAir, DinPro, DinProMax
from tqdm import trange, tqdm
import torch.nn.functional as F


import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

import wandb
from sklearn.metrics import accuracy_score

import logging
import math
import json




logging.basicConfig(
    filename='my_log_file.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)



game_cat_map = {}
for key, value in json.load(open('small_10/game_tags_mapping.json', 'r')).items():
    game_cat_map[int(key)] = torch.zeros(442, dtype=torch.int).scatter(0, torch.tensor(value, dtype=torch.long), 1)
print("Game category map loaded.")



class StreamingDataset(IterableDataset):
    def __init__(self, file_path, chunksize=128, transform=None):
        """
        :param file_path: data file path
        :param chunksize: chunk size for reading the file
        :param transform: function to transform the data
        """
        self.file_path = file_path
        self.chunksize = chunksize
        self.transform = transform
            

    def __iter__(self):
        for chunk in pd.read_csv(
            self.file_path,
            chunksize=self.chunksize
        ):
            for _, row in chunk.iterrows():
                sample = row.to_dict()
                if self.transform:
                    sample = self.transform(sample)
                yield sample




def transform_fn_din_air(sample):
    result = {}
    result['user_id'] = torch.tensor(sample['user_id'], dtype=torch.int)
    result['game_id'] = torch.tensor(sample['game_id'], dtype=torch.int)
    
    result['game_cats'] = game_cat_map.get(sample['game_id'], torch.zeros(442, dtype=torch.int))

    game_history_seq = torch.tensor(ast.literal_eval(sample['game_history']), dtype=torch.int)
    result['game_history'] = game_history_seq

    game_history_cats_list = []
    for game in game_history_seq.tolist():
        game_history_cats_list.append(game_cat_map.get(game, torch.zeros(442, dtype=torch.int)))
    result['game_history_cats'] = torch.stack(game_history_cats_list)
    
    result['num_features'] = torch.tensor([
        sample['num_feature1'],
        sample['num_feature2'],
        sample['num_feature3'],
        sample['num_feature4']
    ], dtype=torch.float)
    
    if "label" in sample:
        result['label'] = torch.tensor(sample['label'], dtype=torch.long).float()
    return result




def collate_fn_din_air(batch, max_hist_len=256):
    """
    Dynamic padding for batch data according to the maximum length of the history.
    """
    user_ids      = torch.stack([sample['user_id'] for sample in batch])
    item_ids      = torch.stack([sample['game_id'] for sample in batch])
    item_cats     = torch.stack([sample['game_cats'] for sample in batch])
    num_features  = torch.stack([sample['num_features'] for sample in batch])
    labels        = torch.stack([sample['label'] for sample in batch]) if "label" in batch[0] else None

    orig_lengths = [sample['game_history'].size(0) for sample in batch]
    target_seq_len = min(max(orig_lengths), max_hist_len)
    
    
    histories = [
        sample['game_history'][-target_seq_len:] if sample['game_history'].size(0) > target_seq_len 
        else sample['game_history']
        for sample in batch
    ]
    # in case the padded value collides with the actual data 0, first use NaN to pad
    histories = [h.float() for h in histories]
    # （padding_value = NaN）
    padded_histories = torch.nn.utils.rnn.pad_sequence(histories, batch_first=True, padding_value=float('nan'))
    # generate mask on NaN values
    seq_masks = torch.isnan(padded_histories)
    # replace NaN with 0
    padded_histories = torch.nan_to_num(padded_histories, nan=0).long()
    
    histories_cats = [
        sample['game_history_cats'][-target_seq_len:] if sample['game_history_cats'].size(0) > target_seq_len 
        else sample['game_history_cats']
        for sample in batch
    ]
    padded_hist_cats = torch.nn.utils.rnn.pad_sequence(histories_cats, batch_first=True, padding_value=0)
    
    
    return {
        'user_ids':           user_ids.long(),
        'item_ids':           item_ids.long(),
        'item_cats':          item_cats.float(),
        'item_num_features':  num_features,
        'hist_item_ids':      padded_histories.long(),      # [B, T]
        'hist_item_cats':     padded_hist_cats.float(),       # [B, T, 442]
        'seq_mask':           seq_masks,                    # [B, T]
        'labels':             labels.float() if labels is not None else None
    }





if __name__ == "__main__":
    # total_len = 11076267
    # n_category = 442

    train_dataset = StreamingDataset('small_10/local_train_mapped', chunksize=4096, transform=transform_fn_din_air)
    train_dataloader = DataLoader(train_dataset, batch_size=512, collate_fn=collate_fn_din_air)
    test_dataset = StreamingDataset('small_10/local_test_mapped', chunksize=4096, transform=transform_fn_din_air)
    test_dataloader = DataLoader(test_dataset, batch_size=512, collate_fn=collate_fn_din_air)

    model = DinAir(
        # user_num=274003,
        item_num=18369,
        cat_num=442,
        num_feature_size=4,
        hidden_size=64,
        num_heads=2
    )
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    wandb.init(project="si699_proj", name="run")

    num_epochs = 20
    global_step = 0
    total_train_len = 11076267


    for epoch in trange(num_epochs, desc="Epoch"):
        model.train()
        train_loss_sum = 0
        train_set_len = 0

        for step, data in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
            # user_ids         = data['user_ids'].to(device)         # [B]
            item_ids         = data['item_ids'].to(device)         # [B]
            item_cats        = data['item_cats'].to(device)        # [B, N]
            item_num_features= data['item_num_features'].to(device)  # [B, F]
            hist_item_ids    = data['hist_item_ids'].to(device)    # [B, T]
            hist_item_cats   = data['hist_item_cats'].to(device)   # [B, T, N]
            seq_mask         = data['seq_mask'].to(device)         # [B, T]
            label            = data['labels'].to(device).float()

            optimizer.zero_grad()
            # 模型输入顺序需与forward方法保持一致
            y_pred = model(
                # user_ids,
                item_ids,
                item_cats,
                item_num_features,
                hist_item_ids,
                hist_item_cats,
                seq_mask
            )
            loss = criterion(y_pred.squeeze(), label)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_set_len += len(label)
            wandb.log({"train_loss": loss.item()}, step=global_step)
            global_step += 1

        avg_train_loss = train_loss_sum / train_set_len
        wandb.log({"train_epoch_loss": avg_train_loss}, step=global_step)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")


        # save model checkpoint
        torch.save(model.state_dict(), f"models_pro/model_checkpoint_{epoch}.pt")
        print(f"Model checkpoint saved at model_checkpoint_{epoch}.pt")

        # 模型验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in tqdm(test_dataloader, desc="Evaluating", leave=False):
                # user_ids         = data['user_ids'].to(device)         # [B]
                item_ids         = data['item_ids'].to(device)         # [B]
                item_cats        = data['item_cats'].to(device)        # [B, N]
                item_num_features= data['item_num_features'].to(device)  # [B, F]
                hist_item_ids    = data['hist_item_ids'].to(device)    # [B, T]
                hist_item_cats   = data['hist_item_cats'].to(device)   # [B, T, N]
                seq_mask         = data['seq_mask'].to(device)         # [B, T]
                label            = data['labels'].to(device).float()

                y_pred = model(
                    # user_ids,
                    item_ids,
                    item_cats,
                    item_num_features,
                    hist_item_ids,
                    hist_item_cats,
                    seq_mask
                )
                y_pred = torch.sigmoid(y_pred)

                preds = (y_pred.squeeze() >= 0.5).long().cpu().numpy()
                labels = label.long().cpu().numpy()

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        wandb.log({"eval_accuracy": accuracy}, step=global_step)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Evaluation Accuracy: {accuracy:.4f}")


    """
    train_dataset = StreamingDataset('small_10/local_train_mapped', chunksize=4096, transform=transform_fn_din_air)
    train_dataloader = DataLoader(train_dataset, batch_size=512, collate_fn=collate_fn_din_air)
    test_dataset = StreamingDataset('small_10/local_test_mapped', chunksize=4096, transform=transform_fn_din_air)
    test_dataloader = DataLoader(test_dataset, batch_size=512, collate_fn=collate_fn_din_air)


    model = DinAir(
        #user_num=274003,
        item_num=18369,
        cat_num=442,
        num_feature_size=4,
        hidden_size=64,
        num_heads=2
    )
    model.load_state_dict(torch.load("models_air/model_checkpoint_19.pt"))
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    wandb.init(
        project="si699_proj",
        id="p36kjfo0",     # 唯一标识符，用于恢复
        resume="allow"             # 允许恢复或创建新 run
    )

    num_epochs = 25
    start_epoch = 20
    global_step = 432680
    total_train_len = 11076267


    for epoch in trange(start_epoch, num_epochs, desc="Epoch"):
        model.train()
        train_loss_sum = 0
        train_set_len = 0

        for step, data in enumerate(tqdm(train_dataloader, desc="Training", leave=False)):
            user_ids         = data['user_ids'].to(device)         # [B]
            item_ids         = data['item_ids'].to(device)         # [B]
            item_cats        = data['item_cats'].to(device)        # [B, N]
            item_num_features= data['item_num_features'].to(device)  # [B, F]
            hist_item_ids    = data['hist_item_ids'].to(device)    # [B, T]
            hist_item_cats   = data['hist_item_cats'].to(device)   # [B, T, N]
            seq_mask         = data['seq_mask'].to(device)         # [B, T]
            label            = data['labels'].to(device).float()

            optimizer.zero_grad()
            # 模型输入顺序需与forward方法保持一致
            y_pred = model(
                #user_ids,
                item_ids,
                item_cats,
                item_num_features,
                hist_item_ids,
                hist_item_cats,
                seq_mask
            )
            loss = criterion(y_pred.squeeze(), label)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_set_len += len(label)
            wandb.log({"train_loss": loss.item()}, step=global_step)
            global_step += 1

        avg_train_loss = train_loss_sum / train_set_len
        wandb.log({"train_epoch_loss": avg_train_loss}, step=global_step)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")


        # save model checkpoint
        torch.save(model.state_dict(), f"models_air/model_checkpoint_{epoch}.pt")
        print(f"Model checkpoint saved at model_checkpoint_{epoch}.pt")

        # 模型验证阶段
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in tqdm(test_dataloader, desc="Evaluating", leave=False):
                user_ids         = data['user_ids'].to(device)         # [B]
                item_ids         = data['item_ids'].to(device)         # [B]
                item_cats        = data['item_cats'].to(device)        # [B, N]
                item_num_features= data['item_num_features'].to(device)  # [B, F]
                hist_item_ids    = data['hist_item_ids'].to(device)    # [B, T]
                hist_item_cats   = data['hist_item_cats'].to(device)   # [B, T, N]
                seq_mask         = data['seq_mask'].to(device)         # [B, T]
                label            = data['labels'].to(device).float()

                y_pred = model(
                    #user_ids,
                    item_ids,
                    item_cats,
                    item_num_features,
                    hist_item_ids,
                    hist_item_cats,
                    seq_mask
                )
                y_pred = torch.sigmoid(y_pred)

                preds = (y_pred.squeeze() >= 0.5).long().cpu().numpy()
                labels = label.long().cpu().numpy()

                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())

        accuracy = accuracy_score(all_labels, all_preds)
        wandb.log({"eval_accuracy": accuracy}, step=global_step)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Evaluation Accuracy: {accuracy:.4f}") 
        """



