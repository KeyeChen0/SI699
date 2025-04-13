import torch
import torch.nn as nn
import torch.nn.functional as F


class Dice(nn.Module):
    """Dice activation function (adaptive normalization)."""
    def __init__(self, dim=2, eps=1e-8):
        super(Dice, self).__init__()
        self.bn = nn.BatchNorm1d(dim, eps=eps)
        self.alpha = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        x_norm = self.bn(x)
        p = torch.sigmoid(x_norm)
        return self.alpha * (1 - p) * x + p * x
    
    

import torch
import torch.nn as nn

class MultiheadAttentionLayer(nn.Module):
    """DIN's attention module using PyTorch Multi-Head Attention."""
    def __init__(self, hidden_size, num_heads=2):
        super(MultiheadAttentionLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
    
    def forward(self, query, keys, mask=None):
        """
        query: [B, H] - Candidate item embedding
        keys:  [B, T, H] - Historical behavior sequence
        mask:  [B, T] - Padding mask (boolean), where True indicates padding positions.
        """
        # expand query to match the shape of keys
        # query: [B, H] -> [B, 1, H]
        query = query.unsqueeze(1)
        attn_output, _ = self.attn(query, keys, keys, key_padding_mask=mask)
        return attn_output.squeeze(1)  # [B, H]


    


# user_num = 274003
# item_num = 18369
# max_hist_seq_len = 3368



class Din2025(nn.Module):
    """DIN network without category data."""
    def __init__(self, user_num, item_num, hidden_size=64, num_heads=2):
        super(Din2025, self).__init__()
        self.hidden_size = hidden_size
        
        self.user_emb = nn.Embedding(user_num, hidden_size)
        self.item_emb = nn.Embedding(item_num, hidden_size)
        
        # Multihead Attention
        self.attention = MultiheadAttentionLayer(hidden_size, num_heads)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(3 * hidden_size, 2 * hidden_size),
            Dice(dim=2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size),
            Dice(dim=hidden_size),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, user_ids, item_ids, hist_item_ids, hist_seq_mask):
        """
        user_ids:    [B]
        item_ids:    [B]
        hist_item_ids:  [B, T]
        hist_seq_mask:  [B, T]
        """
        user_emb = self.user_emb(user_ids)  # [B, H]
        item_emb = self.item_emb(item_ids)  # [B, H]
        hist_emb = self.item_emb(hist_item_ids)  # [B, T, H]
        
        max_len = hist_item_ids.shape[1]
        interest = self.attention(item_emb, hist_emb, hist_seq_mask)  # [B, H]
        combined = torch.cat([user_emb, interest, item_emb], dim=1)  # [B, 3H]
        logits = self.fc(combined).squeeze(-1)  # [B]
        return logits



class DinAir(nn.Module):
    def __init__(self, item_num, cat_num, num_feature_size, hidden_size=64, num_heads=2, max_seq_len=512):
        super(DinAir, self).__init__()
        self.hidden_size = hidden_size
        self.item_emb = nn.Embedding(item_num, hidden_size // 2)
        self.cat_linear = nn.Linear(cat_num, hidden_size // 2)
        
        # Multihead Attention modules
        self.attention_seq = MultiheadAttentionLayer(hidden_size, num_heads)
        
        # batch norm
        self.seq_transformer_bn = nn.BatchNorm1d(hidden_size)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size + num_feature_size, 2 * hidden_size),
            Dice(dim=2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size),
            Dice(dim=hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, item_ids, item_cats, item_num_features,
                hist_item_ids, hist_item_cats, seq_mask):
        """
          item_ids:         [B]
          item_cats:        [cat_num] - item category vector
          item_num_features:[B, F] - numerical features for items
          hist_item_ids:    [B, T] - historical item ids sequence, padding with 0
          hist_item_cats:   [B, cat_num] - historical item category ids, padding with 0
          seq_mask:         [B, T] - historical sequence padding mask (True for padding)
        """
        item_emb = self.item_emb(item_ids)          # [B, H//2]
        hist_item_emb = self.item_emb(hist_item_ids)  # [B, T, H//2]
        item_cats = self.cat_linear(item_cats)      # [H//2]
        hist_item_cats = self.cat_linear(hist_item_cats)  # [B, T, H//2]
        
        # Concatenate item embedding and pooled category embedding
        item_concat_emb = torch.cat([item_emb, item_cats], dim=1)  # [B, H]
        hist_concat_emb = torch.cat([hist_item_emb, hist_item_cats], dim=2)  # [B, T, H]
        
        # Sequence attention：
        interest = self.attention_seq(
            query=item_concat_emb,  # [B, H]
            keys=hist_concat_emb,   # [B, T, H]
            mask=seq_mask           # [B, T]
        )  # [B, H]


        # sequence add & nor
        interest = self.seq_transformer_bn(item_concat_emb + interest)

        
        combined = torch.cat([interest, item_concat_emb, item_num_features], dim=1)  # [B, 2H + F]
        logits = self.fc(combined).squeeze(-1)  # [B]
        return logits


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, theta=10000.0):
        """
        params:
          dim: fixed dimension of the input tensor, must be even
          max_seq_len: max supported sequence length
          theta: default 10000.0
        """
        super(RotaryPositionalEmbedding, self).__init__()
        assert dim % 2 == 0, "dim must be even for RoPE."
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len


        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_cached", cos)
        self.register_buffer("sin_cached", sin)

    def forward(self, x):
        """
        params:
          x: Tensor, shape [B, L, D]。
        return:
          x_rotated: shape [B, L, D]
        """
        B, L, D = x.shape
        if D != self.dim:
            raise ValueError(f"Input dimension {D} does not match model dimension {self.dim}.")
        if L > self.max_seq_len:
            raise ValueError(f"Sequence length {L} exceeds max_seq_len {self.max_seq_len}.")

        cos = self.cos_cached[:L]
        sin = self.sin_cached[:L]

        cos = cos.unsqueeze(0).expand(B, -1, -1)
        sin = sin.unsqueeze(0).expand(B, -1, -1)

        x1, x2 = x[..., ::2], x[..., 1::2] 

        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        x_rotated = x_rotated.flatten(-2)
        return x_rotated
    


class DinPro(nn.Module):
    def __init__(self, user_num, item_num, cat_num, num_feature_size, hidden_size=64, num_heads=2, max_seq_len=512):
        super(DinPro, self).__init__()
        self.hidden_size = hidden_size
        self.user_emb = nn.Embedding(user_num, hidden_size)
        self.item_emb = nn.Embedding(item_num, hidden_size // 2)
        self.cat_linear = nn.Linear(cat_num, hidden_size // 2)
        
        # Multihead Attention modules
        self.attention_seq = MultiheadAttentionLayer(hidden_size, num_heads)
        
        # batch norm
        self.seq_transformer_bn = nn.BatchNorm1d(hidden_size)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(3 * hidden_size + num_feature_size, 2 * hidden_size),
            Dice(dim=2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size),
            Dice(dim=hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, user_ids, item_ids, item_cats, item_num_features,
                hist_item_ids, hist_item_cats, seq_mask):
        """
          user_ids:         [B]
          item_ids:         [B]
          item_cats:        [cat_num] - item category vector
          item_num_features:[B, F] - numerical features for items
          hist_item_ids:    [B, T] - historical item ids sequence, padding with 0
          hist_item_cats:   [B, cat_num] - historical item category ids, padding with 0
          seq_mask:         [B, T] - historical sequence padding mask (True for padding)
        """
        user_emb = self.user_emb(user_ids)          # [B, H]
        item_emb = self.item_emb(item_ids)          # [B, H//2]
        hist_item_emb = self.item_emb(hist_item_ids)  # [B, T, H//2]
        item_cats = self.cat_linear(item_cats)      # [H//2]
        hist_item_cats = self.cat_linear(hist_item_cats)  # [B, T, H//2]
        
        # Concatenate item embedding and pooled category embedding
        item_concat_emb = torch.cat([item_emb, item_cats], dim=1)  # [B, H]
        hist_concat_emb = torch.cat([hist_item_emb, hist_item_cats], dim=2)  # [B, T, H]
        
        # Sequence attention：
        interest = self.attention_seq(
            query=item_concat_emb,  # [B, H]
            keys=hist_concat_emb,   # [B, T, H]
            mask=seq_mask           # [B, T]
        )  # [B, H]


        # sequence add & nor
        interest = self.seq_transformer_bn(item_concat_emb + interest)

        combined = torch.cat([user_emb, interest, item_concat_emb, item_num_features], dim=1)  # [B, 3H + F]
        logits = self.fc(combined).squeeze(-1)  # [B]
        return logits




class DinProMax(nn.Module):
    """DIN network with category data and numerical features.
    """
    def __init__(self, user_num, item_num, cat_num, num_feature_size, hidden_size=64, num_heads=2, max_seq_len=512):
        super(DinProMax, self).__init__()
        self.hidden_size = hidden_size
        
        self.user_emb = nn.Embedding(user_num, hidden_size)
        self.item_emb = nn.Embedding(item_num, hidden_size // 2)
        self.cat_emb = nn.Embedding(cat_num, hidden_size // 2)

        # Positional encoding
        self.pos_emb = RotaryPositionalEmbedding(hidden_size, max_seq_len=max_seq_len)
        
        # Multihead Attention modules
        self.attention_cat = MultiheadAttentionLayer(hidden_size // 2, num_heads // 2)
        self.attention_seq = MultiheadAttentionLayer(hidden_size, num_heads)
        
        # batch norm
        self.cat_transformer_bn = nn.BatchNorm1d(hidden_size // 2)
        self.seq_transformer_bn = nn.BatchNorm1d(hidden_size)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(3 * hidden_size + num_feature_size, 2 * hidden_size),
            Dice(dim=2 * hidden_size),
            nn.Linear(2 * hidden_size, hidden_size),
            Dice(dim=hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, user_ids, item_ids, item_cats, item_cat_mask, item_num_features,
                hist_item_ids, hist_item_cats, hist_item_cat_mask, seq_mask):
        """
          user_ids:         [B]
          item_ids:         [B]
          item_cats:        [B, N] - item category ids, padding with 0
          item_cat_mask:    [B, N] - item category padding mask (True for padding)
          item_num_features:[B, F] - numerical features for items
          hist_item_ids:    [B, T] - historical item ids sequence, padding with 0
          hist_item_cats:   [B, T, N] - historical item category ids, padding with 0
          hist_item_cat_mask:[B, T, N] - historical item category padding mask (True for padding)
          seq_mask:         [B, T] - historical sequence padding mask (True for padding)
        """
        user_emb = self.user_emb(user_ids)          # [B, H]
        item_emb = self.item_emb(item_ids)          # [B, H//2]
        hist_item_emb = self.item_emb(hist_item_ids)  # [B, T, H//2]
        cat_emb = self.cat_emb(item_cats)           # [B, N, H//2]
        hist_cat_emb = self.cat_emb(hist_item_cats)   # [B, T, N, H//2]
        
        # Attention for item category data:
        item_cat_pooled_emb = self.attention_cat(
            query=item_emb,    # [B, H//2]
            keys=cat_emb,      # [B, N, H//2]
            mask=item_cat_mask # [B, N]
        )  # [B, H//2]

        # category bn
        item_cat_pooled_emb = self.cat_transformer_bn(item_emb + item_cat_pooled_emb) # [B, H//2]
        
        # Concatenate item embedding and pooled category embedding
        item_concat_emb = torch.cat([item_emb, item_cat_pooled_emb], dim=1)  # [B, H]
        
        # Pooling historical item category embeddings
        B, T, N, _ = hist_cat_emb.shape
        hist_cat_emb = hist_cat_emb.view(-1, N, hist_cat_emb.size(-1))
        hist_item_emb = hist_item_emb.view(-1, hist_item_emb.size(-1))
        hist_item_cat_mask = hist_item_cat_mask.view(-1, N)
        
        hist_cat_pooled_emb = self.attention_cat(
            query=hist_item_emb,       # [B*T, H//2]
            keys=hist_cat_emb,         # [B*T, N, H//2]
            mask=hist_item_cat_mask    # [B*T, N]
        )  # [B*T, H//2]


        hist_concat_emb = torch.cat([hist_item_emb, hist_cat_pooled_emb], dim=1)  # [B*T, H]
        hist_concat_emb = hist_concat_emb.view(B, T, -1) # [B, T, H]
        
        # Sequence attention：
        interest = self.attention_seq(
            query=item_concat_emb,  # [B, H]
            keys=hist_concat_emb,   # [B, T, H]
            mask=seq_mask           # [B, T]
        )  # [B, H]

        # sequence bn
        interest = self.seq_transformer_bn(item_concat_emb + interest)

        
        combined = torch.cat([user_emb, interest, item_concat_emb, item_num_features], dim=1)  # [B, 3H + F]
        logits = self.fc(combined).squeeze(-1)  # [B]
        return logits