from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Transformer Encoder 模块：对特征序列建模
输入: [B, T, D_in]
输出: [B, T, D_model]
"""


class PositionalEncoding(nn.Module):
    """
    标准的正弦/余弦位置编码，支持 batch_first=True 的 Transformer。
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # 如果 d_model 为奇数，最后一个维度保持为 0
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        length = x.size(1)
        x = x + self.pe[:, :length]
        return x


class TransformerEncoderModule(nn.Module):
    def __init__(self,
                 input_dim: int = 256,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: CNN 输出特征维度
            d_model: Transformer 隐层维度
            nhead: 注意力头数
            num_layers: encoder 层数
        """
        super().__init__()
        # 如果 input_dim != d_model，用线性映射先投影
        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='relu',
                                                   batch_first=True)
  

        try:
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            # 兼容老版本 torch（没有 enable_nested_tensor 参数）
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_enc = PositionalEncoding(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim]
            src_key_padding_mask: [B, T] 布尔 mask，True 表示该位置被 mask（可选）
        Returns:
            out: [B, T, d_model]
        """
        x = self.input_proj(x)  # [B, T, d_model]
        x = self.pos_enc(x)
        out = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return out
