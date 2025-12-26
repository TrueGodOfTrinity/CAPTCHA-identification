"""captcha_model.py

CNNEncoder + TransformerEncoder + 分类头。

支持两种训练/推理模式：
- mode="ctc"：输出 (T,B,V_ctc)，用于 CTCLoss（V_ctc = vocab_size + 1，包含 blank）。
- mode="ce" ：固定长度分类（常见验证码），输出 (B,L,V) 用 CrossEntropyLoss。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .cnn_encoder import CNNEncoder
    from .transformer_encoder import TransformerEncoderModule
except Exception:  # pragma: no cover
    from cnn_encoder import CNNEncoder
    from transformer_encoder import TransformerEncoderModule


class CaptchaRecognitionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        blank_idx: int,
        in_channels: int = 3,
        cnn_out: int = 256,
        trans_d_model: int = 256,
        trans_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.blank_idx = int(blank_idx)
        self.num_classes_ctc = self.vocab_size + 1

        self.cnn = CNNEncoder(in_channels=in_channels, out_channels=cnn_out)
        self.encoder = TransformerEncoderModule(
            input_dim=cnn_out,
            d_model=trans_d_model,
            nhead=nhead,
            num_layers=trans_layers,
            dropout=dropout,
        )

        # CTC head: includes blank
        self.ctc_classifier = nn.Linear(trans_d_model, self.num_classes_ctc)
        # Fixed-length CE head: excludes blank
        self.ce_classifier = nn.Linear(trans_d_model, self.vocab_size)

        if not (0 <= self.blank_idx < self.num_classes_ctc):
            raise ValueError(
                f"blank_idx={self.blank_idx} is out of range for num_classes_ctc={self.num_classes_ctc}."
            )

    @staticmethod
    def _resize_mask_to_T(masks: torch.Tensor, T: int) -> torch.Tensor:
        """把像素宽度级别的 mask 下采样到序列长度 T。"""
        if masks.dim() != 2:
            raise ValueError(f"masks must be 2D [B,W], got {tuple(masks.shape)}")
        if masks.size(1) == T:
            return masks
        # bool -> float -> (B,1,W)
        m = masks.float().unsqueeze(1)
        m_down = F.adaptive_avg_pool1d(m, output_size=T).squeeze(1)
        return (m_down > 0.5)

    @staticmethod
    def _pool_to_len(x: torch.Tensor, L: int) -> torch.Tensor:
        """把 [B,T,D] 沿 T 维自适应池化到长度 L，输出 [B,L,D]。"""
        if L <= 0:
            raise ValueError(f"max_len must be > 0, got {L}")
        x = x.transpose(1, 2)  # [B,D,T]
        x = F.adaptive_avg_pool1d(x, L)  # [B,D,L]
        return x.transpose(1, 2).contiguous()  # [B,L,D]

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        *,
        mode: str = "ctc",
        max_len: Optional[int] = None,
    ):
        """Forward.

        Args:
            images: [B,C,H,W]
            masks:  [B,W]（像素级 padding mask，True=pad）
            mode:   "ctc" or "ce"
            max_len: mode="ce" 时必须提供固定长度 L

        Returns:
            - mode="ctc": (logits, input_lengths)
                logits: (T,B,V_ctc)
                input_lengths: (B,)
            - mode="ce": logits_ce: (B,L,V)
        """

        # 1) CNN -> sequence
        seq = self.cnn(images)  # [B,T,D]
        B, T, _ = seq.shape

        mask_T: Optional[torch.Tensor] = None
        if masks is not None:
            masks = masks.to(device=images.device)
            mask_T = self._resize_mask_to_T(masks, T)

        # 2) Transformer
        enc = self.encoder(seq, src_key_padding_mask=mask_T)  # [B,T,D]

        if mode == "ctc":
            logits = self.ctc_classifier(enc)  # [B,T,V_ctc]
            logits = logits.permute(1, 0, 2).contiguous()  # (T,B,V)

            if mask_T is None:
                input_lengths = torch.full((B,), T, dtype=torch.long, device=images.device)
            else:
                # True=pad -> False=valid
                input_lengths = (~mask_T).sum(dim=1).to(dtype=torch.long)
                input_lengths = input_lengths.clamp(min=1)
            return logits, input_lengths

        if mode == "ce":
            if max_len is None:
                raise ValueError("mode='ce' requires max_len")
            pooled = self._pool_to_len(enc, int(max_len))  # [B,L,D]
            logits_ce = self.ce_classifier(pooled)  # [B,L,V]
            return logits_ce

        raise ValueError(f"Unknown mode: {mode}. Use 'ctc' or 'ce'.")