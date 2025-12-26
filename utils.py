"""utils.py

工具函数：
- 字符映射（text <-> indices）
- CTC greedy 解码
- DataLoader 的 collate_fn（同时支持 CTC 与固定长度 CE）
- checkpoint 保存

"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from configs import CFG


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """替代 torchvision.transforms.functional.to_tensor，避免 torchvision 版本兼容问题。

    Returns:
        FloatTensor [C,H,W] in [0,1]
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    # HWC -> CHW
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
    t = torch.from_numpy(arr).float().div(255.0)
    return t


def seed_everything(seed: int) -> None:
    """尽量保证可复现。"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    meta: Optional[Dict] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta": meta or {},
    }
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")


def text_to_indices(text: str, *, strict: bool = True) -> List[int]:
    """把字符串转换为索引列表。
    strict=True：遇到未知字符直接抛异常，方便定位数据问题。
    """
    ids: List[int] = []
    for ch in text:
        if ch in CFG.char2idx:
            ids.append(CFG.char2idx[ch])
        else:
            if strict:
                raise ValueError(
                    f"Unknown character {ch!r} in label {text!r}. "
                    f"Please add it to CFG.vocab (configs.py) or fix filename labels."
                )
            # 非严格模式：跳过
    return ids


def indices_to_text(indices: List[int]) -> str:
    return "".join(CFG.idx2char[i] for i in indices if i in CFG.idx2char)


@torch.no_grad()
def ctc_greedy_decode(logits: torch.Tensor, input_lengths: torch.Tensor) -> List[str]:
    """Greedy CTC decode。

    Args:
        logits: (T, B, V) raw logits（或 log_probs），我们仅用 argmax
        input_lengths: (B,) 每个样本的有效时间步长度
    """
    if logits.device.type != "cpu":
        logits = logits.cpu()
    if input_lengths.device.type != "cpu":
        input_lengths = input_lengths.cpu()

    T, B, V = logits.shape
    best = logits.argmax(dim=2).permute(1, 0).contiguous()  # (B, T)

    results: List[str] = []
    for b in range(B):
        L = int(input_lengths[b].item())
        seq = best[b, :L].tolist()
        out: List[int] = []
        prev = None
        for idx in seq:
            if idx == CFG.blank_idx:
                prev = None
                continue
            if prev is None or idx != prev:
                out.append(idx)
            prev = idx
        results.append(indices_to_text(out))
    return results


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    把不同宽度的图片 pad 到同一宽度，并生成 mask。
    """

    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]

    # 1) image -> tensor, pad width
    tensors: List[torch.Tensor] = []
    widths: List[int] = []

    for img in images:
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if isinstance(img, Image.Image):
            t = _pil_to_tensor(img)  # [C,H,W] float32 in [0,1]
        else:
            t = img
        if t.dtype != torch.float32:
            t = t.float()
        tensors.append(t)
        widths.append(int(t.shape[-1]))

    max_w = max(widths)
    B = len(tensors)
    C, H, _ = tensors[0].shape

    images_padded = torch.zeros((B, C, H, max_w), dtype=torch.float32)
    # masks: True means padded (to be masked)
    masks = torch.zeros((B, max_w), dtype=torch.bool)
    for i, t in enumerate(tensors):
        w = t.shape[-1]
        images_padded[i, :, :, :w] = t
        if w < max_w:
            masks[i, w:] = True

    # 2) labels -> indices
    target_seqs = [text_to_indices(s, strict=True) for s in labels]
    target_lengths = torch.tensor([len(s) for s in target_seqs], dtype=torch.long)
    if (target_lengths == 0).any():
        # 理论上 strict=True 不会到这一步（除非 label 为空）
        raise ValueError(f"Found empty target after mapping. Labels: {labels}")

    targets_concat = torch.tensor([i for seq in target_seqs for i in seq], dtype=torch.long)

    # 3) padded targets for CE
    max_t = int(target_lengths.max().item())
    targets_padded = torch.full((B, max_t), fill_value=-100, dtype=torch.long)  # ignore_index
    for i, seq in enumerate(target_seqs):
        targets_padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)

    return {
        "images": images_padded,
        "masks": masks,
        "widths": torch.tensor(widths, dtype=torch.long),
        "targets_concat": targets_concat,
        "target_lengths": target_lengths,
        "targets_padded": targets_padded,
        "targets_list": labels,
    }