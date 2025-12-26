"""eval.py

评估脚本：evaluate(model, dataloader, device)

支持两种模式：
- loss_mode='ctc'：CTC greedy decode
- loss_mode='ce' ：固定长度 CE decode
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from configs import CFG
from dataset import CaptchaDataset
from utils import collate_fn, ctc_greedy_decode, indices_to_text

try:
    from model.captcha_model import CaptchaRecognitionModel
except Exception:  # pragma: no cover
    from captcha_model import CaptchaRecognitionModel


@torch.no_grad()
def _ce_decode(logits: torch.Tensor) -> List[str]:
    """logits: [B,L,V]"""
    preds = logits.argmax(dim=2).tolist()
    return [indices_to_text(p) for p in preds]


@torch.no_grad()
def evaluate(
    model: CaptchaRecognitionModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    loss_mode: str = "ctc",
    max_label_len: int = 0,
    max_examples: Optional[int] = None,
) -> Tuple[float, List[Tuple[str, str, str]]]:
    model.eval()
    model.to(device)

    n = 0
    n_correct = 0
    examples: List[Tuple[str, str, str]] = []

    for batch in dataloader:
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)
        gts = batch["targets_list"]

        if loss_mode == "ctc":
            logits, input_lengths = model(images, masks, mode="ctc")
            preds = ctc_greedy_decode(logits, input_lengths)
        elif loss_mode == "ce":
            if max_label_len <= 0:
                # 兜底：用 batch 里的 padded 长度
                max_label_len = int(batch["targets_padded"].shape[1])
            logits = model(images, masks, mode="ce", max_len=max_label_len)  # [B,L,V]
            preds_full = _ce_decode(logits)
            # CE 模式下，为了兼容（万一）label 不是固定长度：按 GT 长度截断预测
            preds = [p[: len(gt)] for p, gt in zip(preds_full, gts)]
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        for p, gt in zip(preds, gts):
            if p == gt:
                n_correct += 1
            n += 1
            if len(examples) < 20:
                examples.append((gt, p, "✓" if p == gt else "✗"))

        if max_examples is not None and n >= max_examples:
            break

    acc = n_correct / max(1, n)

    print("\nEvaluation Results:")
    print(f"  total: {n}")
    print(f"  correct: {n_correct}")
    print(f"  seq_acc: {acc:.4f} ({acc*100:.2f}%)")
    print("\nSample predictions (GT -> Pred):")
    print("-" * 40)
    for gt, p, status in examples[:20]:
        print(f"  {status} {gt:12s} -> {p}")
    print()

    return acc, examples


def main() -> None:
    device = torch.device(CFG.device)
    print(f"Using device: {device}\n")

    # model
    model = CaptchaRecognitionModel(
        vocab_size=len(CFG.vocab),
        blank_idx=CFG.blank_idx,
        in_channels=CFG.in_channels,
        cnn_out=CFG.cnn_out,
        trans_d_model=CFG.trans_d_model,
        trans_layers=CFG.trans_layers,
        nhead=CFG.nhead,
        dropout=CFG.dropout,
    )

    ckpt_path = CFG.eval_ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first: python train.py")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    meta = ckpt.get("meta", {}) or {}
    loss_mode = meta.get("loss_mode", "ctc")
    max_label_len = int(meta.get("max_label_len", 0) or 0)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"  loss_mode: {loss_mode}")
    if loss_mode == "ce":
        print(f"  max_label_len: {max_label_len}")
    print()

    split = "val" if os.path.isdir(os.path.join(CFG.data_dir, "val")) else "test"
    ds = CaptchaDataset(CFG.data_dir, split=split)
    dl = DataLoader(ds, batch_size=CFG.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=CFG.num_workers)
    print(f"Evaluating on {split} set ({len(ds)} samples)...\n")
    evaluate(model, dl, device, loss_mode=loss_mode, max_label_len=max_label_len)


if __name__ == "__main__":
    main()
