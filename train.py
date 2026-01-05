"""
train.py
"""

from __future__ import annotations

import os
import time
from typing import Optional, Tuple
import contextlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import CFG
from dataset import CaptchaDataset
from utils import collate_fn, save_checkpoint, seed_everything

# 兼容两种项目布局
try:
    from model.captcha_model import CaptchaRecognitionModel
except Exception:  # pragma: no cover
    from captcha_model import CaptchaRecognitionModel


def _ctc_supported_on(device: torch.device) -> bool:
    """在当前 device 上做一次 CTCLoss 前向+反向，检测是否可用。"""
    ctc = torch.nn.CTCLoss(blank=CFG.blank_idx, zero_infinity=True, reduction="mean")
    try:
        T, B, V = 8, 2, CFG.num_classes_ctc
        logits = torch.randn(T, B, V, device=device, requires_grad=True)
        targets = torch.randint(0, V - 1, (B * 3,), dtype=torch.long, device=device)  # 不采样 blank
        input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
        target_lengths = torch.full((B,), 3, dtype=torch.long, device=device)
        loss = ctc(F.log_softmax(logits, dim=2), targets, input_lengths, target_lengths)
        loss.backward()
        return True
    except Exception as e:
        print(f"⚠️  CTCLoss not supported on device={device}: {e}")
        return False


def _infer_label_stats(dataset: CaptchaDataset) -> Tuple[bool, int]:
    """从 dataset 解析 label 长度，判断是否固定长度。"""
    lengths = [len(s) for s in dataset.labels]
    uniq = sorted(set(lengths))
    fixed = len(uniq) == 1
    max_len = max(lengths) if lengths else 0
    print(f"Label lengths (unique): {uniq[:10]}{' ...' if len(uniq) > 10 else ''}")
    return fixed, max_len


def train_one_epoch(
    model: CaptchaRecognitionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    *,
    loss_mode: str,
    max_label_len: int,
    use_amp: bool,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> float:
    model.train()
    total_loss = 0.0
    n = 0

    ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    ctc_loss_fn = torch.nn.CTCLoss(blank=CFG.blank_idx, zero_infinity=True, reduction="mean")

    for step, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        masks = batch["masks"].to(device)

        optimizer.zero_grad(set_to_none=True)

        amp_ctx = (
            torch.autocast(device_type="cuda")
            if use_amp and device.type == "cuda"
            else contextlib.nullcontext()
        )

        if loss_mode == "ctc":
            targets = batch["targets_concat"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            with amp_ctx:
                logits, input_lengths = model(images, masks, mode="ctc")
                # CTC expects log-probs
                log_probs = F.log_softmax(logits, dim=2)

                # 额外安全检查：避免 input_len < target_len 导致 inf -> 0
                if (input_lengths < target_lengths).any():
                    # 不直接跳过（会让你又遇到“loss=0”），这里直接报错更容易定位。
                    bad = (input_lengths < target_lengths).nonzero(as_tuple=False).view(-1).tolist()
                    raise RuntimeError(
                        "CTC input_lengths < target_lengths for some samples. "
                        f"Bad indices: {bad[:10]}. "
                        "Consider increasing image width / reducing width downsampling in CNN."
                    )

            with torch.cuda.amp.autocast(device_type=device.type, enabled=False):
                loss = ctc_loss_fn(log_probs.float(), targets, input_lengths, target_lengths)

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                optimizer.step()

        elif loss_mode == "ce":
            targets_padded = batch["targets_padded"].to(device)  # [B,L]
            with amp_ctx:
                logits = model(images, masks, mode="ce", max_len=max_label_len)  # [B,L,V]
                B, L, V = logits.shape
                loss = ce_loss_fn(logits.reshape(B * L, V), targets_padded.reshape(B * L))

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
                optimizer.step()

        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        # scheduler step（OneCycle 是按 batch step）
        if scheduler is not None and CFG.scheduler == "onecycle":
            scheduler.step()

        total_loss += float(loss.item())
        n += 1

        if step % CFG.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  step {step:04d}/{len(dataloader)}  loss={loss.item():.4f}  lr={lr:.6f}")

    return total_loss / max(1, n)


@torch.no_grad()
def validate(model: CaptchaRecognitionModel, dataloader: DataLoader, device: torch.device, *, loss_mode: str, max_label_len: int):
    # 延迟 import，避免循环依赖
    from eval import evaluate

    return evaluate(model, dataloader, device, loss_mode=loss_mode, max_label_len=max_label_len, max_examples=500)


def main() -> None:
    seed_everything(CFG.seed)

    os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    device = torch.device(CFG.device)
    print(f"Using device: {device}\n")

    # CUDA 性能设置
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # datasets
    train_dataset = CaptchaDataset(CFG.data_dir, split="train")
    val_split = "val" if os.path.isdir(os.path.join(CFG.data_dir, "val")) else "test"
    val_dataset = CaptchaDataset(CFG.data_dir, split=val_split)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples ({val_split}): {len(val_dataset)}")

    fixed_len, max_label_len = _infer_label_stats(train_dataset)
    if max_label_len <= 0:
        raise ValueError("max_label_len <= 0, please check your dataset filenames.")

    # loss mode selection
    ctc_ok = _ctc_supported_on(device)
    if CFG.loss_type == "ctc":
        if not ctc_ok:
            raise RuntimeError(f"CFG.loss_type='ctc' but CTCLoss is not supported on {device}.")
        loss_mode = "ctc"
    elif CFG.loss_type == "ce":
        if not fixed_len:
            raise RuntimeError("CFG.loss_type='ce' requires fixed-length labels, but dataset has variable lengths.")
        loss_mode = "ce"
    else:  # auto
        if ctc_ok:
            loss_mode = "ctc"
        elif fixed_len:
            print("⚠️  Falling back to fixed-length CrossEntropyLoss (mode='ce') for this device.")
            loss_mode = "ce"
        else:
            print("⚠️  CTCLoss not supported AND labels are variable-length. Falling back to CPU+CTC.")
            device = torch.device("cpu")
            loss_mode = "ctc"
            ctc_ok = True

    print(f"Selected loss_mode: {loss_mode} (device={device})\n")

    # dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=CFG.num_workers,
        pin_memory=(device.type == "cuda"),
    )

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
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    # scheduler
    scheduler = None
    if CFG.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CFG.lr,
            epochs=CFG.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0,
        )

    # AMP scaler
    use_amp = bool(CFG.use_amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    best_acc = 0.0
    print("Starting training...\n")

    for epoch in range(1, CFG.epochs + 1):
        print(f"Epoch {epoch}/{CFG.epochs}")
        print("-" * 60)
        t0 = time.time()

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            loss_mode=loss_mode,
            max_label_len=max_label_len,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_acc, examples = validate(model, val_loader, device, loss_mode=loss_mode, max_label_len=max_label_len)

        t1 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        print("\nEpoch Summary:")
        print(f"  time: {t1 - t0:.1f}s")
        print(f"  train_loss: {train_loss:.4f}")
        print(f"  val_seq_acc: {val_acc:.4f}")
        print(f"  lr: {lr_now:.6f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(CFG.checkpoint_dir, "best_model.pth"),
                meta={
                    "loss_mode": loss_mode,
                    "max_label_len": int(max_label_len),
                    "vocab": CFG.vocab,
                    "blank_idx": int(CFG.blank_idx),
                },
            )
            print(f"  ✓ Saved new best model (acc={best_acc:.4f})")

        # periodic ckpt
        if epoch % 5 == 0:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(CFG.checkpoint_dir, f"checkpoint_epoch{epoch}.pth"),
                meta={
                    "loss_mode": loss_mode,
                    "max_label_len": int(max_label_len),
                    "vocab": CFG.vocab,
                    "blank_idx": int(CFG.blank_idx),
                },
            )

        print()

    print("=" * 60)
    print(f"Training finished. Best val accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()