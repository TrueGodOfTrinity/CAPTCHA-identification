"""configs.py

项目全局配置。

修复点：
1) CTC blank 类别数与 blank_idx 对齐（num_classes = len(vocab) + 1）。
2) 设备选择更符合常见习惯：优先 CUDA，其次 MPS，最后 CPU。
3) 增加训练相关的默认项（梯度裁剪、AMP、scheduler 等），并保持可跨 CUDA/MPS。
"""

from __future__ import annotations

import os
import torch


class Config:
    def __init__(self):
        # =====================
        # Paths
        # =====================
        self.data_dir = "./data"
        self.checkpoint_dir = "./checkpoints"
        self.eval_ckpt = os.path.join(self.checkpoint_dir, "best_model.pth")

        # =====================
        # Input
        # =====================
        self.in_channels = 3
        self.img_height = 64

        # =====================
        # Model
        # =====================
        self.cnn_out = 256
        self.trans_d_model = 256
        self.trans_layers = 4
        self.nhead = 8
        self.dropout = 0.1

 
        digits = [str(i) for i in range(10)]
        lowers = [chr(ord("a") + i) for i in range(26)]
        uppers = [chr(ord("A") + i) for i in range(26)]
        self.vocab = digits + lowers + uppers

        # CTC blank index 放在 vocab 的末尾
        self.blank_idx = len(self.vocab)
        # CTC 的类别数必须包含 blank
        self.num_classes_ctc = len(self.vocab) + 1

        self.char2idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char = {i: c for i, c in enumerate(self.vocab)}

        # =====================
        # Training
        # =====================
        self.batch_size = 32
        self.epochs = 30

        self.lr = 2e-3
        self.weight_decay = 0.01
        self.grad_clip = 5.0


        self.num_workers = 0


        self.loss_type = "auto"

        # Scheduler: "onecycle" | "none"
        self.scheduler = "onecycle"

        self.use_amp = True

        self.seed = 42
        self.log_every = 10

        # =====================
        # Device
        # =====================
        self.device = self._get_device()

    @staticmethod
    def _get_device() -> str:
        """自动检测可用的设备（优先 CUDA，其次 MPS）。"""
        if torch.cuda.is_available():
            print(" Using CUDA (NVIDIA GPU)")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon GPU)")
            return "mps"
        print(" Using CPU (No GPU available)")
        return "cpu"


CFG = Config()