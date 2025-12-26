"""dataset.py
验证码数据集读取。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from configs import CFG


def _label_from_filename(filename: str) -> str:
    """从文件名提取验证码文本。

    规则：
    - 先去掉扩展名。
    - 如果 stem 里存在 '_'，且最后一段全是数字（常见的索引后缀），则去掉这段。
      例如："aB12_0003.png" -> "aB12"。
    - 否则使用整个 stem。
    """

    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        stem = "_".join(parts[:-1])
    return stem.strip()


class CaptchaDataset(Dataset):
    def __init__(self, data_dir: str = "./data", split: str = "train", transform: Optional[Callable] = None):
        self.root = os.path.join(data_dir, split)
        if not os.path.isdir(self.root):
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

        self.files = [
            f
            for f in os.listdir(self.root)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.files.sort()

        self.transform = transform

        # 预先解析 labels（不读图，只解析文件名）
        self.labels: List[str] = [_label_from_filename(f) for f in self.files]

        # 简单健壮性检查：空 label 直接报错
        bad = [f for f, l in zip(self.files, self.labels) if len(l) == 0]
        if bad:
            raise ValueError(
                "Found empty labels (filename stem is empty). Examples: " + ", ".join(bad[:5])
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fn = self.files[idx]
        path = os.path.join(self.root, fn)

        img = Image.open(path).convert("RGB")

        # 按固定高度缩放，宽度按比例变化（可变宽输入）
        w, h = img.size
        new_h = CFG.img_height
        new_w = max(1, int(round(w * (new_h / max(1, h)))))
        if (new_w, new_h) != (w, h):
            img = img.resize((new_w, new_h), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[idx]
        return {"image": img, "label": label}
