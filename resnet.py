# -*- coding: utf-8 -*-
"""resnet.py —— 可选对照：ImageNet 预训练 ResNet18，冻结主干 + 线性探针。

这是“样本量与先验”的对照：8 类共 4000 张、CPU 上只训 3–10 epoch 的小网络，
比不过预训练主干上已经学好的通用纹理与边缘。冻结主干、只训分类头，
既省时间（单次前向提取特征，随后 sklearn 一步拟合），也把对比点落在“先验”上，
而不是“谁的调参更狠”。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
CLASSES = ["AnnualCrop", "Forest", "Industrial", "Pasture",
           "PermanentCrop", "Residential", "River", "SeaLake"]


def main() -> None:
    from torchvision import models, transforms
    torch.set_num_threads(8)
    w = models.ResNet18_Weights.DEFAULT
    net = models.resnet18(weights=w)
    net.fc = nn.Identity()
    net.eval()

    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize(mean=w.transforms().mean,
                                                  std=w.transforms().std)])
    splits = pd.read_csv(BUILD / "splits.csv", encoding="utf-8-sig")
    t0 = time.perf_counter()
    feats = []
    with torch.no_grad():
        for i in range(0, len(splits), 64):
            batch = splits["path"].iloc[i:i + 64]
            x = torch.stack([tf(Image.open(ROOT / p).convert("RGB")) for p in batch])
            feats.append(net(x).numpy())
    X = np.vstack(feats)
    feat_s = time.perf_counter() - t0
    y = splits["label"].to_numpy()
    tr = (splits["split"] == "train").to_numpy()
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(X[tr], y[tr])

    out = splits[["path", "label", "split"]].copy()
    p = clf.predict_proba(X)
    out["pred"] = clf.classes_[p.argmax(1)]
    out["p_top"] = p.max(1)
    out.to_csv(BUILD / "pred_resnet.csv", index=False, encoding="utf-8-sig")

    meta = {"model": "resnet18 (ImageNet, frozen) + LogisticRegression",
            "feature_dim": int(X.shape[1]), "feature_seconds": round(feat_s, 1),
            "n_train": int(tr.sum()),
            "note": "主干冻结，只训线性分类头；特征一次前向提取"}
    (BUILD / "resnet_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"ResNet18 冻结主干：特征 {feat_s:.0f} s，{X.shape[1]} 维，训练 {tr.sum()} 张")


if __name__ == "__main__":
    main()

