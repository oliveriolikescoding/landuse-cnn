# -*- coding: utf-8 -*-
"""baseline.py —— 手工特征 + 随机森林，本周的地板。

特征（共 60 维）：
    · HSV 颜色直方图：三通道各 16 bin（H 用 180 度归一化后的 0–255 值），48 维；
    · 灰度共生矩阵纹理：32 灰度级、距离 1、4 个方向（0/45/90/135），
      取 contrast / homogeneity / energy 三个统计量，12 维。
模型：RandomForest(n_estimators=200, random_state=42)。

为什么先跑它：CNN 的提升一律相对这条地板说，不许相对 50% 说。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
CLASSES = ["AnnualCrop", "Forest", "Industrial", "Pasture",
           "PermanentCrop", "Residential", "River", "SeaLake"]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
PROPS = ["contrast", "homogeneity", "energy"]
BINS = 16
LEVELS = 32


def features(path: Path) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB"))
    hsv = np.asarray(Image.open(path).convert("HSV"))
    feats = []
    for c in range(3):
        h, _ = np.histogram(hsv[..., c], bins=BINS, range=(0, 256), density=False)
        feats.append(h / max(h.sum(), 1))
    feats = np.concatenate(feats)
    g = (rgb2gray(img) * (LEVELS - 1)).astype(np.uint8)
    glcm = graycomatrix(g, distances=[1], angles=ANGLES, levels=LEVELS,
                        symmetric=True, normed=True)
    tex = np.concatenate([graycoprops(glcm, p).ravel() for p in PROPS])
    return np.concatenate([feats, tex])


def build_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([features(ROOT / p) for p in df["path"]])


def main() -> None:
    splits = pd.read_csv(BUILD / "splits.csv", encoding="utf-8-sig")
    t0 = time.perf_counter()
    X = build_matrix(splits)
    feat_s = time.perf_counter() - t0
    y = splits["label"].to_numpy()

    tr = splits["split"] == "train"
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    t1 = time.perf_counter()
    clf.fit(X[tr], y[tr])
    fit_s = time.perf_counter() - t1

    out = splits[["path", "label", "split"]].copy()
    p = clf.predict_proba(X)[:, :]
    out["pred"] = clf.classes_[p.argmax(axis=1)]
    out["p_top"] = p.max(axis=1)
    out.to_csv(BUILD / "pred_baseline.csv", index=False, encoding="utf-8-sig")

    timing = {"feature_seconds": round(feat_s, 1), "fit_seconds": round(fit_s, 1),
              "n_features": int(X.shape[1]), "n_train": int(tr.sum()),
              "model": "RandomForest(200, random_state=42)"}
    (BUILD / "baseline_timing.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8")
    print("基线：特征 %.1f s，拟合 %.1f s，%d 维，训练 %d 张"
          % (feat_s, fit_s, X.shape[1], tr.sum()))


if __name__ == "__main__":
    main()
