# -*- coding: utf-8 -*-
"""eval.py —— 测试集指标：总体准确率 / macro-F1 / 逐类 F1 / 混淆矩阵 / Cohen's κ。

只吃 build/ 下已落盘的预测文件（基线、CNN 同表），不在这里重训、不在这里重新划分：
两套实验共用 build/splits.csv，才谈得上“CNN 相对基线赢了多少”。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score)

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
CLASSES = ["AnnualCrop", "Forest", "Industrial", "Pasture",
           "PermanentCrop", "Residential", "River", "SeaLake"]


def metrics(pred: pd.DataFrame, split: str = "test") -> dict:
    d = pred[pred["split"] == split]
    y, yh = d["label"].to_numpy(), d["pred"].to_numpy()
    cm = confusion_matrix(y, yh, labels=CLASSES)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    rep = classification_report(y, yh, labels=CLASSES, digits=3, output_dict=True, zero_division=0)
    off = [[CLASSES[i], CLASSES[j], int(cm[i, j])]
           for i in range(len(CLASSES)) for j in range(len(CLASSES)) if i != j and cm[i, j] > 0]
    off.sort(key=lambda r: -r[2])
    return {"n": int(len(d)),
            "accuracy": round(float(accuracy_score(y, yh)), 4),
            "macro_f1": round(float(f1_score(y, yh, average="macro")), 4),
            "weighted_f1": round(float(f1_score(y, yh, average="weighted")), 4),
            "kappa": round(float(cohen_kappa_score(y, yh)), 4),
            "per_class_f1": {c: round(float(rep[c]["f1-score"]), 4) for c in CLASSES},
            "per_class_recall": {c: round(float(rep[c]["recall"]), 4) for c in CLASSES},
            "support": {c: int(rep[c]["support"]) for c in CLASSES},
            "confusion": cm.tolist(),
            "confusion_norm": np.round(cm_norm, 4).tolist(),
            "top_confusions": [{"true": a, "pred": b, "n": n} for a, b, n in off[:5]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn-tag", default="main")
    ap.add_argument("--extra", nargs="*", default=[], help="额外预测文件，形如 name=path")
    args = ap.parse_args()

    splits = pd.read_csv(BUILD / "splits.csv", encoding="utf-8-sig")
    out: dict = {"classes": CLASSES,
                 "split_sizes": {k: int(v) for k, v in splits["split"].value_counts().items()},
                 "models": {}}

    base = pd.read_csv(BUILD / "pred_baseline.csv", encoding="utf-8-sig")
    out["models"]["baseline"] = metrics(base)
    cnn = pd.read_csv(BUILD / f"pred_cnn_{args.cnn_tag}.csv", encoding="utf-8-sig")
    out["models"]["cnn"] = metrics(cnn)
    for kv in args.extra:
        name, path = kv.split("=", 1)
        out["models"][name] = metrics(pd.read_csv(BUILD / path, encoding="utf-8-sig"))

    timing = json.loads((BUILD / "baseline_timing.json").read_text(encoding="utf-8"))
    out["timing"] = {"baseline": timing}
    meta_path = BUILD / f"cnn_meta_{args.cnn_tag}.json"
    if meta_path.exists():
        out["timing"]["cnn"] = json.loads(meta_path.read_text(encoding="utf-8"))

    d = {k: v["macro_f1"] - out["models"]["baseline"]["macro_f1"] for k, v in out["models"].items()}
    out["macro_f1_delta_vs_baseline"] = {k: round(v, 4) for k, v in d.items()}

    (BUILD / "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'模型':<10}{'acc':>8}{'macroF1':>9}{'κ':>8}  最差类")
    for k, v in out["models"].items():
        worst = min(v["per_class_f1"], key=v["per_class_f1"].get)
        print(f"{k:<10}{v['accuracy']:>8.3f}{v['macro_f1']:>9.3f}{v['kappa']:>8.3f}  "
              f"{worst} F1={v['per_class_f1'][worst]:.3f}")


if __name__ == "__main__":
    main()

