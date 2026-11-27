# -*- coding: utf-8 -*-
"""gradcam.py —— 手写 Grad-CAM（不引第三方包），挑 3 个有代表性的误分类样本。

公式：对目标类 logit 反传到最后一层卷积特征，w = 梯度在空间维度取均值，
cam = ReLU(Σ w·feature)，再双线性放大回 64×64。
它解释的是“模型看了哪块”，不是“物理上那块是什么”——图注里写清这句。

三个样本不是随机挑的：
    ① 高置信错误：模型最确信却错了；
    ② 易混对：真值→预测落在混淆矩阵最重的那一格里；
    ③ 边界样本：真值概率与预测概率最接近的错误。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn import functional as F

from cnn import CLASSES, PatchDS, SmallCNN

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
SIZE = 64


def compute_cam(model: SmallCNN, img_t: torch.Tensor, target: int) -> np.ndarray:
    """最后一层卷积特征 × 目标类梯度均值 → ReLU → 双线性放大到 64×64。"""
    holder: dict = {}

    def hook(_m, _i, out):
        out.retain_grad()
        holder["act"] = out

    layer = model.body[-1][2]  # 第 4 块的 ReLU，输出 128×8×8
    h = layer.register_forward_hook(hook)
    logits = model(img_t.unsqueeze(0))
    model.zero_grad(set_to_none=True)
    logits[0, target].backward()
    h.remove()

    act = holder["act"][0].detach()
    grad = holder["act"].grad[0]
    w = grad.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((w * act).sum(0)).numpy()
    cam = cam / (cam.max() + 1e-8)
    cam_t = torch.from_numpy(cam)[None, None]
    return F.interpolate(cam_t, size=(SIZE, SIZE), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def main() -> None:
    splits = pd.read_csv(BUILD / "splits.csv", encoding="utf-8-sig")
    pred = pd.read_csv(BUILD / "pred_cnn_final.csv", encoding="utf-8-sig")
    d = pred[pred["split"] == "test"].reset_index(drop=True)
    err = d[d["label"] != d["pred"]].copy()

    picks = []
    a = err.sort_values("p_top", ascending=False).iloc[0]
    picks.append(("高置信错误", a))
    pairs = json.loads((BUILD / "eval.json").read_text(encoding="utf-8"))
    top = pairs["models"]["cnn"]["top_confusions"][0]
    b_cand = err[(err["label"] == top["true"]) & (err["pred"] == top["pred"])]
    b = b_cand.sort_values("p_top", ascending=False).iloc[0] if len(b_cand) else err.iloc[1]
    picks.append((f"易混对 {top['true']}→{top['pred']}", b))
    c = err.assign(margin=err["p_top"] - 1 / 8).sort_values("p_top").iloc[0]
    picks.append(("边界样本", c))

    model = SmallCNN()
    model.load_state_dict(torch.load(BUILD / "cnn_final.pt", map_location="cpu"))
    model.eval()

    ds = PatchDS(splits[splits["split"] == "test"].reset_index(drop=True))
    idx_map = {p: i for i, p in enumerate(ds.paths)}

    payload, rows = [], []
    for tag, row in picks:
        i = idx_map[ROOT / row["path"]]
        img_t, _ = ds[i]
        cam = compute_cam(model, img_t, CLASSES.index(row["pred"]))
        img = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        payload.append({"path": row["path"], "cam": cam, "img": img})
        rows.append({"tag": tag, "path": row["path"], "true": row["label"],
                     "pred": row["pred"], "p_top": round(float(row["p_top"]), 3),
                     "cam_peak_xy": [int(np.unravel_index(cam.argmax(), cam.shape)[1]),
                                     int(np.unravel_index(cam.argmax(), cam.shape)[0])]})

    np.savez_compressed(BUILD / "gradcam.npz",
                        **{f"s{i}": p["cam"] for i, p in enumerate(payload)},
                        **{f"i{i}": p["img"] for i, p in enumerate(payload)})
    (BUILD / "gradcam_picks.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    for r in rows:
        print(f"{r['tag']:<22} 真值 {r['true']:<13} 预测 {r['pred']:<13} p={r['p_top']:.2f} "
              f"热力峰值 @{r['cam_peak_xy']}")


if __name__ == "__main__":
    main()

