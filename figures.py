# -*- coding: utf-8 -*-
"""figures.py —— 四张图：样例 / 混淆矩阵 / 逐类 F1 对照 / Grad-CAM。

图注按 RV-16：图号＋题名居中置图下，整图注在图题下方、左对齐缩进 2 字；
中文宋体 + 西文 Times New Roman；信息类走专业制图风，色阶语义化、灰度可辨。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from PIL import Image

import _caption as C

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

CLASSES = ["AnnualCrop", "Forest", "Industrial", "Pasture",
           "PermanentCrop", "Residential", "River", "SeaLake"]
CN = {"AnnualCrop": "一年生作物", "Forest": "森林", "Industrial": "工业区",
      "Pasture": "牧场", "PermanentCrop": "多年生作物", "Residential": "住宅区",
      "River": "河流", "SeaLake": "湖海"}
MAP_DATE = "2026-11-27"
COOL, WARM, NEUT = "#2F5C8A", "#B4483C", "#5E6B75"


def setup_fonts() -> None:
    plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 300


def fp_cn(size: float) -> font_manager.FontProperties:
    return font_manager.FontProperties(family="SimSun", size=size)


def fp_en(size: float) -> font_manager.FontProperties:
    return font_manager.FontProperties(family="Times New Roman", size=size)


def fig_samples(splits: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(4, 8, figsize=(10.6, 6.9))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.90, bottom=0.20,
                        wspace=0.06, hspace=0.10)
    for c, cls in enumerate(CLASSES):
        sub = splits[splits["label"] == cls].sort_values("path").head(4)
        for r in range(4):
            ax = axes[r, c]
            ax.imshow(Image.open(ROOT / sub.iloc[r]["path"]).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#B9C2C9")
                s.set_linewidth(0.6)
        pos = axes[0, c].get_position()
        cx = (pos.x0 + pos.x1) / 2
        C.mix(fig, cx, pos.y1 + 0.026, CN[cls], fp_cn(8.8), fp_en(8.8), ha="center")
        C.mix(fig, cx, pos.y1 + 0.008, cls, fp_cn(8.8), fp_en(8.8), ha="center")
    C.add_caption(fig, 1, "欧洲 34 国 Sentinel-2 影像的八类样例", [
        "每类取 4 张，源数据每类 500 张（seed 42 抽样）；图像为 Sentinel-2 的 64×64 像元块，10 m 分辨率",
        "三组刻意保留的易混对：一年生作物与多年生作物、住宅区与工业区、河流与湖海",
        f"据 EuroSAT RGB（Helber et al., 2019, IEEE JSTARS；CC BY 4.0）绘制，制图 {MAP_DATE}",
    ], lb=0.035, rb=0.985)
    out = DOCS / "fig_samples.png"
    fig.savefig(out, dpi=300, bbox_inches=None)
    plt.close(fig)
    return out


def fig_confusion(ev: dict) -> Path:
    cm = np.array(ev["models"]["cnn"]["confusion_norm"])
    fig, ax = plt.subplots(figsize=(7.8, 7.4))
    fig.subplots_adjust(left=0.16, right=0.90, top=0.985, bottom=0.30)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.set_xticklabels([CN[c] for c in CLASSES], rotation=45, ha="right",
                       fontproperties=font_manager.FontProperties(family="SimSun", size=8.4))
    ax.set_yticklabels([CN[c] for c in CLASSES],
                       fontproperties=font_manager.FontProperties(family="SimSun", size=8.4))
    for i in range(8):
        for j in range(8):
            v = cm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        fontproperties=font_manager.FontProperties(family="Times New Roman", size=8.2),
                        color=("#f7f7f7" if v > 0.55 else "#1a1a1a"))
    ax.set_xlabel("预测类", fontproperties=font_manager.FontProperties(family="SimSun", size=9))
    ax.set_ylabel("真实类", fontproperties=font_manager.FontProperties(family="SimSun", size=9))
    ax.set_xticks(np.arange(-0.5, 8, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.9)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.ax.tick_params(labelsize=7.5)
    cpos = cb.ax.get_position()
    C.mix(fig, cpos.x1 + 0.030, (cpos.y0 + cpos.y1) / 2, "行归一比例 / %",
          fp_cn(8.4), fp_en(8.4), ha="center", rotation=90)
    C.add_caption(fig, 2, "欧洲 34 国 Sentinel-2 影像的八类分类混淆矩阵（测试集）", [
        "数值为真实类别的行归一比例（%），对角为召回率；测试集 600 张，八类各 75 张",
        f"模型为自建 4-block CNN（24.2 万参数），训练 {ev['timing']['cnn']['epochs']} 轮、"
        f"最优轮次取验证集准确率，实测总耗时 {ev['timing']['cnn']['total_seconds']/60:.1f} min（CPU）",
        f"据 EuroSAT RGB（Helber et al., 2019；CC BY 4.0）绘制，制图 {MAP_DATE}",
    ], lb=0.16, rb=0.90)
    out = DOCS / "fig_confusion.png"
    fig.savefig(out, dpi=300, bbox_inches=None)
    plt.close(fig)
    return out


def fig_f1(ev: dict) -> Path:
    fig, ax = plt.subplots(figsize=(10.6, 5.8))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.26)
    series = [("手工特征 + 随机森林", "baseline", NEUT, "o"),
              ("自建 4-block CNN", "cnn", COOL, "s"),
              ("ResNet18 冻结主干", "resnet18", WARM, "^")]
    x = np.arange(len(CLASSES))
    for label, key, color, marker in series:
        y = [ev["models"][key]["per_class_f1"][c] for c in CLASSES]
        ax.plot(x, y, marker=marker, color=color, lw=1.4, ms=5.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([CN[c] for c in CLASSES], fontproperties=fp_cn(8.8))
    ax.set_xlim(-0.4, len(CLASSES) - 0.6)
    ax.set_ylim(0.55, 1.0)
    ax.grid(axis="y", color="#E3E7EA", lw=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    pos = ax.get_position()
    # 纵轴标题：set_ylabel 只能给一种字体，中文与“F1”必须分段画
    C.mix(fig, pos.x0 - 0.055, (pos.y0 + pos.y1) / 2, "逐类 F1", fp_cn(9.0), fp_en(9.0),
          ha="center", rotation=90)
    # 图例同样分段画：先量三组标签的宽度，再按列摆
    y_leg = pos.y0 + 0.035
    widths = [C.size_pt(fig, lab, fp_cn(8.6), fp_en(8.6)) for lab, *_ in series]
    seg, gap = 0.030, 0.020
    x0 = pos.x0 + 0.008
    for (label, _key, color, marker), wlab in zip(series, widths):
        line = Line2D([x0, x0 + seg], [y_leg, y_leg], transform=fig.transFigure,
                      color=color, lw=1.4, marker=marker, markersize=5.2, markevery=[1])
        fig.add_artist(line)
        C.mix(fig, x0 + seg + 0.008, y_leg - 0.006, label, fp_cn(8.6), fp_en(8.6))
        x0 += seg + 0.008 + wlab + gap
    C.add_caption(fig, 3, "欧洲 Sentinel-2 影像上三类模型的逐类 F1 对比", [
        "同一份划分、同一测试集 600 张；点线只连同一模型的八类结果，未做插值",
        f"macro-F1：手工特征基线 {ev['models']['baseline']['macro_f1']:.3f}，"
        f"自建 CNN {ev['models']['cnn']['macro_f1']:.3f}（{ev['macro_f1_delta_vs_baseline']['cnn']*100:+.1f} pp），"
        f"ResNet18 {ev['models']['resnet18']['macro_f1']:.3f}"
        f"（{ev['macro_f1_delta_vs_baseline']['resnet18']*100:+.1f} pp）",
        "ResNet18 为 ImageNet 预训练主干冻结后仅训线性分类头，用于把“先验”与“调参”分开",
        f"据 EuroSAT RGB（Helber et al., 2019；CC BY 4.0）绘制，制图 {MAP_DATE}",
    ], lb=0.085, rb=0.985)
    out = DOCS / "fig_baseline_vs_cnn.png"
    fig.savefig(out, dpi=300, bbox_inches=None)
    plt.close(fig)
    return out


def fig_gradcam(picks: list[dict]) -> Path:
    npz = np.load(BUILD / "gradcam.npz")
    fig, axes = plt.subplots(3, 3, figsize=(9.0, 6.5))
    fig.subplots_adjust(left=0.03, right=0.985, top=0.94, bottom=0.30,
                        wspace=0.04, hspace=0.13)
    letters = ["a", "b", "c"]
    for r, p in enumerate(picks):
        img, cam = npz[f"i{r}"], npz[f"s{r}"]
        axes[r, 0].imshow(img)
        axes[r, 1].imshow(cam, cmap="magma")
        axes[r, 2].imshow(img)
        axes[r, 2].imshow(cam, cmap="magma", alpha=0.55)
        if r == 0:
            for c, name in enumerate(["原图", "Grad-CAM 热力", "叠加"]):
                cpos = axes[r, c].get_position()
                C.mix(fig, (cpos.x0 + cpos.x1) / 2, cpos.y1 + 0.012, name,
                      fp_cn(8.8), fp_en(8.8), ha="center")
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for s in axes[r, c].spines.values():
                s.set_color("#B9C2C9")
                s.set_linewidth(0.6)
        axes[r, 0].text(0.06, 0.94, letters[r], transform=axes[r, 0].transAxes,
                        va="top", ha="left",
                        fontproperties=font_manager.FontProperties(family="Times New Roman", size=10),
                        color="#ffffff",
                        bbox={"boxstyle": "square,pad=0.18", "facecolor": "#1a1a1a", "edgecolor": "none"})
    desc = "；".join(f"{letters[i]} {p['tag']}：真值{CN[p['true']]}，预测{CN[p['pred']]}"
                     f"，p = {p['p_top']:.2f}" for i, p in enumerate(picks))
    C.add_caption(fig, 4, "欧洲 Sentinel-2 影像误分类样本的 Grad-CAM 叠加", [
        desc,
        "热力越高表示该像元对预测类 logit 的贡献越大；梯度取自第 4 个卷积块的 ReLU 输出（8×8），再双线性放大",
        "原图与热力均为 64×64 像元的块（10 m 分辨率，约 640 m 见方），本图按约 6 倍放大显示，细部偏软属数据分辨率所限；"
        "热力另经 8×8→64 的一次插值，故比原图更平滑",
        f"据 EuroSAT RGB（Helber et al., 2019；CC BY 4.0）与自训权重绘制，制图 {MAP_DATE}",
    ], lb=0.03, rb=0.985)
    out = DOCS / "fig_gradcam.png"
    fig.savefig(out, dpi=300, bbox_inches=None)
    plt.close(fig)
    return out


def main() -> None:
    setup_fonts()
    ev = json.loads((BUILD / "eval.json").read_text(encoding="utf-8"))
    splits = pd.read_csv(BUILD / "splits.csv", encoding="utf-8-sig")
    picks = json.loads((BUILD / "gradcam_picks.json").read_text(encoding="utf-8"))
    outs = [fig_samples(splits), fig_confusion(ev), fig_f1(ev), fig_gradcam(picks)]
    for p in outs:
        print(f"出图 {p.name} {p.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()









