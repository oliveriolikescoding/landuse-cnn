# -*- coding: utf-8 -*-
"""run_all.py —— 一条命令跑完四段：prepare → baseline → cnn → eval（图另算）。

用法：
    python run_all.py                    # 全跑（清空 build/ 也能一条命令跑完）
    python run_all.py --skip-train       # 复用 build/cnn_final.pt，只重算指标与图
    python run_all.py --skip-figures     # 只要数字，不出图
    python run_all.py --stages prepare,baseline
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
PY = sys.executable
FIGURES = ["fig_samples.png", "fig_confusion.png", "fig_baseline_vs_cnn.png", "fig_gradcam.png"]


def run(script: str, *args: str) -> None:
    cmd = [PY, str(ROOT / script), *args]
    print(f"\n=== {script} {' '.join(args)} ===")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"{script} 失败，退出码 {r.returncode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="prepare,baseline,cnn,resnet,eval,gradcam,figures")
    ap.add_argument("--skip-train", action="store_true", help="跳过 cnn / resnet 训练段")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.skip_figures:
        stages = [s for s in stages if s != "figures"]
    if args.skip_train:
        stages = [s for s in stages if s not in ("cnn", "resnet")]

    for s in stages:
        if s == "prepare":
            run("prepare.py")
        elif s == "baseline":
            run("baseline.py")
        elif s == "cnn":
            run("cnn.py", "--epochs", "10", "--lr", "5e-4", "--tag", "final")
        elif s == "resnet":
            run("resnet.py")
        elif s == "eval":
            run("eval.py", "--cnn-tag", "final", "--extra",
                "cnn_3ep_lr1e3=pred_cnn_main.csv", "cnn_3ep_lr5e4=pred_cnn_lr5e4.csv",
                "resnet18=pred_resnet.csv")
        elif s == "gradcam":
            run("gradcam.py")
        elif s == "figures":
            run("figures.py")
        else:
            raise SystemExit(f"未知阶段：{s}")

    # —— 收尾校验：划分行数、指标文件、四张图在位
    splits = (BUILD / "splits.csv").read_text(encoding="utf-8-sig").strip().splitlines()
    n = len(splits) - 1
    print(f"\n划分行数 {n}（应为 4000）")
    assert n == 4000, "splits.csv 行数不对，划分可能被重算过"

    ev = json.loads((BUILD / "eval.json").read_text(encoding="utf-8"))
    print(f"{'模型':<22}{'acc':>8}{'macroF1':>9}{'κ':>8}")
    for k, v in ev["models"].items():
        print(f"{k:<22}{v['accuracy']:>8.3f}{v['macro_f1']:>9.3f}{v['kappa']:>8.3f}")

    miss = [f for f in FIGURES if not (ROOT / "docs" / f).exists()]
    print("四张图在位" if not miss else f"缺图：{miss}")
    print("\n完成。")


if __name__ == "__main__":
    main()
