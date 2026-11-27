# -*- coding: utf-8 -*-
"""cnn.py —— 自建 4-block 小 CNN，CPU 上训 3 epoch。

网络（参数量 < 1M，实测见 model_card）：
    Conv(3→32,k3)-BN-ReLU-MaxPool → Conv(32→64)-BN-ReLU-MaxPool
    → Conv(64→128)-BN-ReLU-MaxPool → Conv(128→128)-BN-ReLU-MaxPool
    → GAP → Dropout(0.2) → Linear(128, 8)

增强只做水平 / 垂直翻转：遥感 patch 无固定朝向，而旋转与色彩抖动会改变光谱统计，
与“颜色直方图”那条基线冲突，不能一边改光谱一边说 CNN 赢在手艺。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
CLASSES = ["AnnualCrop", "Forest", "Industrial", "Pasture",
           "PermanentCrop", "Residential", "River", "SeaLake"]
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PatchDS(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = False):
        self.paths = [ROOT / p for p in df["path"]]
        self.y = [CLASSES.index(l) for l in df["label"]]
        tf = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()] if train else []
        self.tf = transforms.Compose(tf + [transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.y[i]


def conv_bn(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class SmallCNN(nn.Module):
    def __init__(self, n_cls: int = 8, use_bn: bool = True, flatten: bool = False):
        super().__init__()
        def blk(i, o):
            layers = [nn.Conv2d(i, o, 3, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(o))
            layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            return nn.Sequential(*layers)
        self.body = nn.Sequential(blk(3, 32), blk(32, 64), blk(64, 128), blk(128, 128))
        self.flatten = flatten
        head_in = 128 * 4 * 4 if flatten else 128
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(head_in, n_cls))

    def forward(self, x):
        f = self.body(x)
        f = f.reshape(f.size(0), -1) if self.flatten else f.mean(dim=(2, 3))
        return self.head(f)


def run_epoch(model, loader, crit, opt=None):
    model.train() if opt is not None else model.eval()
    loss_sum, n, correct = 0.0, 0, 0
    torch.set_grad_enabled(opt is not None)
    for xb, yb in loader:
        logits = model(xb)
        loss = crit(logits, yb)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        loss_sum += loss.item() * len(yb)
        n += len(yb)
        correct += (logits.argmax(1) == yb).sum().item()
    return loss_sum / n, correct / n


def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ps, ys = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            ps.append(torch.softmax(logits, 1).numpy())
            ys.append(yb.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--no-bn", action="store_true")
    ap.add_argument("--flatten", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    set_seed()

    splits = pd.read_csv(BUILD / "splits.csv", encoding="utf-8-sig")
    tr_df = splits[splits["split"] == "train"]
    va_df = splits[splits["split"] == "val"]
    te_df = splits[splits["split"] == "test"]
    dl_tr = DataLoader(PatchDS(tr_df, train=True), batch_size=64, shuffle=True, num_workers=0)
    dl_va = DataLoader(PatchDS(va_df), batch_size=128, shuffle=False, num_workers=0)
    dl_te = DataLoader(PatchDS(te_df), batch_size=128, shuffle=False, num_workers=0)

    model = SmallCNN(use_bn=not args.no_bn, flatten=args.flatten)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[{args.tag}] 参数量 {n_par/1e3:.1f} K（上限 1 M）")
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    log, t0 = [], time.perf_counter()
    best = {"val_acc": -1.0, "epoch": 0, "state": None}
    for ep in range(1, args.epochs + 1):
        te0 = time.perf_counter()
        tr_loss, tr_acc = run_epoch(model, dl_tr, crit, opt)
        va_loss, va_acc = run_epoch(model, dl_va, crit, None)
        dt = time.perf_counter() - te0
        if va_acc > best["val_acc"]:
            best = {"val_acc": va_acc, "epoch": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        log.append({"tag": args.tag, "epoch": ep, "train_loss": round(tr_loss, 4),
                    "train_acc": round(tr_acc, 4), "val_loss": round(va_loss, 4),
                    "val_acc": round(va_acc, 4), "epoch_seconds": round(dt, 1),
                    "lr": args.lr, "params_k": round(n_par / 1e3, 1)})
        print(f"  epoch {ep}: train_loss {tr_loss:.3f} train_acc {tr_acc:.3f} "
              f"val_loss {va_loss:.3f} val_acc {va_acc:.3f}  {dt:.0f}s")
    total = time.perf_counter() - t0
    print(f"  最佳 val_acc {best['val_acc']:.3f} @ epoch {best['epoch']}（按最佳轮次存档）")

    log_path = BUILD / "train_log.csv"
    new = pd.DataFrame(log)
    if log_path.exists():
        old = pd.read_csv(log_path, encoding="utf-8-sig")
        new = pd.concat([old[old["tag"] != args.tag], new], ignore_index=True)
    new.to_csv(log_path, index=False, encoding="utf-8-sig")

    model.load_state_dict(best["state"])
    torch.save(model.state_dict(), BUILD / f"cnn_{args.tag}.pt")
    probs_te, ys_te = predict(model, dl_te)
    out = te_df[["path", "label", "split"]].copy().reset_index(drop=True)
    out["pred"] = [CLASSES[i] for i in probs_te.argmax(1)]
    out["p_top"] = probs_te.max(1)
    out.to_csv(BUILD / f"pred_cnn_{args.tag}.csv", index=False, encoding="utf-8-sig")

    meta = {"tag": args.tag, "epochs": args.epochs, "lr": args.lr, "batch": 64,
            "params": int(n_par), "params_k": round(n_par / 1e3, 1),
            "total_seconds": round(total, 1), "use_bn": not args.no_bn,
            "pool": "flatten" if args.flatten else "gap", "threads": args.threads,
            "epoch_seconds": [r["epoch_seconds"] for r in log], "best_epoch": best["epoch"], "best_val_acc": round(best["val_acc"], 4)}
    (BUILD / f"cnn_meta_{args.tag}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{args.tag}] 总耗时 {total/60:.1f} min（预算 20 min）")


if __name__ == "__main__":
    main()


