# -*- coding: utf-8 -*-
"""prepare.py —— 抽 8 类 × 500 张，落盘 + 一次性划分。

数据：EuroSAT RGB（Sentinel-2，10 m，64×64，Helber et al. 2019，CC BY 4.0）。
本地副本取自 Hugging Face 镜像 `blanchon/EuroSAT_RGB` 的三份 parquet（train/validation/test），
三份先合并成一个池再抽样——不沿用镜像自带划分，避免与自己的 70/15/15 口径混淆。

口径（写进 README 与 model_card，改口径先改这里）：
    · 每类 500 张，random_state=42，类别完全平衡；
    · 70/15/15 分层划分，**只划分一次**，落 build/splits.csv，基线与 CNN 共用；
    · 抽样结果落 data/eurosat_subset/<类>/，供复现，不进 .gitignore。
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
DATA = ROOT / "data" / "eurosat_subset"
BUILD = ROOT / "build"
BUILD.mkdir(exist_ok=True)

CLASSES = ["AnnualCrop", "Forest", "Industrial", "Pasture",
           "PermanentCrop", "Residential", "River", "SeaLake"]
CN = {"AnnualCrop": "一年生作物", "Forest": "森林", "Industrial": "工业区",
      "Pasture": "牧场", "PermanentCrop": "多年生作物", "Residential": "住宅区",
      "River": "河流", "SeaLake": "湖海"}
PER_CLASS = 500
SEED = 42


def read_pool() -> pd.DataFrame:
    """三份 parquet 合成一个池，返回 filename / label / image_bytes。"""
    rows = []
    for f in ["train", "validation", "test"]:
        p = RAW / f"{f}.parquet"
        t = pq.read_table(p, columns=["image", "label", "filename"]).to_pandas()
        t["src_split"] = f
        rows.append(t)
    pool = pd.concat(rows, ignore_index=True)
    pool["cls"] = pool["label"].map(lambda i: CLASSES_OF_10[i])
    return pool


CLASSES_OF_10 = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
                 "Industrial", "Pasture", "PermanentCrop", "Residential",
                 "River", "SeaLake"]


def decode(cell) -> Image.Image:
    if isinstance(cell, dict):
        if cell.get("bytes") is not None:
            return Image.open(io.BytesIO(cell["bytes"]))
        return Image.open(cell["path"])
    raise TypeError(f"未知的图像字段类型: {type(cell)}")


def sample_and_save(pool: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.RandomState(SEED)
    records = []
    recorded = {d.name for d in DATA.glob("*")} if DATA.exists() else set()
    for cls in CLASSES:
        sub = pool[pool["cls"] == cls].sort_values("filename").reset_index(drop=True)
        idx = rng.choice(len(sub), size=PER_CLASS, replace=False)
        idx = np.sort(idx)
        outdir = DATA / cls
        outdir.mkdir(parents=True, exist_ok=True)
        for i in idx:
            row = sub.iloc[i]
            stem = Path(str(row["filename"])).stem
            dst = outdir / f"{stem}.png"
            if not dst.exists():
                img = decode(row["image"]).convert("RGB")
                assert img.size == (64, 64), img.size
                img.save(dst, optimize=True)
            records.append({"path": f"data/eurosat_subset/{cls}/{dst.name}",
                            "label": cls, "src_split": row["src_split"],
                            "src_filename": row["filename"], "pool_n": len(sub)})
    return pd.DataFrame(records)


def split_once(df: pd.DataFrame) -> pd.DataFrame:
    idx = np.arange(len(df))
    y = df["label"].to_numpy()
    tr, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=SEED)
    out = df.copy()
    out["split"] = ""
    out.loc[tr, "split"] = "train"
    out.loc[va, "split"] = "val"
    out.loc[te, "split"] = "test"
    return out.sort_values(["split", "label", "path"]).reset_index(drop=True)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    splits_path = BUILD / "splits.csv"
    if splits_path.exists():
        print("splits.csv 已存在，不重算（口径只划分一次）")
        return
    pool = read_pool()
    print(f"原始池：{len(pool)} 张，{pool['cls'].nunique()} 类（EuroSAT RGB 全集 10 类）")
    df = sample_and_save(pool)
    df = split_once(df)
    df.to_csv(splits_path, index=False, encoding="utf-8-sig")
    counts = df.pivot_table(index="label", columns="split", values="path", aggfunc="count")
    counts = counts.reindex(CLASSES)[["train", "val", "test"]]
    print(counts)
    print("合计", len(df), dict(df["split"].value_counts()))
    manifest = {
        "dataset": "EuroSAT RGB", "citation": "Helber et al. 2019, IEEE JSTARS",
        "license": "CC BY 4.0", "crs": "WGS84 (EPSG:4326)", "gsd_m": 10,
        "patch_px": 64, "classes_used": CLASSES, "per_class": PER_CLASS,
        "seed": SEED, "pool_total": int(len(pool)), "subset_total": int(len(df)),
        "split": {"train": int((df.split == "train").sum()),
                  "val": int((df.split == "val").sum()),
                  "test": int((df.split == "test").sum())},
        "per_class_pool": {c: int((pool["cls"] == c).sum()) for c in CLASSES},
        "mirror": "huggingface.co/datasets/blanchon/EuroSAT_RGB (train/validation/test parquet)",
        "note": "三份 parquet 合并成池后重新按 seed 42 抽样，未沿用镜像自带划分",
    }
    (BUILD / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                              encoding="utf-8")


if __name__ == "__main__":
    main()
