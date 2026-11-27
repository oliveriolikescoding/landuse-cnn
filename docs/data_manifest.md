# 数据清单 · EuroSAT RGB 八类子集

## 一、数据集身份

| 项 | 值 |
|---|---|
| 名称 | EuroSAT RGB（Sentinel-2 影像的 RGB 三波段版本） |
| 原始论文 | Helber, P., Bischke, B., Dengel, A., Borth, D. (2019). EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification. *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing*, 12(7), 2217–2226. |
| 许可 | **CC BY 4.0**（转载图例与再分发须保留原作者与许可） |
| 覆盖范围 | 欧洲 34 国的 Sentinel-2 影像；每块 64×64 像元、地面分辨率 10 m（覆盖约 640 m × 640 m） |
| 原始体量 | 27 000 张、10 类（每类 2 000–3 000 张），全部带地理参考 |
| 本地来源 | Hugging Face 镜像 `blanchon/EuroSAT_RGB` 的 `train / validation / test` 三份 parquet（合计 174 MB） |

> 说明：原始发布渠道 `madm.dfki.de` 在本机网络下连接超时，改用上述镜像；镜像的类别名与顺序与原始一致（10 类），
> 镜像自带划分未沿用：三份 parquet 先合并成一个池（27 000 张），再按本项目的口径重新抽样与划分。

## 二、类别选择（8 类）

选取 8 类，其中三组在 10 m 尺度上视觉相近（一年生作物与多年生作物、住宅区与工业区、河流与湖海），
以便混淆矩阵反映类间可分性。草地植被与公路两类未使用。

| 用于 | 类别（英文） | 中文 | 池内张数 |
|---|---|---|---|
| ✓ | AnnualCrop | 一年生作物 | 3 000 |
| ✓ | Forest | 森林 | 3 000 |
| ✓ | Industrial | 工业区 | 2 500 |
| ✓ | Pasture | 牧场 | 2 000 |
| ✓ | PermanentCrop | 多年生作物 | 2 500 |
| ✓ | Residential | 住宅区 | 3 000 |
| ✓ | River | 河流 | 2 500 |
| ✓ | SeaLake | 湖海 | 3 000 |
| ✗ | HerbaceousVegetation | 草地植被 | 未用 |
| ✗ | Highway | 公路 | 未用 |

## 三、抽样与落盘

- 每类 **500 张**，`numpy.random.RandomState(42).choice` 无放回抽样，抽样后按文件名排序写入；
- 落盘为 PNG（无损，保留原 64×64 尺寸、不做任何重采样），路径 `data/eurosat_subset/<类名>/`；
- 合计 4 000 张，**24.1 MB**，随仓库分发（不做 git-lfs），保证任何人 clone 后可直接复现；
- 抽样清单：`build/splits.csv`（`path, label, split, src_split, src_filename, pool_n`）。

## 四、划分（只做一次）

分层抽样（`train_test_split(stratify=label, random_state=42)`），三份写进同一份 `build/splits.csv`，
**基线与 CNN 共用**；训练脚本里没有任何第二次 split，重跑不会变。

| split | 张数 | 每类 |
|---|---|---|
| train | 2 800 | 350 |
| val | 600 | 75 |
| test | 600 | 75 |

## 五、已知损耗与失真（写进 model_card 的局限里）

1. **类别完全平衡**：真实地表覆盖并不平衡（作物与森林远多于湖海），平衡化让 macro-F1 与准确率可比，
   代价是低估大类、抬高小类在总体指标里的权重。
2. **下采样到 64×64**：原始 Sentinel-2 为 10 m，EuroSAT 已把像元块裁成 64×64，
   本项目未恢复更高分辨率；细小地物（小路、单栋厂房）在此尺度上不可辨。
3. **同源同季**：同一景/相邻景的 patch 会同时出现在训练与测试集里，随机划分存在泄漏，
   测试分数不能读作跨区域泛化能力。
4. **仅 RGB 三波段**：EuroSAT 另有 13 波段版本，本项目用 RGB 版；
   近红外与短波红外对作物与水体区分很有帮助，这里没有使用。
5. **地理范围**：全部样本来自欧洲，**不能**直接迁移成“成都地块分类能力”的结论。


