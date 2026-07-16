# B-H2 三维 OpenFOAM 交接说明

Case：`tests/test_02_cong2017/cases/BH2_Dr21_H066_L061/openfoam/3d`  
论文：Cong, Chan & Lee (2017) Series B / Run **B-H2**  
分支：`cursor/test2-bh2-3d-41f6`  
求解器：OpenFOAM v2512 `compressibleInterIsoFoam`（水常密度 + 空气理想气体 + isoAdvection）

本文档供本地重建 / ParaView 渲染使用。云端环境仅 **4 核**，refined 网格约 **58 万单元**，吞吐约 **0.11–0.13 s 物理时间/小时**，完整敏感性队列墙钟很长。

---

## 1. 几何与物理（勿按 geyser 结果反调）

详见 `PAPER_AUDIT.md`、`MODEL_INPUTS.md`。要点：

| 项目 | 值 |
|---|---|
| 主管 | 圆管 `D=0.050 m`，`x=0…6.59 m` |
| 立管 | `Dr=0.021 m`，中心 `x=3.47 m`，壁高 1.8 m， rim `z=1.825 m` |
| 球阀平面 | `x=5.98 m`（ACMI 有效面积 0→1，约 0.2 s；instant 变体瞬时全开） |
| 气袋 | `x=5.98…6.59 m`，`L0=0.61 m` |
| 初水位 | 立管自由面 `z=0.635 m`（`H0=0.66 m`） |
| 外域 | 立管口外 `0.30×0.30×1.20 m` 空气域（承接喷出，非实验立管本体） |
| 事件窗 | `endTime=13 s` |

**不做**压力/动量源去“逼出”喷发。

---

## 2. 算例变体与目录

所有可抛运行目录在 `runs/`（默认被 `.gitignore` 忽略；交付时会按需放开并 LFS 上传）。

| Run ID | 网格 | 阀门 | 状态（交接时以最新 commit / `results/` 为准） |
|---|---|---|---|
| `base_closed` | base ~24 万单元 | 闭阀 | 短时静压 hold，已有紧凑结果 |
| `base_baseline` | base ~24 万 | 0.2 s 线性开阀 | **13 s 已完成**，紧凑 CSV/JSON/PNG 已在 `results/` |
| `refined_baseline` | refined ~58 万 | 0.2 s 线性开阀 | 运行中 / 或完成后上传场数据 |
| `base_instant` | base ~24 万 | 瞬时全开 | 排队 / 或完成后上传 |

模板 case：`case/`  
入口脚本：`Allrun`、`postprocess.py`、`make_mesh.py`

### 已提交的紧凑结果（随时可看）

`results/`：

- `openfoam_base_baseline_metrics.json` / `_series.csv`
- `openfoam_base_closed_metrics.json` / `_series.csv`
- `comparison_*.json/png`、`experiment_1d_3d.png`、`mass_conservation.png`、`variant_sensitivity.png`

`base_baseline` 标量摘要（模型未调参）：`Ta≈8.33 s`，`Yfs_max≈1.46 m`（管口 1.8 m），`vfs≈0.830`，`vint≈1.015`，喷出≈0 → **模型判 no-geyser**（实验为 geyser）。

---

## 3. 云端运行中的重要事件

1. `refined_baseline` 在 `t≈4.686 s` 曾因 **负温度**（`T0≈-58 K`）`FOAM FATAL` 中止；湍流 `k/ε` 同期出现大幅 bounding。
2. 自动恢复曾误从 `t=0` 重启；已纠正为 **`startFrom latestTime`**，从最新写出场 **`t=4.6`** 续算。
3. 续算时将 `maxCo` / `maxAlphaCo` 暂降为 **0.25**、`maxDeltaT=1e-4`（数值稳定，非“调参造喷发”）。越过崩溃点后若仍稳定，可酌情回调到 0.35。
4. 看门狗 / 续算逻辑：异常时只允许 `latestTime` 续跑，禁止再从 0 覆盖。

---

## 4. 本地如何生成渲染

### 4.1 环境

```bash
# OpenFOAM v2512（或兼容的 compressibleInterIsoFoam）
source /path/to/OpenFOAM-v2512/etc/bashrc
cd tests/test_02_cong2017/cases/BH2_Dr21_H066_L061/openfoam/3d
```

### 4.2 若仓库已包含 `runs/<run>/processor*` 场数据

```bash
cd runs/base_baseline   # 或 refined_baseline / base_instant
reconstructPar -latestTime          # 或 -time '0:13'
# 推荐导出 VTK 供 ParaView：
foamToVTK -latestTime               # 或指定时间列表
# ParaView 打开 VTK/ 或用 .foam 伪文件：
touch case.foam && paraview case.foam
```

建议可视化字段：

- `alpha.water`：自由面 / Taylor 泡 / 喷发判定
- `U`、`p` 或 `p_rgh`：流场与压力
- `T`：检查续算后温度是否仍物理
- 立管中心线 / 外域：对照 `postProcessing/` 探针与通量

### 4.3 若只有模板 + 紧凑结果、没有场数据

```bash
./Allrun prepare base baseline
./Allrun solve base baseline          # 重跑（耗时长）
python3 postprocess.py --run base_baseline
```

网格由 `make_mesh.py` + gmsh 生成；勿手改尺寸，先读 `PAPER_AUDIT.md`。

### 4.4 后处理标量

```bash
python3 postprocess.py                # 聚合所有已完成 run
python3 postprocess.py --run refined_baseline
```

输出只写 `results/` 紧凑文件。

---

## 5. 场数据上传策略（算完后执行）

目标：本地能渲染，尽量多给数据，同时避开巨型无用文件。

计划纳入 Git LFS（算完后由 agent 执行）：

- `runs/*/processor*/constant/polyMesh/**`（重建几何必需）
- `runs/*/processor*/<time>/{alpha.water,U,p,p_rgh,T,phi}`（主渲染场）
- `runs/*/system/**`、`runs/*/constant/thermophysical*`、`turbulenceProperties`、`g`、`hRef`
- `runs/*/postProcessing/**`（探针与通量时间序列）
- `runs/*/mesh_stats.json`、`initial_audit.json`
- 可选：`runs/*/log.solve`（体积大，优先截取末尾或单独附件）

通常不上传：

- `*.msh` 原始 gmsh（可用 `make_mesh.py` 再生）
- 完整 `log.*` 全量（数百 MB–GB）
- 中间临时目录、`results/raw/`

若 GitHub 配额不够：优先保证 **`base_baseline` 全时程** + **refined / instant 稀疏时刻**（如每 0.5–1.0 s 或事件窗关键时刻）。

---

## 6. 复现命令速查

```bash
BH2_NP=4 ./Allrun prepare base baseline
BH2_NP=4 ./Allrun closed base
BH2_NP=4 ./Allrun smoke base baseline
BH2_NP=4 ./Allrun solve base baseline
BH2_NP=4 ./Allrun solve refined baseline
BH2_NP=4 ./Allrun solve base instant
python3 postprocess.py
```

续算（已有 `processor*`）：

```bash
cd runs/refined_baseline
foamDictionary system/controlDict -entry startFrom -set latestTime
mpirun --oversubscribe -np 4 compressibleInterIsoFoam -parallel 2>&1 | tee -a log.solve
```

---

## 7. 当前进度快照（文档生成时）

- 监控：每 20 分钟健康检查；异常则 `latestTime` 续算。
- `base_baseline`：完成（13 s）。
- `refined_baseline`：进行中（文档生成时约 38%+，以最新 `HANDOFF` / `results` / 进度消息为准）。
- `base_instant`：refined 结束后自动启动。
- 全部 `End` 后：运行交付脚本，把场数据 + 更新后的本文件推送到本分支。

更新本文件时请同步改本节进度与“已上传内容清单”。

---

## 8. 联系 / 分支

- Repo：`brant123451/geysering`
- Branch：`cursor/test2-bh2-3d-41f6`
- Case 根：`tests/test_02_cong2017/cases/BH2_Dr21_H066_L061/openfoam/3d`
