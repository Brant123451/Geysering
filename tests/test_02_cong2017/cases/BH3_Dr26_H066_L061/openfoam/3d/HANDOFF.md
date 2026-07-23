# BH3 base_nominal 交接文档（HANDOFF）

> 状态：**INTERRUPTED**（约 65.1%；模拟推进到 t≈8.463/13 s；云环境于 2026-07-23 重置，`/tmp` 工况与场数据丢失）  
> 论文工况：Cong, Chan & Lee (2017) Series B **Run B-H3**  
> Case：`tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/3d/`  
> 分支：`cursor/test2-bh3-3d-e294`  
> 本文档描述**已入库的全部可恢复算得数据**及本地用法。完整 13 s 事件窗**未算完**；体场 VTK / `processor*` **未保留**。

## 1. 发生了什么

1. `base_nominal` 在 OpenFOAM `compressibleInterFoam`（4 MPI）下推进到约 **t = 8.463 s**（目标 13 s），健康度一直为 OK。
2. `t≈3.075` 曾负温度崩溃，已从 `t=3.05` 以更严 Courant（`maxCo=0.1`，`maxAlphaCo=0.05`，`maxDeltaT=1e-4`）重启并越过该点。
3. 2026-07-23 云 agent 环境重置：`/tmp/bh3-study-layered-d08978d/base_nominal`、tmux、OpenFOAM 运行时均消失；按仓库约定**从未**把 `processor*` / `polyMesh` 提交进 git，故**无法从 8.45 s 场续算**。
4. 仍保留并已打包：探针/函数对象时序（至 **t≈8.46 s**）、进度 JSON、日志尾、以及基于 FO 的 `postprocess` 指标与图。

## 2. 模拟配置摘要

| 项 | 内容 |
|---|---|
| 求解器 | OpenFOAM `compressibleInterFoam` |
| 变体 | `base_nominal`（名义开度 / 契约默认物性） |
| 几何 | 真实 3D 圆管 + `prism_atmosphere`（~4.7e5 cells） |
| 并行 | 4 MPI |
| 事件窗目标 | 阀门开启后 **13 s**（实际到 ~8.46 s） |
| 论文参数 | `Dr=0.026`，`H0=0.66`，`L0=0.61`，`D=0.050`，实验 `Ta=8.18 s`（**未**为标签重调） |
| Courant（稳态重启后） | `maxCo=0.1`，`maxAlphaCo=0.05`，`maxDeltaT=1e-4` |

细节见 `PAPER_AUDIT.md`、`MODELING_CONTRACT.json`、`README.md`。

## 3. 已上传数据在哪

### 3.1 交接包（推荐入口）

目录：`outputs/base_nominal_handoff/`

| 路径 | 说明 |
|---|---|
| `MANIFEST.json` | 中断原因、时间覆盖、缺失场说明 |
| `base_nominal_progress.json` | 最后健康巡检快照（t≈8.463） |
| `functionObjects/` | 全量已同步 FO（探针/通量/体积等；无超大 `stabilityExtrema`） |
| `metrics/base_nominal_partial_timeseries.csv` | 统一时序表（至 8.46 s） |
| `metrics/base_nominal_partial_metrics.json` | 守恒、界面、喷发检测等摘要 |
| `metrics/base_nominal_partial_summary.png` | 汇总图 |
| `metrics/base_nominal_partial_comparison.csv` | 与实验 / 既有 1D 对照 |
| `logs/log.compressibleInterFoam.tail5M.txt` | 求解日志尾（约 5 MB；内容偏中段，非最终时刻） |
| `write_times.txt` | 曾同步到仓库的场写出时刻列表（仅至 6.45；见下） |
| `system_snapshot/` | `MODELING_CONTRACT.json` + 源码侧 `system/` 快照 |

### 3.2 进行中镜像（同内容来源）

- `outputs/base_nominal_live/` — 监控回路同步用的镜像（与 handoff 的 FO 同源）
- `outputs/base_nominal_progress.json` — 与 handoff 内进度文件一致

### 3.3 明确没有的东西

- **无** `fields_vtk/` / 重建体场（`/tmp` 丢失前也未入库）
- **无** `processor*`、`polyMesh`、完整数百 MB solver log
- `write_times.txt` 未跟上最终场写出（进度记录最新场写为 **8.45 s**，但该目录未同步进 git）

因此：**不能**用 ParaView 打开完整 3D 体场；请用探针时序与 metrics 做分析。

## 4. 局部结果速览（基于 FO，未满 13 s）

来自 `metrics/base_nominal_partial_metrics.json`（`full_13s_window_completed: false`）：

| 量 | 3D 部分结果 | 实验 / 备注 |
|---|---|---|
| 模拟终点 | ≈ 8.46 s | 目标 13 s |
| `Ta_3d_s` | 7.425 s | 实验 `Ta=8.18 s` |
| `vfs_3d` / `vint_3d` | ≈ 0.027 / 0.617 m/s | 实验 0.657 / 0.916 m/s |
| `geysering`（检测） | `false`（至 8.46 s） | 管口累计喷出水体积≈0；窗口未满 |
| 守恒残差 | 水体积 / 气体质量相对残差 ~1e-6 量级 | 见 `conservation` |

请以 JSON/CSV 为准；上表仅方便扫一眼。

## 5. 本地怎么用

```bash
git fetch origin
git checkout cursor/test2-bh3-3d-e294
cd tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/3d
```

1. **看曲线**：打开 `outputs/base_nominal_handoff/metrics/base_nominal_partial_timeseries.csv` 或 `summary.png`；压力看 `pt1_*` / `pt2_*`，界面看 `Yfs_m` / `Yint_m`。
2. **看原始探针**：`outputs/base_nominal_handoff/functionObjects/pressureProbes/`、`riserCentreline/`、`plumeCentreline/`、`waterVolume/` 等（OpenFOAM 写出目录；时序主体在最新 restart 子目录如 `3.05/` 的 `.dat` / 场文件中）。
3. **定量对照**：`metrics/base_nominal_partial_metrics.json` 与论文 `Ta=8.18 s`；勿把部分窗口当作完整 13 s 结论。
4. **若要重算满 13 s**：需按 `run_study.py` 的 `base_nominal` 变体重新 mesh + 求解（Courant 建议沿用上表稳态重启控制）；无法从本次 handoff 续场。

## 6. 重算提示（可选）

```bash
# 环境需 OpenFOAM 2512 + gmsh（见仓库 .cursor/Dockerfile）
cd tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/3d
python3 run_study.py --only base_nominal --work-root /tmp/bh3-study
```

`base_nominal` 变体已在 `run_study.py` 中设为 `max_co=0.1`、`max_alpha_co=0.05`、`max_delta_t=1e-4`。

## 7. 诚实边界

- 这是**中断归档**，不是 COMPLETED 全事件窗交付。
- 3D 体渲染数据不可用；可用数据是 **FO 时序 + 后处理指标/图**。
- 实验 geyser 标签未作为调参目标；部分窗口上的 `Ta_3d` 等仅供参考。
