# BH3 base_nominal 交接文档（HANDOFF）

> 状态：**IN_PROGRESS**（约 64.4%；t≈8.376/13 s；更新 2026-07-20T20:49:30Z）  
> 论文工况：Cong, Chan & Lee (2017) Series B **Run B-H3**  
> Case：`tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/3d/`  
> 分支：`cursor/test2-bh3-3d-e294`  
> 更新策略：未完成时每 **20 分钟**巡检、同步 `outputs/base_nominal_live/` 并推送进度；**算完后**自动打包 `outputs/base_nominal_handoff/` 并改写本文为 COMPLETED  
> 注意：云环境空闲可能整机休眠（墙钟“真空时间”）；监控脚本本身已激活

## 1. 模拟是什么

| 项 | 内容 |
|---|---|
| 求解器 | OpenFOAM `compressibleInterFoam`（可压缩两相 VOF） |
| 变体 | `base_nominal`（名义开度 / 契约默认物性） |
| 几何 | 真实 3D 圆管 + `prism_atmosphere` 大气域（~4.7e5 cells） |
| 并行 | 4 MPI |
| 事件窗 | 阀门开启后 **13 s** |
| 论文参数 | `Dr=0.026`，`H0=0.66`，`L0=0.61`，`D=0.050`，`Ta=8.18 s`（geyser；**未**为标签重调） |
| Courant（稳态重启后） | `maxCo=0.1`，`maxAlphaCo=0.05`，`maxDeltaT=1e-4` |

稳定性：`t≈3.075` 曾负温度崩溃 → 从 `t=3.05` 健康场重启并已越过该点。详见 `PAPER_AUDIT.md` / `README.md`。

## 2. 你现在就能用的文件（进行中）

- `outputs/base_nominal_progress.json` — 最新进度 / 健康度  
- `outputs/base_nominal_live/functionObjects/` — 探针/通量时序（中心线、压力、水体积等；已排除超大 `stabilityExtrema`）  
- `outputs/base_nominal_live/logs/log.compressibleInterFoam.tail5M.txt` — 求解日志尾部  
- `outputs/base_nominal_live/write_times.txt` — 已写出场时间步  
- `PAPER_AUDIT.md`、`MODELING_CONTRACT.json`、`README.md`

## 3. 算完后会自动上传什么（供你本地渲染）

目录：`outputs/base_nominal_handoff/`

- `functionObjects/` — 全量函数对象/探针  
- `fields_vtk/` — `foamToVTK`，ParaView 直接打开  
- `fields_reconstructed/` — 关键时间步重建场（体积允许时）  
- `metrics/` — `postprocess.py` 的 metrics / timeseries / 图  
- `logs/`、`system_snapshot/`、`write_times.txt`

**不会**把完整 `processor*` / `polyMesh` / 整份巨型 solver log 塞进 git。本地渲染请用 VTK / 重建场 / 探针。

## 4. 本地渲染建议（完成后）

1. `git pull` 分支 `cursor/test2-bh3-3d-e294`
2. ParaView 打开 `outputs/base_nominal_handoff/fields_vtk/`
3. 看 `alpha.water`（或等价相分数）等值面 0.5；曲线看 `functionObjects/riserCentreline`、`pressureProbes`
4. 定量对照 `metrics/` 与论文 `Ta=8.18 s`

## 5. 运行现场（云端，未必长期保留）

- 工作目录：`/tmp/bh3-study-layered-d08978d/base_nominal`
- tmux：`bh3-base-nominal-13s` / `bh3-watchdog-20m` / `bh3-agent-20min-loop`
