# Case A 3D 本地交接说明（Cloud Agent → 本机）

**日期**：2026-07-21  
**分支**：`cursor/test1-case-a-openfoam-78de`  
**案例路径**：`tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356/openfoam/3d/`  
**论文**：Vasconcelos & Wright (2011) Case A（大塔、不喷发）

---

## 1. 结论先说

| 项目 | 状态 |
| --- | --- |
| 均匀细网格 8 mm / **342,135** 单元 / 完整 9 s | **已完成**，紧凑结果已在 `outputs/` 并入库 |
| 塔壁加密 wall-refined / 目标 **~463,060** 单元 / 完整 9 s | **未完成**。云端曾跑到约 **72%（t≈6.5/9）**，环境重置后 **processor\*/polyMesh/log 全部丢失**，无法续算 |
| 本仓库当前可交付 | 配置 + 脚本 + 已完成的 342k 紧凑对比结果 + 本交接文档 |

请在你本机按下文重跑 **wall-refined 全时长 9 s**，算完后把新的 `outputs/` 紧凑文件提交回本分支。

---

## 2. 硬约束（不要改）

- **几何**、初值 **Ha0 = 0.305 m**、**Yfs0 = 0.356 m** 不得改
- `Dt = 57.1 mm`（`Dt/D = 0.607`）
- 已知简化（与论文叙述一致）：瞬时开阀、无蝶阀局部损失、光滑壁面
- 数字化试验：`../../data/digitized/fig5_caseA_Hstar_band.csv`、`fig7_caseA_levels.csv`

---

## 3. 已入库、可直接查看的结果（342k / 8 mm）

目录：`outputs/`（勿提交 `processor*` / `polyMesh` / `log.*` / `*.msh`）

| 文件 | 用途 |
| --- | --- |
| `openfoam_3d_metrics.json` | 总指标 |
| `openfoam_3d_series.csv` | 压力/液位时间序列 |
| `openfoam_3d_levels.csv` / `*_comparison.png|.pdf` | Fig.7 液位对比 |
| `openfoam_3d_pressure_comparison.png|.pdf` | Fig.5 压力对比 |
| `openfoam_3d_tower_sections.csv` | 塔截面 `areaAverage(alpha.water)` |
| `openfoam_3d_plume.csv` / `water_mass.csv` | 羽流与质量守恒 |
| `openfoam_3d_interface_sensitivity.csv` | 鼻端阈值 0.80–0.95 |

**342k 关键结果摘要**（`simulation_end_s = 9.0`）：

| 量 | 试验目标 | 342k 结果 |
| --- | ---: | ---: |
| 压力平台 `H*` | 0.54 | **0.552** |
| 最高自由面 `Yfs*` | 0.63 | **0.623** |
| 界面爬升 `V*`（Fig.7 重复约 0.426–0.431） | ~0.43 | **0.465（偏快约 +8%）** |
| liftoff / catch `T*` | 合理区间 | 7.61 / 8.71 |
| 喷发 | 否 | 否 |

界面后处理：水平截面 `areaAverage(alpha.water)`，鼻端阈值默认 **0.90**（非单中心线）。

---

## 4. 未完成的 wall-refined 工况（请你本机重跑）

### 4.1 动机

342k 均匀 8 mm 上 `H*`/`Yfs*`/不喷发已较好，但 **`V*` 偏快**。本轮目标是提高塔壁膜流分辨率 + 改进界面数值，压低 `V*` 使其落入 Fig.7 的约 **0.426–0.431**。

### 4.2 目标网格（`Allrun` 当前默认）

```bash
CASEA_CORE_SIZE=0.008   # 主管/核心 8 mm
CASEA_TOWER_SIZE=0.005  # 塔内 5 mm
CASEA_WALL_SIZE=0.0025  # 塔壁 2.5 mm
CASEA_PLUME_SIZE=0.020  # 大气羽流区 20 mm
```

云端 `log.gmsh` 曾记录：`cells_3d=463060`（约 46.3 万四面体）。  
网格由 `make_mesh.py` 的 Distance+Threshold 塔壁带生成；`StopAtDistMax=1`，避免全域被迫加密到千万级。

### 4.3 数值设置（已在仓库中）

- 求解器：`compressibleInterFoam`（OpenFOAM **v2512**）
- `system/fvSchemes`：`div(phirb,alpha) Gauss interfaceCompression`；温度 `limitedLinear`；动量保持 **`upwind`**（`limitedLinearV` 曾在 t≈0.082 s 触发 GaussSeidel FPE，已回退）
- `system/fvSolution`：`MULESCorr yes`，`nLimiterIter 8`
- `system/controlDict`：`endTime 9`；`maxCo 0.20`；`maxAlphaCo 0.15`；`maxDeltaT 0.00025`；场写出间隔 0.1 s
- 温度限幅：`constant/fvOptions`（250–350 K），指标中可审计激活次数

### 4.4 云端中断时的进度（仅供参考，不可续算）

- 网格：wall-refined ~463k
- 并行：4 核 `mpirun`
- 最后观察到约 **t≈6.5 / 9.0（~72%）**，`deltaT` 多在 `1.5e-5–9e-5`，`Co_max` 常顶在 `maxAlphaCo≈0.15`
- 健康时推进约 `1.5e-5–2.5e-5` sim/s → 剩余纯算大约 **1.3–2 天**（视机器与暂停而定）
- **没有**可恢复的 `processor*` / `log.compressibleInterFoam`；本机必须 **`./Allrun` 全新开跑**，不要指望 `Allrun.resume` 接云端残骸

---

## 5. 本机怎么跑

### 5.1 依赖

- OpenFOAM **v2512**（与 `.cursor/Dockerfile` / `Allrun` 中 `openfoam2512` 一致）
- Gmsh（云端曾用 4.12.1）
- Python3 + NumPy + Matplotlib
- MPI（建议 4 核；`Allrun` 默认 `min(nproc,6)`）

### 5.2 干净开跑（推荐）

```bash
cd tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356/openfoam/3d
# 确保没有残留运行时目录
rm -rf 0 processor* constant/polyMesh postProcessing caseA3d.msh log.* \
       system/decomposeParDict.runtime system/towerCrossSections.runtime

chmod +x Allrun Allrun.resume
# 可选显式指定（与默认相同）
export CASEA_CORE_SIZE=0.008 CASEA_TOWER_SIZE=0.005 CASEA_WALL_SIZE=0.0025 CASEA_PLUME_SIZE=0.020
export OPENFOAM_NP=4

./Allrun
```

`Allrun` 流程：`make_mesh.py` → `gmshToFoam` → `createPatch` → `checkMesh` → `setFields` → `decomposePar` → `compressibleInterFoam -parallel` → 塔截面后处理 → `postprocess_compare.py` → 刷新 `outputs/`。

成功结束时日志应出现：`CASE_A_3D_DONE`。

### 5.3 中断后续算（仅本机自己的分解结果）

```bash
./Allrun.resume
```

要求本地已有 `processor*` 与 `log.compressibleInterFoam`。

### 5.4 冒烟（可选）

```bash
CASEA_DRY_RUN=1 ./Allrun
```

### 5.5 算完后请提交什么

只提交紧凑产物（与现有一致）：

```text
outputs/openfoam_3d_*.{json,csv,png,pdf}
```

**不要**提交：`processor*`、`constant/polyMesh/`、`caseA3d.msh`、`log.*`、`postProcessing/`、时间目录 `[0-9]*`（`0.orig` 除外）。  
这些已在仓库根 `.gitignore` 中忽略。

建议在 PR / commit message 中写明：

- `cells_3d`、`core/tower/wall/plume` 尺寸
- 新的 `H*` / `Yfs*` / `V*` / liftoff / catch / 是否喷发
- `V*` 是否落入 Fig.7 的 0.426–0.431

### 5.6 本地渲染

场为 binary、`writeInterval 0.1 s`。本机可用 ParaView 打开重构后的案例，或对 `processor*` 做 `reconstructPar` 后再渲染。云端不负责出渲染图；你本地生成即可。

---

## 6. 建议验收标准（wall-refined 收官）

1. 完整跑到 **9.0 s**，不喷发  
2. `H*` 接近 0.54，`Yfs*` 接近 0.63  
3. **重点**：`V*` 尽量落入 Fig.7 重复拟合 **0.426–0.431**（342k 为 0.465）  
4. 质量漂移保持很小（342k 量级约 `1e-5`）  
5. 更新 `outputs/` 与本文件中的“完成状态”表

若 `V*` 仍明显偏快：可再考虑塔壁更细 / 界面格式，**仍不改几何与 Ha0/Yfs0**。

---

## 7. 相关提交

- `bd4cc465` — 记录 342k 完整验证结果  
- `ee3bceda` — 截面平均界面追踪  
- `3ba642cf` — 塔壁加密网格 + MULES / interfaceCompression  
- `104848de` / `6184b8c1` — limitedLinearV FPE 后回退动量 upwind，并恢复 Co 限制  

PR：`https://github.com/brant123451/geysering/pull/11`（若远端大小写为 `Brant123451/Geysering`，以实际仓库为准）

---

## 8. 联系本交接的意图

云 Agent 环境已无法继续长算；把**可复现配置**和**已完成的 342k 紧凑结果**交给你本机完成 wall-refined 9 s 收官验证。有问题以本目录 `README.md` + `Allrun` + `make_mesh.py` 为准。
