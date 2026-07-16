# B-H6 3D OpenFOAM 交接说明（Cloud Agent）

**分支：** `cursor/test2-series-b-3d-c89c`  
**案例根：** `tests/test_02_cong2017/cases/BH6_Dr41_H066_L061/`  
**3D 目录：** `openfoam/3d/`  
**工况：** Cong, Chan & Lee (2017) Series B **B-H6**  
（`Dr=0.041 m`, `H0=0.66 m`, `L0=0.61 m`，实验 **NO GEYSER**）

本文说明本分支已完成内容、仓库内可下载的模拟数据位置，以及如何在本地用这些数据做正视渲染。

**本地渲染入口（一句话）：** 拉取本分支后，用 ParaView 打开  
`results/frontview-3d/cuttingPlane/<time>/yMid.vtp`（130 帧，`alpha.water`，y=0 正视），按水/气两色出图即可。细节见 §2。

### 已入库数据量（约）

| 内容 | 路径 | 规模 |
|------|------|------|
| 正视 VTP 时间序列 | `results/frontview-3d/cuttingPlane/` | **130** 帧，约 **123 MB** |
| 参考 GIF/MP4 + PNG | `results/animations/` | 约 **28 MB**（含 `real_y0_frames/` 130 PNG） |
| frontview 探针 | `results/frontview-3d/probes/` | 约 **13 MB** |
| refined / wall-bl 探针 | `results/{refined,wall-bl-v6}/probes/` | 各约 **12 MB** |
| 各 profile 指标 CSV/JSON | `results/{base,valve-*,refined,wall-bl-v6}/` | 约数 MB |
| **合计（results/）** | | 约 **190 MB** |

---

## 1. 任务完成状态

| 项目 | 状态 |
|------|------|
| 论文几何审计（Fig.1：管长 6.59 m，三通 x=3.47 m） | 完成，见 `PAPER_AUDIT.md` |
| 筛选战役 base / valve-fast / valve-slow / refined | 完成，均 NO GEYSER |
| 近壁 Distance/Threshold（`wall_bl_v6`）门控 + 13 s | 完成，NO GEYSER |
| **真实 y=0 中面 `alpha.water` 正视动图数据（13 s）** | 完成（base 网格重跑 + cuttingPlane） |
| 全场体积时刻（processor 时间目录） | **未保留**（`purgeWrite 3`），请用下方切割面 VTP |

定量结果摘要（与实验对照）见各 `results/*/metrics.json` 与 `results/campaign_summary.json`。

---

## 2. 本地渲染优先使用的数据（最重要）

### 2.1 正视切割面 VTP（推荐主数据）

路径：

```text
openfoam/3d/results/frontview-3d/cuttingPlane/<time>/yMid.vtp
```

- **内容：** 真三维 `compressibleInterFoam` 结果在 **y=0** 中面的 `alpha.water` 切片（VTK PolyData / VTP）
- **帧数：** 130（`t = 0.1 … 13.0 s`，名义间隔 0.1 s；后期加速段名义 0.2 s 仍按写出时刻目录）
- **清单：** `results/frontview-3d/vtp_manifest.json`
- **切面定义：** 过点 `(3.47, 0, 0)`，法向 `(0, 1, 0)`（与三通轴线正交的正视面）
- **几何：** 全长主管 `x=0…6.59 m` + 立管 + rim 上外域

ParaView 用法示例：

1. 打开任意 `yMid.vtp`，或用 **File → Open** 选中 `cuttingPlane` 下各时刻目录中的 `yMid.vtp` 组成时间序列（或用 `Glob` / `File Series`）。
2. 着色字段：`alpha.water`（或 Point Data 中带 alpha 的数组名）。
3. 视图：正交投影，看 **X–Z**（正视）；蓝/白自行设查色表即可。
4. 另存 Animation / GIF / AVI。

仓库内已有一份预渲染参考 GIF（可用可忽略）：

```text
openfoam/3d/results/animations/BH6_base_real_y0_frontview.gif
openfoam/3d/results/animations/BH6_base_real_y0_frontview.mp4
openfoam/3d/results/animations/real_y0_frames/   # 逐帧 PNG
```

脚本（从 VTP 画蓝/灰正视帧）：

```text
openfoam/3d/scripts/render_real_frontview_vtp.py
```

示例：

```bash
python3 openfoam/3d/scripts/render_real_frontview_vtp.py \
  --cut-root openfoam/3d/results/frontview-3d/cuttingPlane \
  --out-dir /tmp/bh6_frames --prefix BH6_local
```

（依赖：`vtk`、`matplotlib`、`numpy`。）

### 2.2 探针时序（定量 / 自由面轨迹）

| 数据 | 路径 |
|------|------|
| frontview-3d 立管/羽流/PT | `results/frontview-3d/probes/{riserCentreline,plumeCentreline,PT1,PT2}/` |
| refined | `results/refined/probes/` + `results/refined/{metrics.json,series.csv,mass_balance.csv}` |
| wall-bl-v6 | `results/wall-bl-v6/probes/` + 同结构 metrics/series |
| base / valve-* | `results/<profile>/{metrics.json,series.csv,mass_balance.csv}`（无完整探针目录时以 CSV 为准） |

OpenFOAM 探针文件在各 `*/0/alpha.water`（或 `p`）内为**全时程追加**表，可用 `numpy.loadtxt(..., comments='#')` 读取。

---

## 3. 目录地图

```text
openfoam/3d/
├── HANDOFF.md                 ← 本文件
├── PAPER_AUDIT.md             ← 论文几何/边界审计
├── model_inputs.json          ← 建模输入总表
├── README.md
├── Allrun / Allclean
├── make_mesh.py               ← 基线 HXT 棱柱立管网格
├── make_mesh_wall_bl.py       ← 近壁 Distance/Threshold
├── make_mesh_ogrid_riser.py   ← hex O-grid 尝试（未过门控）
├── postprocess.py
├── system/
│   ├── controlDict            ← 含 #include "frontViewCut"
│   ├── frontViewCut           ← y=0 surfaces 采样字典
│   └── …
├── wall_bl_gate_status.json
├── ogrid_riser_gate_status.json
└── results/
    ├── campaign_summary.json
    ├── base|valve-fast|valve-slow|refined|wall-bl-v6/
    │   ├── metrics.json
    │   ├── series.csv
    │   ├── mass_balance.csv
    │   └── probes/            ← refined & wall-bl-v6 有
    ├── frontview-3d/          ← ★ 正视渲染主数据
    │   ├── cuttingPlane/<t>/yMid.vtp
    │   ├── probes/
    │   ├── vtp_manifest.json
    │   ├── system/            ← 运行时 controlDict / frontViewCut 快照
    │   ├── config/
    │   └── run_status.json
    └── animations/            ← 参考 GIF/MP4/PNG
```

说明：仓库根 `.gitignore` 忽略 `**/openfoam/**/postProcessing/`，因此归档时使用了 `cuttingPlane/`、`probes/` 命名，避免被忽略。

---

## 4. 几何与坐标（渲染时务必一致）

| 量 | 值 |
|----|-----|
| 主管内径 D | 0.050 m |
| 立管内径 Dr | 0.041 m |
| 管长 | 6.59 m（3.47 + 3.12） |
| 三通 | x = 3.47 m |
| 阀 / 气囊 | x = 5.98 … 6.59 m（L0=0.61 m） |
| 管轴 z=0；管底/管顶 | z=±0.025 m |
| H0 | 0.66 m（管底起算）→ 自由面 z=0.635 m |
| 立管 rim | z=1.825 m（管顶以上 1.8 m） |
| 外域顶 | z=3.025 m（数值域，非实验管长） |
| Yfs / Yint 图像基准 | **立管入口（管顶）**；与 H0 管底基准差 D=0.05 m |

勿与仓库旧 1D 的「6.0 m / 三通 2.88 m」混用；3D 以 Fig.1 / `PAPER_AUDIT.md` 为准。

---

## 5. 各工况结果速查

| profile | cells（约） | Ta_3d (s) | Yfs_max (m) | geyser |
|---------|-------------|-----------|-------------|--------|
| 实验 B-H6 | — | 8.10 | ~1.21（入口上） | NO |
| base | 1.5e5 | 9.60 | 1.040 | NO |
| valve-fast | 1.5e5 | 9.55 | 1.020 | NO |
| valve-slow | 1.5e5 | 9.68 | 1.019 | NO |
| refined | 2.75e5 | 9.55 | 1.125 | NO |
| wall-bl-v6 | 1.73e5 | 9.63 | 1.061 | NO |
| frontview-3d（本次正视重跑） | base 级 | （见 probes/series） | — | 用于可视化 |

近壁门控通过版：`wall_bl_v6`（立管壁 Distance/Threshold，排除 rim 唇；`max_nonOrth≈68.2°`）。

---

## 6. 未上传 / 无法恢复的内容

- **全场体积时刻目录**（`processor*/[0-9]*` 中的 `U,p,alpha.water,…`）：运行时 `purgeWrite 3`，仅保留末尾若干步；完整瞬态体积场**没有**。
- 大网格 `.msh`、并行分解目录、求解日志：按 `.gitignore` 未入库。
- hex O-grid 近壁路线：未过求解器门控；过程记在 `ogrid_riser_gate_status.json`。

若要重新生成体积场：在本 case 下 `Allrun`，并设 `purgeWrite 0`（或较大），保留 `system/frontViewCut`；磁盘与墙钟成本很高（完整 13 s 数小时级）。

---

## 7. 本地复现求解（可选）

```bash
cd tests/test_02_cong2017/cases/BH6_Dr41_H066_L061/openfoam/3d
# 需要 OpenFOAM v2512 + gmsh
export BH6_PROFILE=base          # 或 refined
export OPENFOAM_NP=4
export BH6_RESULTS_DIR=$PWD/results/$BH6_PROFILE
./Allclean   # 若已有生成物
./Allrun
```

近壁网格：

```bash
export BH6_MESH_SCRIPT=make_mesh_wall_bl.py
export BH6_MESH_EXTRA_ARGS='--first-wall-m 0.004 --bl-growth 1.15 --bl-layers 3'
```

正视采样已写入 `system/controlDict` 的 `#include "frontViewCut"`。

---

## 8. 交接检查清单（给你）

1. [ ] `git checkout cursor/test2-series-b-3d-c89c` 并 `git pull`
2. [ ] 确认 `results/frontview-3d/cuttingPlane/` 下约 130 个时刻的 `yMid.vtp`
3. [ ] 用 ParaView / VisIt / VTK 打开 VTP，按 `alpha.water` 做正视动画
4. [ ] 需要定量曲线时读 `results/*/series.csv` 或 `probes/`
5. [ ] 几何疑问先查 `PAPER_AUDIT.md` / `model_inputs.json`

---

## 9. 联系上下文

- 求解器：OpenFOAM **v2512** `compressibleInterFoam`
- 网格：主管/三通四面体 + 立管垂向扫掠三棱柱（HXT）；近壁为 Distance+Threshold
- 阀门：`cyclicACMI` 变面积，基准开阀 0.2 s
- 本 Cloud Agent 分支专用于 B-H6 真三维与正视数据归档；PR 工具若报仓库 URL 不匹配，以 GitHub 上该分支推送内容为准

**本地渲染请以 `results/frontview-3d/cuttingPlane/**/yMid.vtp` 为准，不要使用早期示意塞状界面 GIF。**
