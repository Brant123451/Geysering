# B-H4 3D 模拟交接文档

面向本地 ParaView 渲染与复现核对。云端基线已跑完；本文件说明工况、结果与数据包用法。

## 1. 身份

| 项 | 值 |
|---|---|
| 论文 | Cong, Chan & Lee (2017), Test 2 / Series B |
| 工况 | **B-H4**（实验 **NO GEYSER**，与 B-H3 成对） |
| 参数 | `Dr=0.031 m`，`Dr/D=0.62`，`H0=0.66 m`，`L0=0.61 m` |
| Case 目录 | `tests/test_02_cong2017/cases/BH4_Dr31_H066_L061/` |
| OpenFOAM case | `.../openfoam/3d/` |
| 求解器 | `compressibleInterFoam`（OpenFOAM v2512） |
| 基线标签 | `base_topen0p20`（阀 `t_open=0.20 s`） |
| 分支 | `cursor/test2-bh4-3d-3d79` |
| PR | https://github.com/Brant123451/Geysering/pull/16 |

论文几何/IC/BC 审计：`openfoam/3d/PAPER_AUDIT.md`  
参数快照：`openfoam/3d/case_parameters.json`

## 2. 最终结果（已完成）

紧凑产物（已入库）：

- `outputs/openfoam3d/base_topen0p20_metrics.json`
- `outputs/openfoam3d/base_topen0p20_timeseries.csv`
- `outputs/openfoam3d/base_topen0p20_comparison.png`

| 量 | 3D | 实验 |
|---|---|---|
| 分类 | **NO_GEYSER** | NO_GEYSER |
| 匹配 | **True** | — |
| `Ta` | 8.41 s | 8.14 s |
| `Yfs_max`（冠上） | 1.59 m | rim=1.8 m（未过 rim） |
| 外域出水 | ~0 | — |
| 事件窗 | 完整（≈13.03 s） | 13 s |
| `vfs` / `vint` | 0.91 / 1.11 m/s | 0.207 / 0.418 m/s（偏高，见下） |

说明：论文对 B-H4 的核心是 **No geyser**；本算例未越顶、外域无实质出水，分类一致，`Ta` 接近。攀升速度明显快于实验，**不要**按速度标定去改接触角/阀律/气压。

数值控制：`maxCo=0.40`，`maxAlphaCo=0.20`，`nAlphaSubCycles=4`，`nOuterCorrectors=3`。

## 3. 模拟数据包（尽量全量）

全部在：

```
tests/test_02_cong2017/cases/BH4_Dr31_H066_L061/outputs/openfoam3d/handoff/
```

总包约 **894 MB**。大文件按 90 MB 分片（GitHub 单文件硬限 ~100 MB）。校验：`sha256sum -c SHA256SUMS.txt`。

### 3.1 渲染必需（推荐）

| 文件 | 内容 |
|---|---|
| `polyMesh.tar.xz` | `constant/polyMesh` |
| `fields_early.tar.xz.part-*` | 串行场 `0 … 4` |
| `fields_mid.tar.xz` | 串行场 `7.5 … 9.214` |
| `fields_late.tar.xz.part-*` | 串行场 `11.41 … 13.01`（后期密采样） |
| `bh4_3d.foam` | ParaView 空 sentinel |
| `field_times.txt` / `MANIFEST.json` | 时间列表与清单 |
| `reassemble.sh` | 拼回 `*.part-*` |

共 **24** 个重构串行时间步（本地全部可用 checkpoint）：

```
0, 0.5, 1, 1.5, 3, 3.5, 4,
7.5, 8, 8.5, 9, 9.214,
11.414, 11.514, 11.614,
12.214, 12.314, 12.414, 12.514, 12.614, 12.714, 12.814, 12.914, 13.014
```

### 3.2 探针 / 可选并行场

| 文件 | 内容 |
|---|---|
| `postProcessing.tar.xz` | FO 探针与通量原始输出 |
| `processors_early/late.tar.xz.part-*` | `processor0..3` 全部分解场（resume / 并行可视化） |

### 3.3 几何、日志、运行时字典（本次补传）

| 文件 | 内容 |
|---|---|
| `bh4-physical.stl.xz` | 多实体物理几何（cfMesh 输入） |
| `system_runtime.tar.xz` | `meshDict`、探针点、`endTime.inc` 等 |
| `logs_solver.tar.xz` | 最终求解日志 + checkMesh / cartesianMesh / postprocess |
| `log.compressibleInterFoam.rotated.*.xz` | 更早阶段旋转日志（拼接可还原长跑历史） |
| `log.compressibleInterFoam.aborted_restart.*.xz` | AMI FATAL 后中断的重启段日志 |
| `watchdog_history20.tsv` / `agent_loop_history.tsv` | 20 分钟巡检记录 |
| `SHA256SUMS.txt` | 校验和 |

AMI FATAL 事故摘要（已入库）：`outputs/openfoam3d/ami_event_fatal_20260716.json`（约 `t=11.71 s`，已从 `11.614` 恢复）。

## 4. 本地还原（ParaView）

在仓库根目录：

```bash
CASE=tests/test_02_cong2017/cases/BH4_Dr31_H066_L061/openfoam/3d
HAND=tests/test_02_cong2017/cases/BH4_Dr31_H066_L061/outputs/openfoam3d/handoff

# 1) 校验 + 拼分片
( cd "$HAND" && sha256sum -c SHA256SUMS.txt && ./reassemble.sh )

# 2) 网格 + 全部串行场
tar -xJf "$HAND/polyMesh.tar.xz" -C "$CASE/constant"
tar -xJf "$HAND/fields_early.tar.xz" -C "$CASE"
tar -xJf "$HAND/fields_mid.tar.xz" -C "$CASE"
tar -xJf "$HAND/fields_late.tar.xz" -C "$CASE"

# 3) 可选
tar -xJf "$HAND/postProcessing.tar.xz" -C "$CASE"
xz -dkc "$HAND/bh4-physical.stl.xz" > "$CASE/bh4-physical.stl"
tar -xJf "$HAND/logs_solver.tar.xz" -C "$CASE"

# 4) ParaView
cp "$HAND/bh4_3d.foam" "$CASE/"
# File → Open → $CASE/bh4_3d.foam
# 显示 alpha.water / p / U；动画优先 11.4–13.0 s
```

## 5. 工况要点（对齐论文时用）

- 主管 `D=0.05 m`，有效长 `6.59 m`；三通 `x=3.47 m`；阀 `x=5.98 m`；气囊 `L0=0.61 m`。
- Series B：上游定水头与竖管自由面 `z=0.66 m`；气囊初压大气；`T0=296.15 K`。
- 阀：`cyclicACMI` + 互补壁面；开度 `A/A0=3s²−2s³`，`t_open=0.20 s`。
- 竖管物理高 1.8 m（rim `z=1.85`）；口外另有开放大气域用于判喷发。
- 分类依据：自由面是否越过 rim / 外域出水；不是单独拟合 `vfs/vint`。

## 6. 一般不需要续跑

若必须 resume：

```bash
cd tests/test_02_cong2017/cases/BH4_Dr31_H066_L061/openfoam/3d
# 先解压 processors_* 到 case
BH4_END_TIME=13 BH4_LABEL=base_topen0p20 ./Allrun.resume
```

## 7. 未做（可选后续）

- refined mesh 敏感性
- 阀时 `t_open=0.10 / 0.30 s` 敏感性
- 多工况总表（`summarize_study.py`）
