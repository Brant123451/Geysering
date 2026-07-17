# B-H4 3D OpenFOAM 交接文档（最终版）

基线 `base_topen0p20` 已跑满事件窗；3D 分类为 **NO_GEYSER**，与实验一致。本目录可供你在本地解压场数据并用 ParaView 渲染。

## 工况身份

| 项 | 值 |
|---|---|
| 论文 | Cong, Chan & Lee (2017), Test 2 / Series B |
| 工况 | **B-H4**（实验 **NO GEYSER**，与 B-H3 成对） |
| 参数 | `Dr=0.031 m`，`Dr/D=0.62`，`H0=0.66 m`，`L0=0.61 m` |
| 分支 | `cursor/test2-bh4-3d-3d79` |
| PR | https://github.com/brant123451/geysering/pull/16 |
| Case | `tests/test_02_cong2017/cases/BH4_Dr31_H066_L061/` |
| OpenFOAM | `.../openfoam/3d/` |
| 求解器 | `compressibleInterFoam`（水 `rhoConst`，空气 perfect gas） |
| 基线标签 | `base_topen0p20`（阀 `t_open=0.20 s`） |

## 最终结论（紧凑产物）

文件：`outputs/openfoam3d/base_topen0p20_{metrics.json,timeseries.csv,comparison.png}`

| 量 | 3D | 实验 |
|---|---|---|
| 分类 | **NO_GEYSER** | NO_GEYSER |
| 匹配 | **True** | — |
| `Ta` | 8.41 s | 8.14 s |
| `Yfs_max`（冠上） | 1.59 m | rim=1.8 m（未过 rim） |
| 外域出水体积 | ~0（数值噪声量级） | — |
| 事件窗 | 完整（`simulated_end_time_s≈13.03`） | 13 s |

数值控制：`maxCo=0.40`，`maxAlphaCo=0.20`，`nAlphaSubCycles=4`，`nOuterCorrectors=3`。  
论文几何/IC/BC 审计见 `openfoam/3d/PAPER_AUDIT.md`。

## 交接数据包位置

全部在：

`outputs/openfoam3d/handoff/`

| 文件 | 内容 |
|---|---|
| `polyMesh.tar.xz` | `constant/polyMesh`（渲染必需） |
| `postProcessing.tar.xz` | 探针/通量时序原始 FO 输出 |
| `fields_early.tar.xz` | 重构后串行场：`0,4,8,9` |
| `fields_late.tar.xz.part-00/01` | 重构后串行场：`11.41…13.01`（需先拼回） |
| `processors_*.tar.xz.part-*` | 并行 `processor0..3` 全部分解场（可选，用于 resume/并行可视化） |
| `reassemble.sh` | 把 `*.part-*` 拼回完整 `.tar.xz` |
| `MANIFEST.json` / `field_times.txt` | 时间列表与清单 |
| `bh4_3d.foam` | 空 sentinel；解压到 case 后可复制/打开 |

> GitHub 单文件约 100 MB 限制，故较大包已按 90 MB 分片。

## 本地还原与渲染（推荐路径）

在仓库根目录：

```bash
CASE=tests/test_02_cong2017/cases/BH4_Dr31_H066_L061/openfoam/3d
HAND=tests/test_02_cong2017/cases/BH4_Dr31_H066_L061/outputs/openfoam3d/handoff

# 1) 拼回分片
( cd "$HAND" && ./reassemble.sh )

# 2) 解压网格 + 重构场（ParaView 最省事）
tar -xJf "$HAND/polyMesh.tar.xz" -C "$CASE/constant"
tar -xJf "$HAND/fields_early.tar.xz" -C "$CASE"
tar -xJf "$HAND/fields_late.tar.xz" -C "$CASE"

# 3) （可选）探针原始数据
tar -xJf "$HAND/postProcessing.tar.xz" -C "$CASE"

# 4) （可选）完整并行分解场
tar -xJf "$HAND/processors_early.tar.xz" -C "$CASE"
tar -xJf "$HAND/processors_late.tar.xz" -C "$CASE"
```

ParaView：

1. `File → Open` → 打开 `$CASE`（或把 `handoff/bh4_3d.foam` 拷到 `$CASE/bh4_3d.foam` 再打开）。
2. 勾选时间步；显示 `alpha.water`（自由面）、`p` / `p_rgh`、`U`。
3. 建议视图：
   - 立管中心竖直切片看 `Yfs`（冠上最大约 1.59 m，rim=1.8 m）
   - 外空气域检查是否有水体外溢（本基线无实质喷出）
   - 气囊区域看压力响应（PT1 为口袋体积平均代理）

## 模拟说明（给你本地渲染时对齐语义）

- 主水平管 `D=0.050 m`，有效长度约 6.59 m；三通、阀、口袋位置按论文 Fig.1 / `PAPER_AUDIT.md`。
- Series B IC：恒定水位 / 立管自由面 `z=0.66 m`；口袋大气压；`T0=296.15 K`。
- 阀：`cyclicACMI` + 互补壁面；开度律 `A/A0=3s²−2s³`，`t_open=0.20 s`。
- 后期为保留场文件使用了 `purgeWrite 0`；因此 `12.2…13.0` 附近时间步较密，适合动画。
- 中途曾出现一次 cyclicACMI `updateAreas` AMI/event FATAL（约 11.71 s），已从场 `11.614` 恢复；记录见 `outputs/openfoam3d/ami_event_fatal_20260716.json`。
- `vfs/vint` 的 3D 值显著高于实验（与既有 1D/判定逻辑一致）；**分类依据是自由面是否越过 rim / 外域出水**，不是单独校准速度去拟合实验。

## 恢复续跑（一般不需要）

```bash
cd tests/test_02_cong2017/cases/BH4_Dr31_H066_L061/openfoam/3d
BH4_END_TIME=13 BH4_LABEL=base_topen0p20 ./Allrun.resume
```

## 未做项（可选后续）

- refined mesh 敏感性
- 阀时 `t_open=0.10 / 0.30 s` 敏感性（worktree 曾准备，为保 CPU 暂停）
- `summarize_study.py` 多工况总表（目前单基线已完整）
