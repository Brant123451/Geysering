# 渲染交接：Case B 3D · maxCo=0.10 闭阀筛查（已拒绝）

面向本地 ParaView / VTK 渲染。本目录只服务这一次 **0.12 s 闭阀 rim-onset 筛查**，不是通过的 hold，也不是论文复现终态。

## 算例身份

| 项 | 值 |
|---|---|
| Case | VW2011 Test 1 Case B（`B_Dt12p7_Ha0610_Yfs0356`） |
| 分支 | `cursor/test1-caseb-3d-0614` |
| PR | https://github.com/brant123451/geysering/pull/10 |
| 筛查 ID | `tight_pressure_maxco_0p10_screen_0p12_s` |
| 判定 | **拒绝**（rim 外气体热点单调增长） |
| 求解器 | TwoPhaseFlow `compressibleInterFlow` @ `de9826f9…` |
| 网格 | base，约 1.90e6 cells，4 核分解 |
| 阀门 | 闭阀 `conformalNoSlipBaffle` |
| 界面 | `isoAdvection + plicRDF + RDF`，`interpolateNormal=false` |
| 时间步 | `maxCo=0.10`，`maxDeltaT=2.5e-4`，自适应 |
| PIMPLE | `nOuter=1`，`nCorrectors=2`，`nNonOrthogonal=0` |
| 写盘 | 场间隔 0.01 s；探针更密 |
| 终时 | 0.1200015 s |

## 目录结构

路径相对 case 根目录  
`tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/`：

```
outputs/sim_data/maxco010_0p12_screen/
  RENDER_HANDOFF.md          ← 本文件
  fields/
    processor0_all_times.tar.xz   # 分解场（按 rank 分卷，避开 GitHub LFS 2GB 上限）
    processor1_all_times.tar.xz
    processor2_all_times.tar.xz
    processor3_all_times.tar.xz
    vtk_key_times.tar.xz          # 关键时刻 VTK（推荐直接渲染）
    case_deck/                    # system/ + 部分 constant（配置快照）
  vtk_index/
    3d.vtm.series                 # ParaView 时间序列索引
    time_map.tsv                  # VTK 文件夹名 ↔ 物理时间
  postProcessing/                 # transducer / tower / plume 探针
  runtime/                        # 后处理 CSV/PNG、run_manifest、metrics
  diagnostics/
    caseb_bounds_history.csv      # CASEB_BOUNDS 全序列（含 mode）
    caseb_accounting_history.csv
    metrics_*.json
    run_manifest.json
  logs/
    log.compressibleInterFlow     # 完整求解日志
    log.postprocess
    log.foamToVTK / log.reconstructPar
    …
```

大文件通过 Git LFS 跟踪（`*.tar.xz`）。拉取后执行：

```bash
git lfs pull
```

## 结果摘要（渲染时应看到什么）

`CASEB_BOUNDS` 速度最大值位置：

| t (s) | \|U\|max (m/s) | y (m) | 模式 |
|------:|---------------:|------:|------|
| 0.01–0.06 | 0.98–2.02 | ≈0.404 | 自由面伪流（可接受筛查噪声） |
| 0.070 | 1.099 | 0.657 | **rim 外气体** onset |
| 0.080 | 1.242 | 0.657 | rim 增长 |
| 0.090 | 1.395 | 0.657 | rim 增长 |
| 0.100 | 1.548 | 0.657 | rim 增长 |
| 0.110 | 1.698 | 0.657 | rim 增长 |
| 0.120 | 1.849 | 0.657 | rim 终态 |

Rim 特征：`alpha.water≈0`，`rho≈1.204`，`K≈0`，位置在塔 rim 外侧气体区  
（约 \(x\approx3.52\)，\(y\approx0.657\)）。  
换能器 \(H^*\) peak-to-peak ≈ 0.00143，**不能**当作通过。

## 推荐渲染路径（VTK，最简单）

```bash
cd tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/outputs/sim_data/maxco010_0p12_screen
mkdir -p _render && tar -xJf fields/vtk_key_times.tar.xz -C _render
# 打开：
#   _render/vtk/3d.vtm.series
# 或单帧：_render/vtk/3d_4416.vtm  (t≈0.070 onset)
#         _render/vtk/3d_11532.vtm (t≈0.120 final)
```

ParaView 建议：

1. 打开 `3d.vtm.series`，字段至少加载 `alpha.water`、`U`、`p_rgh`。
2. 体积：`alpha.water` Contour = 0.5（自由面）；或 Slice 过塔轴。
3. 气体热点：Threshold `alpha.water < 1e-3`，再对 `U` 做 Glyph / Contour。
4. 相机对准塔 rim 附近 \((x,y)\approx(3.52, 0.657)\)。
5. 时间轴对比 0.06（仍偏自由面）→ 0.07（onset）→ 0.12（U≈1.85）。

`vtk_index/time_map.tsv`：

| 文件夹 | t (s) |
|--------|------:|
| 3d_0 | 0 |
| 3d_3428 | 0.0600 |
| 3d_4416 | 0.0700 |
| 3d_5550 | 0.0800 |
| 3d_6829 | 0.0900 |
| 3d_8253 | 0.1000 |
| 3d_9820 | 0.1100 |
| 3d_11532 | 0.1200 |

每帧场：`alpha.water`、`U`、`p`、`p_rgh`、`T`、`rho`，含 walls / atmosphere / valve 边界。

## 完整 OpenFOAM 场（分解态）

若要用原生 OpenFOAM reader 或自行 `reconstructPar`：

```bash
cd outputs/sim_data/maxco010_0p12_screen
mkdir -p _of
for p in 0 1 2 3; do tar -xJf fields/processor${p}_all_times.tar.xz -C _of; done
# 得到 _of/processor0..3 ，内含全部 0.01 s 写盘时刻 + constant/polyMesh
```

ParaView：打开任一 `processor*/` 旁的 case 结构，或用 decomposed OpenFOAM reader。  
也可把 `fields/case_deck/system` 拷到临时 case 根，与解压出的 `processor*` 并列后重建。

写盘时刻包括：  
`0, 0.01…0.12`（共 13 个时间目录，含完整场：`U, alpha.water, p, p_rgh, T, rho, K_, phi, …`）。

## 探针与曲线

- `postProcessing/transducer/0/`：压力 / alpha / U  
- `postProcessing/towerProfiles/0/`、`plumeProfiles/0/`：剖面  
- `runtime/openfoam_3d_pressure_series.csv`、`*_levels_series.csv`  
- `runtime/openfoam_3d_*_comparison.png`：后处理对照图（非论文终稿）  
- `diagnostics/caseb_bounds_history.csv`：逐写盘 Umax 位置与 mode

## 不要据此声称的事项

- 不是 1.0 s closed-valve hold 通过  
- 不是 0.5 s smoke / 10.5 s 全时窗  
- 不是 Fig.6 / Fig.8 / Table 2 复现完成  
- `maxCo=0.10` **已拒绝**，勿当作新 baseline 续跑 hold  

## 下一步（给继续算的人）

PIMPLE 修正次数、非正交修正、以及 `maxCo=0.15→0.10` 均已在同一 rim 模式上拒绝。  
不要再单独继续压小 `maxCo`。需要新的数值轴（自由面/压力耦合或其它界面处理），并先读  
`openfoam/3d/HANDOFF.md` 与 `outputs/openfoam_3d_numerical_diagnostics.json`。
