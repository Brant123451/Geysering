# C9 Phase1 渲染交接（本地 ParaView / OpenFOAM）

面向本地生成渲染图：本目录已打包 **phase1 已完成** 的可用三维场与探针时程。

## 状态结论

- **Phase1 已完成**：求解器 `endTime = 6.75`（论文 phase1 ≈ 6.50 s），日志出现 `End` / `Finalising parallel run`。
- **Phase 2 and eight eruptions have not yet been reproduced.**
  尚未启动 `./Allrun.resume full`（目标 solver `20.25` / 论文 20 s）。

## Git / 路径

- 仓库：`https://github.com/brant123451/geysering`
- 分支：`cursor/c9-openfoam-3d-bf97`
- PR：`https://github.com/brant123451/geysering/pull/8`
- 本目录（相对仓库根）：

```text
tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/artifacts/phase1_render/
```

大文件（`*.tar.xz`）经 **Git LFS** 跟踪。克隆后请执行：

```bash
git lfs install
git lfs pull
```

## 模拟摘要（便于对齐论文）

| 项 | 值 |
|----|----|
| 工况 | Liu et al. (2020) Test 3 / Series C / Case C9 |
| 求解器 | OpenFOAM v2512 `compressibleInterFoam` |
| 网格 | ~481,874 cells（`cartesianMesh`） |
| 并行 | 4 ranks |
| EOS | 空气 perfectGas；水 `perfectFluid`（约 305 m/s 管波速等效） |
| 湍流 | `kOmegaSST` |
| Co 限制 | `maxCo=0.70`，`maxAlphaCo=0.20` |
| 气囊示踪 | `pocketBodyTracer` / `pocketBodyTracerSigma`（守恒相质量通量） |
| Phase1 起止 | 自 `1.2289420474` 重启 → `6.75` |
| 守恒门 | 通量修正残差 `bal_rel ~ 1.6e-4`（通过）；库存 alone 外泄属大气边界 |

`purgeWrite 4` 导致 **中间时刻检查点被清理**。可用三维快照时刻见 `MANIFEST.json` 的 `available_times`：

- 早期：`0` … `1.2289420474`（含 smoke / phase1 起点）
- 晚期：`5.9789420474` … `6.7289420474`（phase1 末段）
- **缺口**：约 `1.23`–`5.98` 之间无体场快照  
  （但 `postProcessing/` 仍有全时段探针/通量时程，可画曲线）

## 本目录内容

| 文件/目录 | 用途 |
|-----------|------|
| `VTK.tar.xz` (~454 MB) | ParaView 主渲染包（推荐） |
| `case.vtm.series` | VTK 系列索引（归档内也有） |
| `reconstructed_render_fields.tar.xz` (~290 MB) | 串行 OpenFOAM 场（需配合网格） |
| `mesh/polyMesh.tar.xz` (~20 MB) | `constant/polyMesh` |
| `postProcessing/` | PT 探针、升管中线、通量、积分等 |
| `results/` | smoke / phase1 守恒门 JSON 等 |
| `logs/` | 20 min 巡检与 phase1 日志头尾摘录 |
| `MANIFEST.json` | 机器可读清单 |

体场字段：`alpha.water`、`U`、`p`、`pocketBodyTracer`、`pocketBodyTracerSigma`、`rho`（VTK 包内）。

## 本地渲染（推荐：ParaView）

```bash
cd tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/artifacts/phase1_render
mkdir -p VTK && tar -xJf VTK.tar.xz -C .
# 若 tar 根目录已是 VTK/，则：
# tar -xJf VTK.tar.xz
paraview VTK/case.vtm.series
# 或打开本目录旁的 case.vtm.series（指向 VTK/ 下各 .vtm）
```

建议可视化：

1. `alpha.water` 等值面 / 切片（气液界面）
2. `pocketBodyTracerSigma` 或 `pocketBodyTracer`（主体气囊示踪）
3. `p` 或 `U` 切片对照 PT / 升管
4. 用 series 播放早期 smoke → 晚期 phase1 末段（注意时间跳跃）

## 可选：OpenFOAM 原生重开

```bash
CASE=../../case   # 指向 openfoam/3d/case
mkdir -p "$CASE/constant"
tar -xJf mesh/polyMesh.tar.xz -C "$CASE/constant"
tar -xJf reconstructed_render_fields.tar.xz -C "$CASE"
# 然后用 paraFoam / foamToVTK 或自己的后处理脚本
```

## 曲线后处理（无需三维）

`postProcessing/` 含全 run 时程，例如：

- `probesPT/` — PT1–PT4 压力
- `riserCentreline/` — 竖管中线
- `atmosphereFlux/`、`gateFlux/`、`inletFlux/`
- `totalPocketBodyTracerMass*`、区域质量积分

可直接画论文对比曲线，不依赖 VTK。

## 下一步（求解侧，尚未做）

1. `./Allrun.resume full` → solver `20.25`（论文 20 s）
2. 后处理对齐喷发计数与相位
3. 气囊/尾门/网格敏感性

再次强调：**Phase 2 and eight eruptions have not yet been reproduced.**
