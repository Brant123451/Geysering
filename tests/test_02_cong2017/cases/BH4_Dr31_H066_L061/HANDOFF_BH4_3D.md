# B-H4 3D OpenFOAM 交接文档（草稿 — 基线尚未跑完 13 s）

> 状态：仿真进行中。本文档在跑完 13 s 后会更新为最终版，并附带可渲染场数据打包说明。

## 工况身份

| 项 | 值 |
|---|---|
| 论文 | Cong, Chan & Lee (2017), Test 2 / Series B |
| 工况 | **B-H4**（实验判定 **NO GEYSER**，与 B-H3 成对） |
| 参数 | `Dr=0.031 m`，`Dr/D=0.62`，`H0=0.66 m`，`L0=0.61 m` |
| 分支 | `cursor/test2-bh4-3d-3d79` |
| PR | https://github.com/brant123451/geysering/pull/16 |
| Case 根目录 | `tests/test_02_cong2017/cases/BH4_Dr31_H066_L061/` |
| OpenFOAM case | `.../openfoam/3d/` |
| 求解器 | `compressibleInterFoam`（水 `rhoConst`，空气 perfect gas） |

## 目录地图

```
BH4_Dr31_H066_L061/
  openfoam/3d/                 # 真实 3D VOF case
    0.orig/                    # 初始场模板
    system/                    # controlDict / fvSolution / meshDict / ACMI …
    constant/                  # 物性、polyMesh（运行时生成，默认 gitignore）
    processor0..3/             # 并行分解场（默认 gitignore；完成后打包上传）
    postProcessing/            # 探针/通量时序（默认 gitignore；完成后打包）
    make_mesh.py, Allrun, Allrun.resume, postprocess_compare.py, …
  outputs/openfoam3d/          # 已跟踪的紧凑产物（metrics/csv/png/gate JSON）
  openfoam/3d/PAPER_AUDIT.md   # 论文几何/IC/BC 一致性审计
  config/ / reference/ …
```

## 数值设置（生产基线 `base_topen0p20`）

- `maxCo=0.40`，`maxAlphaCo=0.20`，`nAlphaSubCycles=4`，`nOuterCorrectors=3`
- 阀开启时间 `t_open=0.20 s`，开度律 `A/A0=3s²−2s³`（cyclicACMI）
- `endTime=13.0 s`（完整事件窗）
- `writeInterval=0.10 s`；为交接渲染，后期已将 **`purgeWrite=0`**（需重启后生效；已从最新场 resume）
- 网格：cfMesh cartesian，约 1.76e5 cells（见 metrics 内 mesh 段）

## 如何本地渲染（算完后）

1. 拉取本分支，进入 `openfoam/3d/`。
2. 解压交接包（完成后会放到 `outputs/openfoam3d/handoff/`）：
   - `polyMesh.tar.*` → `constant/polyMesh/`
   - `fields_*.tar.*` 或 reconstructed 时间目录
   - `postProcessing.tar.*` → `postProcessing/`
3. ParaView：`File → Open` 打开 `*.foam`（或 reconstructed 时间目录 + polyMesh），显示 `alpha.water` / `p` / `U`。
4. 立管中心线与外域喷出判据已在 `postprocess_compare.py` / metrics 中定义；渲染时建议：
   - 竖直立管中心线切片看自由面 `Yfs`
   - 外空气域看是否有 `alpha.water>0.5` 出流

## 已提交的紧凑产物

见 `outputs/openfoam3d/base_topen0p20_{metrics,timeseries,comparison}.*`  
以及 `ami_event_fatal_20260716.json`（cyclicACMI AMI/`updateAreas` 一次 FATAL 与看门狗误杀修复记录）。

## 恢复命令

```bash
cd tests/test_02_cong2017/cases/BH4_Dr31_H066_L061/openfoam/3d
BH4_END_TIME=13 BH4_LABEL=base_topen0p20 ./Allrun.resume
```

## 未完成项（跑完后勾掉）

- [ ] 基线跑满 13 s 并给出最终 `classification_3d`
- [ ] 打包 polyMesh + 可用时间场 + postProcessing 上传到 `outputs/openfoam3d/handoff/`
- [ ] 更新本文档为最终版（含时间列表、分类结论、渲染建议）
- [ ] refined mesh / `t_open` 0.10/0.30 敏感性（当前为保 CPU 暂停）
