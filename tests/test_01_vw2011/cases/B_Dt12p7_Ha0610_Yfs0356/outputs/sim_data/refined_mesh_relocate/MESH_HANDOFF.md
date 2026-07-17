# Refined 网格归档（Relocate，未通过严格门）

面向本机继续网格修复 / 跳过 gmsh 直接检查。  
**不是**通过的 hold，**没有**求解场。

## 身份

| 项 | 值 |
|---|---|
| Case | VW2011 Test 1 Case B |
| 预设 | `CASEB_MESH=refined` |
| 优化器 | `CASEB_MESH_OPTIMIZER=relocate`（云端 Gmsh 无 Netgen） |
| 单元数 | 5,522,016 tet |
| 塔向名义分辨率 | ≈18 cells across \(D_t=12.7\,\mathrm{mm}\) |
| 标准 checkMesh | **Mesh OK.** |
| 严格 checkMesh | **Failed 3**（见下） |
| 求解 | **未启动**（`postprocess --mesh-only` 拒绝后退出） |

## 严格检查失败项

摘自 `logs/log.checkMesh.strict`：

- 866 cells with small determinant (`underdeterminedCells`)
- 2 faces with small interpolation weight (`lowWeightFaces`，min≈0.00239)
- 2 faces with small volume ratio (`lowVolRatioFaces`，min≈0.00240)

`postprocess.py` 的 `accepted_boundary_tet_exception` 要求：严格仅失败 determinant，且 boundary tet excess ≤5。  
当前 **不满足** → `logs/log.meshEvidence`：`mesh did not meet the documented acceptance criteria`。

## 目录

```
outputs/sim_data/refined_mesh_relocate/
  MESH_HANDOFF.md          ← 本文件
  MANIFEST.json
  run_refined_0p12_screen.sh
  watchdog_refined.sh
  mesh/
    caseB3d.msh.tar.xz     # gmsh 原生网格（解压得 caseB3d.msh）
    polyMesh.tar.xz        # OpenFOAM constant/polyMesh（含 sets）
  logs/                    # checkMesh / gmsh / meshEvidence / …
  runtime/                 # mesh_refined.json, preflight.json, …
  diagnostics/             # cellToRegion 等
  case_deck/               # system/ + 部分 constant 源文件快照
```

```bash
git lfs pull
cd tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/outputs/sim_data/refined_mesh_relocate
mkdir -p _restore
tar -xJf mesh/polyMesh.tar.xz -C _restore
# 得到 _restore/polyMesh/ → 可拷到 openfoam/3d/constant/
tar -xJf mesh/caseB3d.msh.tar.xz -C _restore
```

## 本机建议

1. 若本机 Gmsh **有 Netgen**：设 `CASEB_MESH_OPTIMIZER=netgen` 重新 `CASEB_STAGE=mesh CASEB_MESH=refined ./Allrun`。
2. 或基于本归档诊断 2 个低权/体积比面，局部修网格后再过 `postprocess.py --mesh-only`。
3. 网格门通过后，用 case 内 `./run_refined_0p12_screen.sh`（准入基线 `maxCo=0.15`，**不要**用已拒的 0.10）。
4. 总交接见 case 根目录 `LOCAL_HANDOFF.md`。
