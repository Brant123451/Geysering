# 本地交接：VW2011 Test 1 Case B · 3D OpenFOAM 复现

> **读本文即可接手。** 云端已停算；请在本机继续。  
> 分支：`cursor/test1-caseb-3d-0614` · PR：https://github.com/brant123451/geysering/pull/10  
> 算例根：`tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/`

---

## 1. 复现的是哪篇论文、哪个 case

| 项 | 值 |
|---|---|
| 论文 | Vasconcelos & Wright (2011), *Journal of Hydraulic Engineering*（仓库 PDF：`references/vasconcelos2011.pdf`） |
| 试验 | **Test 1**（实验室喷涌 / geysering） |
| Case | **Case B**（表 1 与 Fig.6/8 **中心面板**） |
| 几何标识 | \(D_t=12.7\,\mathrm{mm}\)，\(H_{a0}=0.610\,\mathrm{m}\)，\(Y_{fs,0}=0.356\,\mathrm{m}\) |
| 目录名 | `B_Dt12p7_Ha0610_Yfs0356` |
| 目标图/表 | Fig.6（\(H^*\)）、Fig.8（水位）、Table 2 |
| 观察分支 | 小塔（12.7 mm）每次均喷涌 |

**不是** Case A（\(D_t=57.1\,\mathrm{mm}\)），**不是** Fig.10/11 其它组合，**不要**用旧 2D 分支 `cursor/test1-caseb-2d-4ac2`。

详细纸面审计：`openfoam/3d/PAPER_AUDIT.md`。

---

## 2. 几何 / IC / BC（已与论文对齐）

| 量 | 模型值 |
|---|---|
| 主管内径 / 总长 | 0.094 m / 4.006 m |
| 阀面 / 塔心 | \(x=0.546\,\mathrm{m}\) / \(x=3.516\,\mathrm{m}\) |
| 塔内径 / 冠上高度 / rim | 0.0127 m / 0.610 m / \(y=0.657\,\mathrm{m}\) |
| IC | 阀前干空气加压；阀后满水静止；塔水位冠上 0.356 m；\(T=293.15\,\mathrm{K}\)（论文未给温度） |
| BC（闭阀 hold） | 共形无滑移 baffle；壁无滑移；塔顶大气出口 |
| 有意偏差 | 阀非实体蝶阀盘；开阀时程假定 0.25 s；接触角等未测 |

---

## 3. 当前进度（诚实状态）

**论文复现未完成。** 仅完成 base 网格上的短时闭阀筛查与数值轴穷尽；未通过 1.0 s hold，未跑 smoke / 10.5 s 全时窗，未对 Fig.6/8/Table 2 做终态对比。

### 3.1 已准入数值基线（勿回退）

- 离散静水初始化（`CASEB_HYDROSTATIC_INITIALIZATION=discrete`）
- `p_rghFinal=1e-10`
- 共形无滑移闭阀 baffle
- `isoAdvection + plicRDF + RDF`，`interpolateNormal=false`，`curvFromTr=true`
- `maxCo=0.15`，`maxDeltaT=2.5e-4`
- `nCorrectors=2`，`nOuterCorrectors=1`，`nNonOrthogonalCorrectors=0`
- 求解器：TwoPhaseFlow `compressibleInterFlow` @ commit `de9826f9ffb24f4b635ac97fd388ebd560cfc174`
- OpenFOAM.com **v2512**，Gmsh 4.12.x

该基线曾通过：0.006 s 启动筛查、0.04 s 压力漂移窗（\(H^*\mathrm{ptp}\approx0.0009\)）。  
其后 **1.0 s hold 在 ≈0.117 s 因 rim 外气体热点拒绝**（\(y\approx0.657\,\mathrm{m}\)，\(K\approx0\)，\(H^*\) 仍假平静）。

### 3.2 已拒绝、勿再单独重试

| 设置 | 结果 |
|---|---|
| `nCorrectors=3` | rim 推迟到末样本，U≈1.841 |
| `nOuterCorrectors=2` | 短暂 rim → FS → 再 rim，U→1.841 |
| `nCorr=3`+`nOuter=2` | 0.09→0.12 rim 单调增长 |
| `nNonOrthogonalCorrectors=1` | 0.11 rim → 0.12 U=1.841 |
| `nNonOrthogonalCorrectors=2` | onset 提前到 0.08 |
| **`maxCo=0.10`** | onset **0.070 s**，U 单调到 ≈1.85；**勿再压小 maxCo** |

**PIMPLE / nonortho / `maxCo=0.10` 轴已穷尽。** 下一刀须换自由面–压力耦合或其它界面处理，再做闭阀 hold。

### 3.3 已上传的本地数据

路径均相对 case 根  
`tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/`：

| 目录 | 内容 | 状态 |
|---|---|---|
| `outputs/sim_data/maxco010_0p12_screen/` | 已拒 `maxCo=0.10` 的 0.12 s 场/VTK/探针/日志 | 可渲染；见其 `RENDER_HANDOFF.md` |
| `outputs/sim_data/refined_mesh_relocate/` | refined ≈5.52M tet 的 `.msh` + `polyMesh` + checkMesh 日志 | **网格未过严格门**；求解未启动 |

拉取后务必：

```bash
git fetch origin cursor/test1-caseb-3d-0614
git switch cursor/test1-caseb-3d-0614
git lfs pull
```

---

## 4. Refined 网格当前卡点（云端停下的原因）

用户要求停云端改本机。当时状态：

1. `CASEB_MESH=refined`，优化器 `CASEB_MESH_OPTIMIZER=relocate`（环境 Gmsh **无 Netgen**）
2. 生成约 **5,522,016** tet；塔向名义约 **18 cells across**
3. **标准** `checkMesh`：`Mesh OK.`（non-ortho max 68.2°，skew 1.17）
4. **严格** `checkMesh -allTopology -allGeometry`：**Failed 3**：
   - 866 under-determined cells（determinant）
   - 2 low-weight faces（min weight ≈0.0024）
   - 2 low volume-ratio faces
5. `postprocess.py --mesh-only` 退出：`mesh did not meet the documented acceptance criteria`
6. **求解器未启动**（无 `processor*` 场）

例外规则（`postprocess.py`）：仅当严格检查只失败 determinant、且边界 tet excess ≤5 时可 `accepted_boundary_tet_exception`。  
当前失败 3 项 + excess 远超 5 → **不能**走该例外。

归档细节：`outputs/sim_data/refined_mesh_relocate/MESH_HANDOFF.md`。

---

## 5. 本机建议工作顺序

1. **读完**本文 + `openfoam/3d/PAPER_AUDIT.md` + `openfoam/3d/HANDOFF.md`（英文长日志，细节以 HANDOFF 为准）。
2. 环境：OpenFOAM.com v2512 + Gmsh（最好带 Netgen）+ NumPy/Matplotlib；执行 `openfoam/3d/build_twophaseflow.sh` 钉住 TwoPhaseFlow。
3. **优先数值轴（base 网格即可）**：离开已穷尽的 PIMPLE/nonortho/maxCo，针对 **rim 外气体热点**（\(y\approx0.657\)）改自由面/压力耦合；先做 **新鲜 0.12 s 闭阀筛查**，再谈 1.0 s hold。  
   可用 `outputs/sim_data/maxco010_0p12_screen/` 对照 rim 形态。
4. **并行或其后**：修 refined 网格质量（Netgen / 局部加密 / 放宽或重写质量门需有证据），再跑 `./run_refined_0p12_screen.sh`（脚本内已是准入基线 `maxCo=0.15`）。
5. 通过闭阀 hold 后：smoke 0.5 s → base 10.5 s（\(T^*\ge6\)）→ refined + 敏感性 → Fig.6/8/Table 2。
6. **勿**从已拒绝 decomposed 态续跑到 1.0 s；**勿**声称实验复现完成，除非 `outputs/openfoam_3d_metrics.json` 验收字段通过。

### 快速命令

```bash
cd tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d
./build_twophaseflow.sh

# base 网格重新生成 / hold 筛查
./Allclean
CASEB_STAGE=mesh CASEB_MESH=base ./Allrun

./Allclean
# 例：用准入基线做 0.12 s 闭阀（自行设 END_TIME / 或仿 run_*_screen.sh）
CASEB_STAGE=hold CASEB_MESH=base CASEB_VALVE_MODE=closed \
  CASEB_END_TIME=0.12 CASEB_MAX_CO=0.15 OPENFOAM_NP=4 ./Allrun

# refined（需先过网格门）
./run_refined_0p12_screen.sh
```

解压已归档 refined 网格（跳过 gmsh，直接检查/调试）：

```bash
cd ../outputs/sim_data/refined_mesh_relocate
mkdir -p _restore && tar -xJf mesh/polyMesh.tar.xz -C _restore
# 或：tar -xJf mesh/caseB3d.msh.tar.xz -C _restore
```

---

## 6. 关键文件索引

| 文件 | 用途 |
|---|---|
| `LOCAL_HANDOFF.md` | **本文件**：给本机 AI/人的中文交接 |
| `openfoam/3d/HANDOFF.md` | 英文完整筛查编年史 |
| `openfoam/3d/PAPER_AUDIT.md` | 论文参数审计 |
| `openfoam/3d/NEW_AGENT_PROMPT.md` | 云端/新 agent 完整任务提示 |
| `openfoam/3d/README.md` | 算例操作说明 |
| `outputs/sim_data/README.md` | 模拟数据归档索引 |
| `outputs/sim_data/maxco010_0p12_screen/RENDER_HANDOFF.md` | ParaView/VTK 渲染 |
| `outputs/sim_data/refined_mesh_relocate/MESH_HANDOFF.md` | refined 网格归档说明 |
| `config/case.json` | case 定义 |
| `data/digitized/` | Fig.6/8 数字化曲线 |

---

## 7. 交接时云端动作

- 已 `pkill` 停止 `run_refined` / `watchdog` / `gmsh` / `interFlow` / `mpirun`
- 已将 refined `.msh` + `polyMesh` + 日志打包进 git LFS（本目录树下）
- **不再在本云端继续求解**；后续以本机为准

更新时间：2026-07-17（UTC）
