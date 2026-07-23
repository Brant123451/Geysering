# Case A 已入库计算结果清单

**分支**：`cursor/test1-case-a-openfoam-78de`  
**更新**：2026-07-23  

云端 wall-refined 运行时场（`processor*` / `polyMesh` / `log.*`）在环境重置后**不存在**，无法上传。  
下列文件均为仓库中已跟踪的**紧凑计算结果**（本清单仅作索引）。

---

## 1. OpenFOAM 3D（本轮主验证）

路径：`openfoam/3d/outputs/`

| 文件 | 说明 |
| --- | --- |
| `openfoam_3d_metrics.json` | 总指标（网格、H*/Yfs*/V*、liftoff/catch、质量漂移等） |
| `openfoam_3d_series.csv` | 压力/液位时间序列 |
| `openfoam_3d_levels.csv` | 界面/自由面轨迹 |
| `openfoam_3d_levels_centerline_audit.csv` | 中心线审计 |
| `openfoam_3d_levels_comparison.png` / `.pdf` | Fig.7 对比图 |
| `openfoam_3d_pressure_comparison.png` / `.pdf` | Fig.5 对比图 |
| `openfoam_3d_tower_sections.csv` | 塔截面 `areaAverage(alpha.water)` |
| `openfoam_3d_plume.csv` | 羽流/越顶相关 |
| `openfoam_3d_water_mass.csv` | 质量守恒采样 |
| `openfoam_3d_interface_sensitivity.csv` | 鼻端阈值 0.80–0.95 |
| `HANDOFF_STATUS.json` | 交接机器可读状态 |

**对应工况（已完整 9 s）**：均匀 core `8 mm`，**342,135** 四面体。

| 量 | 值 |
| --- | ---: |
| `H*` | 0.552 |
| `Yfs*` | 0.623 |
| `V*` | 0.465（相对 Fig.7 约 +8%） |
| 喷发 | 否 |

配置与本地续跑说明：`openfoam/3d/HANDOFF_LOCAL.md`、`openfoam/3d/README.md`。

---

## 2. OpenFOAM 2D（早期平面模型）

路径：`openfoam/2d/outputs/`

- `openfoam_2d_metrics.json`
- `openfoam_2d_series.csv`
- `openfoam_2d_levels.csv`
- `openfoam_2d_levels_comparison.png` / `.pdf`
- `openfoam_2d_pressure_comparison.png` / `.pdf`

（2D 不能保持圆管/圆塔面积比，仅作对照，不是最终保真模型。）

---

## 3. 案例级其它输出

路径：`outputs/`

- `caseA_comparison_metrics.json`、`caseA_model_series.csv`、`caseA_table2_velocities.json`
- `caseA_comparison_*.png`、`caseA_*_manual.png/.pdf`、`caseA_fig4_*`、`caseA_tpa_redrawn.*`
- `caseA_animation.gif`、`frames_index.json`、`report.html`

---

## 4. 数字化试验数据

路径：`data/digitized/`

- `fig5_caseA_Hstar_band.csv`、`fig7_caseA_levels.csv`
- 面板/调试图与 `manual/` 下重复曲线 CSV

---

## 5. 明确未上传（不存在或被 gitignore）

- wall-refined ~463k 的 `processor*` / 时间场 / `caseA3d.msh` / `log.compressibleInterFoam`
- 任何未完成的 wall-refined 后处理新 `outputs/`（需本机重跑后提交）

---

拉取本分支即可获得以上全部已算数据：

```bash
git fetch origin cursor/test1-case-a-openfoam-78de
git checkout cursor/test1-case-a-openfoam-78de
```
