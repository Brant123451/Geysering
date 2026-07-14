# Complete continuation prompt

Copy everything below into the new Cursor Cloud Agent. The intended agent name
is `geysering test 3 caseC9`.

---

你负责继续完成 **Geysering Test 3 — Case C9 三维 OpenFOAM 验证**。

## 1. Git 与交接入口

- Repository: `https://github.com/brant123451/geysering`
- Existing PR: `https://github.com/brant123451/geysering/pull/8`
- Continue the existing branch: `cursor/c9-openfoam-3d-bf97`
- Base branch: `main`
- Baseline handoff commit: `4cedc63`
- Always use the current remote branch tip, which contains this complete prompt.
- Agent name: `geysering test 3 caseC9`
- Work on the current PR branch; do not create a parallel implementation
  branch unless the user explicitly requests one.
- If using the Cloud Agent API, target PR #8 and set
  `workOnCurrentBranch: true`.

新 Agent 没有旧 Agent 的运行目录或对话上下文。首先从仓库根目录依次阅读：

1. `tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/HANDOFF.md`
2. `tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/PAPER_AUDIT.md`
3. `tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/README.md`
4. `tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/case_parameters.json`
5. `tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/outputs/openfoam_3d_metrics.json`
6. C9 的 `README.md`、`manifest.yaml`、`config/case.json`、
   `data/digitized/`、`model/`
7. `openfoam/3d/` 下全部生成器、运行脚本和后处理脚本

若论文 PDF、paper scans 或引用文件在新环境中不存在，必须明确记录为缺失输入；
不得假装重新读取了不存在的文件，也不得凭印象制造页码。现有
`PAPER_AUDIT.md` 保存了旧 Agent 已完成的带页码审计。

## 2. 最终目标

建立并验证真实的三维、气液两相、气体可压缩 OpenFOAM 模型，重现 Liu
et al. (2020) Test 3 / Series C / Case C9：

- 初始水力瞬变驱动的 phase 1；
- 上游封闭气囊压缩、膨胀、输运并到达 junction chamber；
- 气囊释放驱动的 phase 2；
- 论文总计约 8 次喷发，其中前 2 次属于 phase 1，后 6 次属于 phase 2。

不能把现有一维结果改名为三维结果。只有真实的三维 OpenFOAM 求解历史才能
支持“复现”。如果 20 s phase 2 未完成或结果不匹配，必须明确报告失败或
不完整，不能补造事件。

## 3. 允许修改范围

代码修改限制在：

`tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/`

不要提交：

- OpenFOAM 时间目录；
- `processor*`；
- `postProcessing/`；
- `constant/polyMesh/`；
- 大型临时网格、日志、core 或 VTK 文件。

必须提交小型、可复现的源码、配置、CSV、JSON、PNG 和文档。

## 4. 已完成工作，不要从零重写

当前 PR 已包含：

- 带论文页码的 C9 审计；
- 参数来源与模型假设分离的 `case_parameters.json`；
- 完整 5.80 m 上游管、0.30 m chamber、5.95 m 下游管、1.22 m riser、
  resolved tailgate 和外部 atmospheric plume 的三维几何；
- OpenFOAM v2512 `compressibleInterFoam` 模型；
- perfect-gas air；
- `perfectFluid` 弱可压缩水，默认用论文约 305 m/s 亚克力管波速对应的
  92.86 MPa 全域等效模量，2.2 GPa 本征水体模量作为敏感性；
- C9 \(Re=O(10^5)\) 的默认 RANS \(k\)-\(\omega\) SST 闭合，laminar
  仅作敏感性；
- 仅初始化于厚气囊主体、排除薄冠层，并使用气相质量通量而非混合物
  `rhoPhi` 输运的守恒 `pocketBodyTracer`；
- VOF、重力、表面张力、接触角；
- MULES 默认界面输运和 isoAdvector 敏感性入口；
- `cartesianMesh`/cfMesh 默认网格和保留的
  `blockMesh` + `snappyHexMesh` 敏感性流程；
- 初始化、smoke、phase1、full、重构和后处理脚本；
- PT1–PT4、riser centreline、upstream crown、区域体积/质量、
  边界通量和极值诊断；
- 质量守恒、气相守恒、气囊输运、喷发事件、图表和敏感性脚本。

不要恢复已证明错误的实现：

- `compressibleInterIsoFoam` 不是“等温气体求解器”，它仅作为
  isoAdvector 界面输运敏感性；
- `compressibleInterFoam` 不接受 `-dict system/controlDict.*`；
- 气相守恒不能读取运行时不存在的 `alphaRhoPhi.*`；
- 守恒必须使用 `rhoPhi` 的质量通量列，而不是 `phi` 体积通量列；
- 当前实现使用
  `interpolate(rho.air)*(phi - alphaPhi0.water)` 直接构造气相质量通量；
- v2512 的 `scalarTransport` 可压缩质量通量分支不执行 `bounded01`；
  禁止恢复逐步清除/截断投影，该方案在论文时间 0.3901 s 已损失
  7.65% 示踪库存；
- 人工载体密度和显式重建时间层的两版可变密度 MULES 均已失界，
  后者在 solver time `0.0202 s` 达到约 `1e24`，禁止恢复；
- 当前 `boundedPhaseMassTransport` 先将气相质量通量投影到离散气相
  连续方程，再以可压缩 MULES 推进
  `ddt(alpha*rho, s) + div(phi, s)`（`s` 限制在 `[0,1]`），
  库存 `sigma = alpha*rho*s`；`Allrun.initialize` 会自动编译；
- 强度量 Sp 形式 `fvm::ddt(alpha,rho,s) - Sp(ddt(alpha,rho)+div(phi),s)`
  在载体投影很紧时 Sp≈0，退化为已拒绝的 product-ddt 形式，
  solver time ~0.52 s 损失约 5.9% 示踪质量，禁止恢复；
- 未加 Sp 的 `fvm::ddt(alpha,rho,s)` 在 0.35 s 损失 1.93%，禁止恢复；
- 将 `sigma` 裁剪到 `[0,αρ]` 可保持 s∈[0,1]，但在边界通量≈0 时
  仍以约 0.5%/0.01 s 销毁库存，禁止作为可接受方案；
- `phi/(αρ)_f` 面速度在薄相面上会爆炸；
- 未裁剪的 `sigma := sigmaOld - dt*div(flux(phi,s))` 在 ~1e-4 s
  恢复 `s>2`，轨迹未接受；
- 到达判据使用物理 `alpha.air*rho.air` 库存，并用三个开放边界
  示踪通量闭合质量预算；预算误差或累计数值示踪平衡残差超过初始
  示踪质量 1% 时判据无效；
- tailgate 不使用会再次扣除速度头的 `prghTotalPressure`；
- 当前 tailgate `p_rgh` 使用静水 tailwater closure；
- STL 接口必须保持共形闭合，不要重新引入 penetration/lip 几何泄漏。

## 5. 论文确定的物理参数与目标

主要几何及工况：

- upstream pipe: `Lu = 5.80 m`, `Du = 0.20 m`, slope `1:100`；
- chamber: `0.30 × 0.30 × 0.45 m`；
- invert drop: `0.18 m`；
- downstream pipe: `Ld = 5.95 m`, `Dd = 0.28 m`；
- riser: diameter `0.06 m`, length `1.22 m`；
- initial flow `Q0 = 0.025 m3/s`；
- final flow `Q1 = 0.040 m3/s`；
- valve ramp `0.40 s`；
- initial riser column `0.30 m`；
- temperature about `293.15 K`；
- riser outlet open to atmosphere；
- downstream initially full, controlled by movable tailgate。

关键验证目标：

- PT2 first peak: `10.69 kPa gauge` at paper time `0.50 s`；
- first free surface reaches riser top: `0.73 s`；
- Eq. (8) oscillation period: `1.45 s`；
- phase-1 second geyser ends around `3.99 s`；
- main upstream air pocket reaches chamber: `6.46 s`；
- third geyser starts around `6.70 s`；
- total experimental eruption count: `8`；
- final steady pressure:
  - PT2 `8.79 kPa`；
  - PT3 `12.76 kPa`；
  - PT4 `9.25 kPa`；
- Fig. 11 中最大 PT2、喷射高度和 final pressure 的关系也要比较。

PT locations and all page citations are in `PAPER_AUDIT.md`; use those values
rather than guessing wall-tap coordinates.

## 6. 不确定参数与禁止调参规则

论文没有报告：

- 初始气囊长度/体积；
- 气囊 upstream/downstream interface 坐标；
- 独立列出的初始气囊压力；
- tailgate 几何开度。

因此这些量必须标记为 prior、closure 或 sensitivity，不能伪装为论文测量值。

当前三组 pocket prior：

- `pocket_small`: analytic volume about `4.641 L`；
- `base`: about `12.642 L`；
- `pocket_large`: about `21.608 L`。

当前 gate closure：

- target effective discharge area `0.00823 m2`；
- resolved sharp-opening coefficient `Cd = 0.817`；
- geometric area `0.01008 m2`。

这个 gate closure 只由已知初始水力状态得到，不能按喷发次数或压力峰值调节。

严禁为了“得到 8 次喷发”任意移动或放大气囊。若使用 6.46 s chronology
反推气囊位置，必须把该算例明确标为 **calibration / chronology-constrained
case**，不能再称为独立 arrival-time validation。必须保留未校准 baseline 和
small/base/large sensitivity 结果。

## 7. 当前真实验证状态

**Phase 2 and eight eruptions have not yet been reproduced.**

当前 VM（分支 tip）进度（勿与下方历史失败证据混淆）：

- 正性缩放强度通量示踪：`boundedPhaseMassTransport`（`∫sigma` 为库存；点态 `s` 可不在 `[0,1]`）。
- initialize 守恒门已过：`∫sigma` 在 0–0.32 s 相对变化为 0；检查点 `0.3289420474`。
- 首次从检查点 resume 曾因 `sigma` `NO_READ` + 未解析单元 `s=0` 重建导致一次性 ~5% 库存跳跃；已修复为 `READ_IF_PRESENT`，并丢弃该错误轨迹后重跑 smoke。
- smoke 正从该检查点跑向 solver `1.25 s`；完成后需再验 `∫sigma`/通量/残差 `<1%`，再 `phase1`→`full`。
- 禁止恢复：clear/clamp、可变密度 MULES on `s`、Sp(ddt)、薄地板 `phiVol`、未缩放 `phi·s`、sigma 裁剪到 `[0,αρ]`。
- 禁止移动/放大气囊。

不要丢失或美化以下失败证据：

- base mesh: `142,343` cells；
- standard `checkMesh`: pass；
- `checkMesh -allGeometry -allTopology`: fail，报告 `2,228` concave cells；
- initialization PT2: `2.853 kPa gauge`，目标 `2.970 kPa`；
- 已完成 paper time `0–1.00 s` smoke；
- 旧 Agent 为交接停止了 phase-1，提交的诊断历史到 paper time `1.504 s`；
- first PT2 peak: `10.818 kPa at 0.392 s`；
- first rim crossing: `0.640 s`；
- total/gas mass residual through `1.504 s`:
  `6.13e-6 / 4.06e-4`；
- operational pocket transfer: `0.620 s`，论文 `6.46 s`，误差约 `-90.4%`；
- upstream gas retained at `1.504 s`: only `8.20%`
  (`1.441 g / 17.557 g`)；
- 12 m/s velocity limiter affected up to `18,217` cells (`12.8%`)；
- phase 1 incomplete；
- phase 2 not run；
- 8 eruptions not reproduced。

当前 close first-pressure peak 不能证明模型正确。最优先问题是：

1. 气囊过早输运/释放；
2. velocity limiter 大范围触发；
3. strict mesh concave-cell failure；
4. phase-1/phase-2 未完成。

## 8. 新账号必须从源码重跑

旧 VM 的 mesh、processor 和时间目录没有提交，不能跨账号恢复 solver
checkpoint。OpenFOAM v2512 预期路径：

`/usr/lib/openfoam/openfoam2512`

从 C9 `openfoam/3d` 目录开始：

```bash
python3 -m pip install -r requirements.txt
python3 prepare_case.py
cd case
./Allrun.mesh
./Allrun.initialize
./Allrun.resume smoke
./Allrun.resume phase1
./Allrun.resume full
./Allrun.postprocess
```

长算例使用持久终端/tmux。启动前检查是否已有同一 solver 进程，避免重复运行。
每个阶段结束后检查 solver log、function-object data、质量守恒、气相守恒、
Courant number、alpha boundedness 和 limiter activation。
当前基线为 `maxCo=0.70`、`maxAlphaCo=0.20`；`maxCo=0.35` 已保留为
有界示踪时间步敏感性，不得通过放宽 alpha-Courant 限制加速界面输运。

时间约定：

- solver `0–0.25 s`: no-ramp initialization；
- solver `0.25 s` = paper `t=0`；
- smoke end: solver `1.25 s` = paper `1.00 s`；
- phase-1 end: solver `6.75 s` = paper `6.50 s`；
- full end: solver `20.25 s` = paper `20.00 s`。

## 9. 分阶段执行要求

### Stage A — source and initialization audit

- 确认所有 paper values、model closures 和 priors 的来源；
- 确认 water/air thermo、EOS、viscosity、surface tension 和 gravity；
- 检查 upstream pocket、riser column、HGL、tailgate、inlet ramp 和 atmosphere；
- 确认纯水 inlet 不会错误注入空气；
- 重跑 no-ramp initialization，量化 PT2/PT3、riser level、gas volume/mass drift。

### Stage B — mesh

- `surfaceCheck` 必须确认 combined STL closed；
- 运行 standard 和 strict `checkMesh`；
- 定位 2,228 concave cells 是否位于 plume、gate、riser 或 pocket；
- 修复或明确证明它们对目标量不敏感；
- 运行 base/refined mesh comparison；
- 报告 cells、max nonorthogonality、max skewness、min volume、concave count。

### Stage C — smoke

- 覆盖完整 ramp 和 first peak；
- 比较 P1m、peak time、first rim crossing；
- 检查 PT1–PT4、riser height、pocket mass/volume、gas conservation；
- 量化 limiter 对多少 cells/faces 生效；
- 至少做一个 limiter/control sensitivity，证明 hard clipping 是否改变核心结果。

### Stage D — complete phase 1

- 至少运行到 paper `6.50 s`；
- 比较 Eq. (7)/(8)、`T = 1.45 s`、第一/第二 geyser chronology；
- 不能用不足两个完整周期的 peak detector 报告 period；
- 检查 cavity formation/disappearance 和 second geyser end；
- 报告 baseline pocket 是否已在 6.46 s 前错误释放。

### Stage E — complete phase 2

- 只有 phase-1 和 pocket transport 数值可信后才推进；
- 运行到 paper `20.00 s`；
- 记录 simulated main-pocket arrival；
- 记录每个 rim-crossing event 的开始、结束、峰值、最大高度和分类；
- 区分 overflow 与 geyser；
- 统计 arrival 后 phase-2 events；
- 比较 8 次实验喷发和 final PT2/PT3/PT4；
- 若仅有 phase 1，状态必须保持 `complete_phase1_only`；
- 若 20 s 跑完但没有 8 次，也必须如实报告实际次数。

### Stage F — sensitivity

至少覆盖可执行的：

- base vs refined mesh；
- default vs tighter timestep/Courant；
- pocket small/base/large；
- MULES vs isoAdvector；
- k-omega SST vs laminar；
- near-adiabatic vs near-isothermal closure sensitivity；
- measured acrylic-pipe wave speed ±20% 与本征水体 2.2 GPa；
- gate area ±20%；
- contact angle 60°/90°/120°；
- interface compression 0.5/1.0/1.5；
- velocity limiter/control sensitivity。

不能只“准备”算例后声称完成 sensitivity；CSV 的 status 必须区分
`prepared`、`mesh_checked`、`initialized`、`smoke_complete`、
`phase1_complete`、`full_complete` 和 `failed`。

## 10. 必须生成/更新的成果

位于 C9 `outputs/`：

- `openfoam_3d_PT1_PT2_PT3_PT4.csv`
- `openfoam_3d_riser_height.csv`
- `openfoam_3d_air_pocket.csv`
- `openfoam_3d_event_table.csv`
- `openfoam_3d_metrics.json`
- `openfoam_3d_pressure_comparison.png`
- `openfoam_3d_riser_comparison.png`
- `openfoam_3d_air_pocket_evolution.png`
- `openfoam_3d_mesh_sensitivity.csv`
- 必要的 Fig. 11 / phase comparison plot

同时更新：

- `openfoam/3d/PAPER_AUDIT.md`
- `openfoam/3d/README.md`
- `openfoam/3d/HANDOFF.md`

`openfoam_3d_metrics.json` 至少保留：

- simulation end/status；
- solver/interface method；
- P1m、peak time、first top、period；
- air-pocket arrival 及检测定义；
- geyser count 和 phase-2 event count；
- final PT2/PT3/PT4 和误差；
- total/gas mass residual；
- initial/end upstream air mass/volume；
- mesh quality；
- max Co、max alpha Co、min dt、max velocity；
- limiter activation；
- paper targets；
- 未报告参数的 provenance status。

## 11. 判断标准

- 质量守恒误差小不代表 pocket transport 正确；
- first pressure peak 接近不代表 phase 2 正确；
- phase-1 match 不得计作 phase-2 reproduction；
- 大范围 limiter clipping 的算例必须标为 numerically qualified；
- strict mesh failure 必须出现在 metrics、README 和最终回复；
- 所有 arrival/event 数值必须附 operational definition；
- 不允许 synthetic data、复制一维结果或手工填写期望事件。

## 12. Git、PR 与最终回复

- 每个逻辑修改单独 commit；
- 测试前先 commit/push 当前实现；
- 测试后若输出或修复改变，再 commit/push；
- 始终更新现有 PR #8；
- 不 force-push，不 amend，不 merge PR；
- 不覆盖或删除失败 baseline 的证据；
- 最终保持 working tree clean。

最终回复必须明确给出：

- branch 和最终 commit SHA；
- PR URL；
- solver、EOS 和 interface method；
- mesh quality 与 strict-check 状态；
- 可复现命令；
- initialization/smoke/phase1/phase2 各自完成状态；
- P1m、时间、first top、period 及误差；
- simulated pocket arrival、定义及误差；
- 实际 geyser count，不得默认写 8；
- final PT2/PT3/PT4；
- total/gas mass conservation；
- limiter activation；
- 已运行的 sensitivity；
- 所有仍未解决的问题。

在真实 20 s 三维数据完成前，必须明确写：

**“Phase 2 and eight eruptions have not yet been reproduced.”**

---
