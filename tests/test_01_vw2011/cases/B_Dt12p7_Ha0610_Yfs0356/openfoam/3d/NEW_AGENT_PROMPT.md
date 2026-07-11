# Complete prompt for the next cloud agent

Copy the full block below into the new Cursor account's Cloud Agent.

```text
你负责继续完成 Geysering Test 1 / Case B 的三维 OpenFOAM 模拟与实验验证。

这是一个真实三维、可压缩气液两相 CFD 任务。接续入口是：

- Repository: https://github.com/brant123451/geysering
- PR: https://github.com/brant123451/geysering/pull/10
- Branch: cursor/test1-caseb-3d-0614
- Latest handoff commit at transfer: 59aa199
- Case root:
  tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356
- CFD root:
  tests/test_01_vw2011/cases/B_Dt12p7_Ha0610_Yfs0356/openfoam/3d

一、不可违反的范围限制

1. 这是 Case B 的全新三维任务，不要继续、合并或借用旧分支
   cursor/test1-caseb-2d-4ac2。旧分支只是未完成的二维调研，不是本任务的
   模型或数值结果。
2. 不得修改 Case A 参考目录：
   tests/test_01_vw2011/cases/A_Dt57p1_Ha0305_Yfs0356/openfoam/3d
3. 所有新增或修改应限制在 Case B 目录，除非修复仓库级基础设施确有必要，
   且必须明确说明。
4. 不能把二维、轴对称、wedge、薄层或单位厚度结果冒充三维结果。
5. 不能伪造网格、求解器日志、实验数据、图、CSV、JSON 或完成状态。
6. 不得用人为压力源、速度源、质量源或针对实验曲线的强制函数制造 geyser。
   气囊膨胀必须来自有限气体库存及可压缩状态方程。
7. 不得声称实验复现完成，除非完整时窗、守恒、hold、base/refined 和验收字段
   都有已提交的运行证据。

二、接手后的第一步

1. 检查当前分支、工作树和 PR，不要重置或覆盖已提交工作。
2. 按顺序阅读：
   - openfoam/3d/HANDOFF.md
   - openfoam/3d/PAPER_AUDIT.md
   - openfoam/3d/README.md
   - config/case.json
   - outputs/openfoam_3d_mesh_sensitivity.csv
   - outputs/openfoam_3d_hold_metrics.json
3. 确认提交 867b2fccd591a9f44325a13c2042bbce32405087 是当前历史祖先。
4. 确认 OpenFOAM.com v2512、Gmsh Python bindings、NumPy 和 Matplotlib 可用。
5. 新 Cloud Agent 的 VM 不包含旧 Agent 的 mesh、processor、time、
   postProcessing、dynamicCode 或日志；这些必须从源文件重新生成。

三、论文与实验权威来源

1. 主要权威来源是 references/vasconcelos2011.pdf，即
   Vasconcelos & Wright (2011)。
2. references/wright2011.pdf 只作为现场尺度背景，不定义本实验 Case B。
3. PAPER_AUDIT.md 已记录确认值、不确定项和冲突。除非找到更直接的原始证据，
   不要静默改写审计结论。
4. Case B 对应 VW Fig. 6 和 Fig. 8 的中间列：
   - tower diameter Dt = 0.0127 m
   - initial air gauge head Ha0 = 0.610 m water
   - initial tower water level Yfs0 = 0.356 m above main-pipe crown
5. Table 2 速度是相同 tower diameter 的汇总平均，不是 Case-B-only 原始测量；
   比较时必须保留此限定。
6. 论文未报告压力传感器的周向/竖向精确位置。当前模型采用
   x = 1.616 m, y = -0.043 m，并在结果中保留最大 D/L = 0.154 的 datum
   不确定性；不能偷偷施加旧的一维 crown correction。
7. 旧 CSV 中 T*=3.846875 的界面点可能被误分类；采用 PAPER_AUDIT.md 中的
   audited range，不要用该单点调参。

四、必须保留的三维几何

1. 主水平管：
   - inside diameter = 0.094 m
   - air chamber length = 0.546 m
   - middle length = 2.970 m
   - downstream length = 0.490 m
   - total length = 4.006 m
2. 竖直 riser/tower：
   - circular inside diameter = 0.0127 m
   - centre x = 3.516 m
   - physical rim = 0.610 m above main-pipe crown
   - model y coordinate of rim = 0.657 m
3. 主水平管与 tower 必须是连通的圆形三维 tee，不得用方管替代。
4. tower 上方必须有真实三维外部空气域，使液柱能越过物理 rim 后继续运动，
   而不是把 rim 直接当出口。
5. 当前外部域：
   - 0.24 m × 0.24 m horizontal footprint
   - 1.20 m above rim
   - 0.40 m below rim outside the tower
6. 由于论文未给 tower wall thickness，外部 casing 假定 2 mm；这是显式不确定
   参数，不得称为实测值。
7. 外部底面、侧面和顶面是 atmospheric/drain boundary，不能设置会积水的
   人工水平托盘。

五、物理模型和初始/边界条件

1. Solver: OpenFOAM.com v2512 compressibleInterFoam 或经充分论证的等价
   可压缩 VOF solver。当前 deck 使用 compressibleInterFoam。
2. 两相：
   - air: perfectGas, molecular weight 28.965 kg/kmol, Cp 1005 J/kg/K,
     mu 1.81e-5 Pa s
   - water: perfectFluid,
     rho = 998.153943 + p/(7504.690432*T) kg/m3
   - baseline temperature = 293.15 K
   - surface tension = 0.072 N/m
3. Gravity = (0 -9.81 0).
4. Baseline air pocket:
   - x = 0 to 0.546 m
   - finite sealed initial gas volume about 0.00378912 m3
   - Ha0 = 0.610 m water gauge
   - initial absolute pressure about 107298.32862 Pa
5. Main pipe downstream of valve initially water-filled.
6. Tower initial free surface:
   - 0.356 m above crown
   - absolute model y = 0.403 m
7. Tower headspace and exterior initially atmospheric air at 101325 Pa.
8. Initial U = 0 and T = 293.15 K.
9. setFields 后必须运行 setAlphaField 生成几何 cut-cell 体积分数，再运行
   setExprFields，保持 p 与 p_rgh 的 hydrostatic consistency；不要恢复阶梯状
   cell-centre 界面或不一致的 uniform pressure 初始化。
10. Atmosphere p_rgh 使用 prghPressure 对应 101325 Pa absolute atmosphere。
11. Walls 为 smooth no-slip、adiabatic。当前 contact angle = 90 degrees，
    因为论文没有提供 acrylic wetting measurement，必须作为假设说明。
12. 当前 baseline 为 laminar single-momentum VOF。若改变 turbulence model，
    必须给出证据并作为敏感性，不得无说明替换 baseline。
13. OpenFOAM 内置的 nAlphaSmoothCurvature 不改变输运的 alpha，仅平滑曲率
    计算。2 次平滑的短诊断在 0.001 s 降低了峰值速度，但随后产生更大热点且
    更慢，因此 baseline 保持 0；必须保留 0/1/2 次迭代敏感性。
14. 已测试并拒绝在 tower 初始自由面增加 conformal internal disk：网格仍为
    单一连通区域且标准 Mesh OK，但 0.001 s 的 |U|max 从 cut-cell 的
    3.207 m/s 增至 8.192 m/s，alpha undershoot 也恶化。stock solver 的曲率来自
    cell-centred alpha，不直接使用 disk 几何法向，因此 baseline 保持
    setAlphaField cut-cell；详见 openfoam_3d_numerical_diagnostics.json。
15. 仅把 curvature normal 的 nHat gradient 改为 leastSquares 也已拒绝：
    |U|max 在 0.001 s 从 3.207 降到 2.018 m/s，但随后升至 0.006 s 的
    4.515 m/s（比 Gauss linear 高 17.7%），alpha undershoot 恶化到 1.6e-7。
    baseline 保持 Gauss linear，不能用首帧改善代替持续稳定性。

六、阀门

1. 阀位于 x = 0.546 m 附近，当前 valveZone 长度 0.012 m。
2. 使用纯耗散 coded fvOption resistance，不得产生压力、速度或质量。
3. baseline opening time = 0.25 s；论文只说明 less than 1 s，因此这是显式假设。
4. fully-open loss coefficient K = 2；closed-state K cap = 1e8，仅作为数值
   impermeability device。
5. 必须通过 1.0 s closed-valve hold 检验泄漏和漂移。
6. CASEB_VALVE_MODE 支持 opening、closed、instant；instant 只是论文模型的
   instantaneous-connection sensitivity。

七、网格要求

1. make_mesh.py 使用 Gmsh OpenCASCADE 和 HXT 生成真实三维网格。
2. 必须局部细化 tower、tee、valve、initial pocket nose、free surface、
   near-rim jet 和 plume corridor。
3. 当前 mesh-size box 使用 40 mm linear transition，并在整个外部 casing 周围
   保留 4 mm corridor，避免 1.05 mm wall triangles 直接连接 25 mm atmosphere
   cells。
4. baseline base preset 当前证据：
   - 1,904,269 tetrahedra
   - tower edge 1.05 mm
   - 12.1 nominal edges across Dt
   - Gmsh 4.12.1, HXT, explicit Gmsh optimizer
5. standard checkMesh 必须打印 Mesh OK.
6. 每次还必须运行 checkMesh -allTopology -allGeometry 并保留严格审计。
7. 当前严格审计的唯一失败是 OpenFOAM v2512 对 sharp-boundary tetrahedra 的
   internal-face determinant 定义：
   - low determinant cells = 448
   - twoInternalFacesCells = 446
   - excess = 2
   - 其他 strict checks 全部通过
8. 该结果只能记录为 accepted_boundary_tet_exception，不能写成 strict Mesh OK.
9. 当前关键质量：
   - max non-orthogonality = 70.2596 degrees
   - severe non-orthogonal faces = 2
   - max skewness = 1.10372
   - min interpolation weight = 0.096809
   - min face-volume ratio = 0.107186
10. 不得通过放宽阈值或删除 strict log 隐藏新问题。若 strict failure 不再仅是上述
    determinant 项，Allrun 必须失败。
11. 必须完成 base/refined grid sensitivity，并保证两者物理和运行控制可比较。

八、运行顺序

严格按以下顺序工作，每阶段失败时先诊断和修复，不能跳过：

1. Static validation:
   python3 validate_case.py
2. Mesh:
   ./Allclean
   CASEB_STAGE=mesh CASEB_MESH=base ./Allrun
3. Full closed-valve hold:
   ./Allclean
   CASEB_STAGE=hold CASEB_MESH=base OPENFOAM_NP=4 ./Allrun
4. Opened-valve smoke:
   ./Allclean
   CASEB_STAGE=smoke CASEB_MESH=base CASEB_END_TIME=0.5 \
     OPENFOAM_NP=4 ./Allrun
5. Full base:
   ./Allclean
   CASEB_STAGE=full CASEB_MESH=base CASEB_END_TIME=10.5 \
     CASEB_VALVE_OPEN_TIME=0.25 CASEB_MAX_CO=0.30 \
     CASEB_MAX_ALPHA_CO=0.20 OPENFOAM_NP=4 ./Allrun
6. Full refined:
   ./Allclean
   CASEB_STAGE=full CASEB_MESH=refined CASEB_END_TIME=10.5 \
     CASEB_VALVE_OPEN_TIME=0.25 CASEB_MAX_CO=0.30 \
     CASEB_MAX_ALPHA_CO=0.20 OPENFOAM_NP=4 ./Allrun
7. Sensitivities.

Allrun.resume 只可用于同一 VM 中仍存在 processor* 状态的中断运行。新账号的新
Cloud Agent 没有旧 VM runtime，必须重新生成。

九、当前已完成证据与性能警告

1. Base mesh 和字段初始化已验证。
2. 四 rank closed-valve startup 已运行并动态编译：
   - phaseAccounting coded function object
   - caseBValveResistance coded fvOption
3. 0.001 s closed-valve run 已成功后处理并提交：
   - all streams reached T* = 0.000578637
   - no overflow
   - no gas entry
   - one-sample liquid/gas/total balance error = 0
   - dimensionless free-surface drift = 0.002459
   - hold passed = false，因为 duration 只有 0.001 s，不是 1.0 s hold
4. 该 0.001 s 运行在四 rank 约需 6.5 分钟，max Co 约 0.3，timestep 最终约
   1.4e-5 s。启动完整时窗前必须检查 startup velocity 和 runtime cost；不得把
   0.001 s 结果外推成完整物理结果。
5. 下一项工作必须是完整 1.0 s closed-valve hold。

十、hold 验收

完整 hold 至少达到 0.9 s，并检查：

1. Hstar peak-to-peak <= 0.02
2. Yfs* maximum drift <= 0.02
3. water above rim < 1e-10 m3
4. no gas-entry event
5. liquid mass error max absolute <= 1%
6. gas mass error max absolute <= 1%

短时 startup 不能代替该验收。

十一、必须完成的敏感性

至少覆盖并记录：

- CASEB_MESH=base|refined
- CASEB_MAX_CO=0.15|0.30
- CASEB_VALVE_OPEN_TIME=0|0.10|0.25|0.50|1.0
- CASEB_C_ALPHA=0.5|1.0|1.5
- CASEB_ALPHA_SMOOTH_CURVATURE=0|1|2
- CASEB_HA0=0.579|0.610|0.641
- CASEB_GAS_EOS=perfectGas|rhoConst

rhoConst 是故意设置的不可压缩气体极限敏感性，不是可信 baseline。Ha0 endpoints
来自论文报告的 ±0.031 m manometer precision。

十二、实验比较目标

使用 PAPER_AUDIT.md 和 digitized data，至少比较：

1. VW Fig. 6 middle panel pressure history：
   - H* = (p_probe - 101325)/(rho_w*g*L)
   - audited plateau target about 0.7575566751
   - plateau window T* = 1 to 4
   - pressure-drop target T* about 4.048
2. VW Fig. 8 middle panel levels：
   - Yfs* plateau target about 0.8225
   - gas-entry range T* = 3.648 to 3.742
   - free-surface top range T* = 3.900 to 3.981
   - audited interface 0.85L range T* = 4.091 to 4.169
3. VW Table 2 diameter-level velocities：
   - Vint* about 1.43
   - Vfs* about 0.44
   - 必须注明不是 Case-B-only raw values
4. Time scale:
   - L/sqrt(g*Dt) = 1.728197891 s
   - full 10.5 s gives T* about 6.08
5. 三方结果必须同图或同表比较：
   - digitized experiment
   - frozen 1-D model
   - 3-D OpenFOAM CFD

十三、geyser 和输出判据

1. 不能仅因 tower 满水就判定 geyser。必须解析 rim 以上的外部液体。
2. 当前判据要求 sustained above-rim water，并且 free surface 在 gas interface
   breakthrough 前到达 top。
3. 报告：
   - pressure series
   - free-surface and gas-interface series
   - local interface/free-surface velocities
   - gas entry、surface top、pressure drop、plateau metrics
   - maximum liquid height
   - water inventory above rim
   - liquid/gas/total mass balance
   - mesh quality and sensitivity
4. 外部 plume 到达 domain top 时必须标记 censored，并扩展 domain 后重跑，
   不能把边界截断高度当物理最大高度。
5. postprocess 输出的 completion_status 和 acceptance fields 是完成声明的门槛。

十四、提交和产物规则

1. 只提交可复现源文件和紧凑证据：
   - CSV
   - JSON
   - comparison plots
   - README/PAPER_AUDIT/HANDOFF updates
2. 不提交：
   - caseB3d.msh
   - constant/polyMesh
   - processor*
   - numeric time directories
   - postProcessing
   - dynamicCode
   - solver logs
   - frame sequences
3. 每个逻辑修改单独 commit，push 到工作分支，并持续更新 PR #10。
4. 保留失败和不确定性说明。Smoke、mesh-only 或 0.001 s startup 不能称为实验
   复现。
5. 如果发现源 deck、物性、边界条件、质量守恒或后处理缺陷，先修复并做最小
   可验证测试，再继续昂贵运行。

十五、最终完成条件

只有以下全部满足才可把任务标记 complete：

1. full closed-valve hold passed
2. base full run all streams reach T* >= 6
3. physically resolved geyser with uncensored plume
4. phase/total balance within documented tolerance
5. compatible refined full run exists
6. grid/timestep/valve/compressibility sensitivities documented
7. Fig. 6、Fig. 8、Table 2 三方定量比较已提交
8. 所有 compact artifacts、limitations 和 recovery instructions 已提交并推送

开始工作时先确认 PR #10 和 HANDOFF.md 的最新状态。不要从头重建已验证的源
deck，也不要删除现有审计；应从完整 1.0 s closed-valve hold 继续。
```

