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
4. 确认 OpenFOAM.com v2512、Gmsh Python bindings、NumPy、Matplotlib，
   以及 `./build_twophaseflow.sh` 固定版本编译产物可用。
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

1. Solver: OpenFOAM.com v2512 加固定提交
   `de9826f9ffb24f4b635ac97fd388ebd560cfc174` 的 DLR-RY TwoPhaseFlow
   `compressibleInterFlow`。source default 使用
   `isoAdvection + plicRDF + RDF`，并按库内静态表面张力 benchmark 设置
   `interpolateNormal=false`；它仍是单动量、可压缩、非等温 VOF。当前 hold
   筛选正在评估 `fitParaboloid`，但 true/false 两个 0.006 s 候选都尚未满足
   联合速度/Courant 门槛。完整 hold 尚未完成，不能仅因短筛选退出正常就称为
   baseline 已验证。
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
    更慢，因此 stock baseline 保持 0；这些历史诊断必须保留，但该开关不适用
    于下述 TwoPhaseFlow RDF 候选。
14. 已测试并拒绝在 tower 初始自由面增加 conformal internal disk：网格仍为
    单一连通区域且标准 Mesh OK，但 0.001 s 的 |U|max 从 cut-cell 的
    3.207 m/s 增至 8.192 m/s，alpha undershoot 也恶化。stock solver 的曲率来自
    cell-centred alpha，不直接使用 disk 几何法向，因此 baseline 保持
    setAlphaField cut-cell；详见 openfoam_3d_numerical_diagnostics.json。
15. 仅把 curvature normal 的 nHat gradient 改为 leastSquares 也已拒绝：
    |U|max 在 0.001 s 从 3.207 降到 2.018 m/s，但随后升至 0.006 s 的
    4.515 m/s（比 Gauss linear 高 17.7%），alpha undershoot 恶化到 1.6e-7。
    baseline 保持 Gauss linear，不能用首帧改善代替持续稳定性。
16. 上述两项是 stock `compressibleInterFoam` 诊断。由于 stock CSF 在
    0.006 s 仍有 3.839 m/s 自由面伪流，当前 source deck 已迁移到固定版本
    TwoPhaseFlow。`CASEB_ALPHA_SMOOTH_CURVATURE` 对 RDF 无意义并被拒绝；
    新的曲率敏感性为 `RDF|fitParaboloid|gradAlpha`，界面压缩敏感性必须用
    `CASEB_ADVECTION_SCHEME=MULESScheme` 配合 `CASEB_C_ALPHA`，不能把
    isoAdvection 下无作用的 cAlpha 当成有效敏感性。
17. 首次完整 hold 尝试暴露了从 stock deck 沿用的过度校正：每步约有
    2 outer × 3 pressure × 3 non-orthogonal = 18 次压力求解。当前候选改用
    TwoPhaseFlow 自带表面张力算例常用的 1 次 alpha correction、2 次 alpha
    subcycle、1 outer、2 pressure corrector、0 non-orthogonal corrector。
    这些值已写入 manifest，并通过 0.006 s 复筛：wall time 587 s，峰值速度
    1.682 m/s，末帧 1.285 m/s，alpha 和质量守恒仍稳定。但随后完整 hold
    尝试在 0.06 s 已不可通过：Hstar peak-to-peak=0.0570，且阀区上游相邻
    单元速度持续增长。短筛选不能替代超过该延迟热点的筛选。
18. conformal baffle 消除了上述阀区热点，但 `plicRDF + RDF` 的扩展筛选仍在
    0.04 s 被拒绝：速度热点留在 tower 初始自由面，末帧为 1.362 m/s；
    transducer 范围已达 Hstar peak-to-peak=0.05658。alpha、质量、rim 水量和
    gas-entry 仍干净。因此闭阀拓扑保留，但 RDF 不再是已通过 hold 候选；
    下一项数值筛选是库支持的 `fitParaboloid` 曲率。

六、阀门

1. 阀位于 x = 0.546 m 附近，当前 valveZone 长度 0.012 m。
2. opening/instant 模式使用纯耗散 coded fvOption resistance，不得产生压力、
   速度或质量。connected-domain penalty 的 closed 模式已被 0.06 s 诊断拒绝；
   closed 模式在同一阀面使用两侧 no-slip conformal baffle 来支撑真实闭阀
   压差；它已验证无阀区热点或泄漏，但仍需配合能通过自由面压力漂移筛选的
   曲率模型。opening/instant 仍使用耗散 resistance。
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
4. baseline base preset 当前 conformal-valve mesh 证据：
   - 1,903,549 tetrahedra
   - tower edge 1.05 mm
   - 12.1 nominal edges across Dt
   - Gmsh 4.12.1, HXT, explicit Gmsh optimizer
5. standard checkMesh 必须打印 Mesh OK.
6. 每次还必须运行 checkMesh -allTopology -allGeometry 并保留严格审计。
7. 当前严格审计的唯一失败是 OpenFOAM v2512 对 sharp-boundary tetrahedra 的
   internal-face determinant 定义：
   - low determinant cells = 646
   - twoInternalFacesCells = 644
   - excess = 2
   - 其他 strict checks 全部通过
8. 该结果只能记录为 accepted_boundary_tet_exception，不能写成 strict Mesh OK.
9. 当前关键质量：
   - max non-orthogonality = 74.2375 degrees
   - severe non-orthogonal faces = 1
   - max skewness = 1.05840
   - min interpolation weight = 0.090922
   - min face-volume ratio = 0.100015
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
5. 固定版本 RDF 候选的干净 0.006 s 筛选已完成：
   - solver exit code = 0，四 rank wall time = 1118 s；
   - 屏幕内 `|U|max` 峰值 1.371 m/s，0.006 s 为 1.077 m/s；stock 同时刻为
     3.839 m/s；
   - alpha 范围约
     \([-2.66\times10^{-9},1+3.40\times10^{-9}]\)；
   - gas/total balance 最大误差分别约
     \(3.74\times10^{-6}\%\) 和 \(1.53\times10^{-8}\%\)；
   - 无 rim 以上水量或 gas-entry。
   第一次尝试在 `End` 后因 stock `libgeometricVoF` 与 TwoPhaseFlow `libVoF`
   重复加载而析构失败；commit `9a59974` 移除了冲突依赖，以上数据来自随后
   exit 0 的干净复测。
6. TwoPhaseFlow-reference corrector 配置的 0.006 s 复筛也已 exit 0：
   - wall time = 587 s，比旧 corrector RDF 筛选缩短 47.5%；
   - `|U|max` 峰值 1.682 m/s，0.006 s 为 1.285 m/s，仍比 stock 同时刻低
     66.5%；
   - alpha 范围约
     \([-5.77\times10^{-9},1+3.52\times10^{-12}]\)；
   - gas/total balance 最大误差分别约
     \(3.16\times10^{-6}\%\) 和 \(1.80\times10^{-8}\%\)；
   - 无 rim 以上水量或 gas-entry。
7. connected-domain penalty 完整 hold 尝试已在 0.06 s 主动停止：
   - Hstar peak-to-peak = 0.0570，已超过 0.02，后续运行无法消除该超限；
   - 同一阀区上游单元 `|U|max` 从 0.04 s 的 1.414 m/s 增至
     0.06 s 的 1.825 m/s；
   - global Co 而非 interface/capillary Co 把 timestep 降至
     \(2.79\times10^{-5}\) s；
   - alpha、相/总质量守恒、rim 水量和 gas-entry 仍干净。
   该结果证明问题是闭阀压差支撑和 sharp porous edge，不是 RDF 输运发散。
8. conformal two-sided no-slip baffle 的生成网格、双侧 patch 和 0.001 s
   startup 已验证；其 RDF 扩展筛选在 0.04 s 主动停止：
   - Hstar peak-to-peak = 0.05658，已不可逆超过 0.02；
   - `|U|max` 始终位于 tower 初始自由面，0.01 s 和 0.04 s 分别为
     1.757 m/s 和 1.362 m/s，没有旧 penalty 的阀区热点；
   - alpha 范围约
     \([-1.75\times10^{-10},1+2.44\times10^{-10}]\)；
   - gas/total balance 误差低于 \(6.5\times10^{-6}\%\)，无 rim 水量和
     gas-entry。
   这证明 baffle 拓扑正确，但剩余缺陷是自由面曲率/压力平衡。
9. `CASEB_CURVATURE_MODEL=fitParaboloid` 的首个 0.006 s 短筛选已 exit 0：
   - 写出时刻峰值和末帧速度为 1.187 和 0.796 m/s，比 reference-corrector
     RDF 分别低 29.4% 和 38.1%；
   - alpha 和相/总质量守恒仍干净，无 rim 水量或 gas-entry；
   - 但 startup 单步 global Co 达 1.957、interface Co 达 0.676，超过声明限制。
   该结果不能延长。
10. source deck 物化并记录 `plicRDF` 的 `interpolateNormal=false` 后，
    `fitParaboloid` 0.006 s 复筛也已 exit 0：
    - global/interface Co 降至 0.350/0.289，严重单步尖峰已消除；
    - 峰值速度却上升 9.7% 至 1.302 m/s，末帧仅微降至 0.788 m/s；
    - alpha 约在
      \([-1.23\times10^{-11},1+5.97\times10^{-12}]\)，gas/total balance
      误差低于 \(3.4\times10^{-6}\%\)，仍无 rim 水量或 gas-entry；
    - 只有一个 pressure sample，Hstar=0 没有稳定性意义。
    因此它不满足“CFL 和峰值速度都改善”的门槛，也不能延长。下一步必须用
    硬 `maxDeltaT` 的配对 startup 诊断分离 normal interpolation 与一步滞后的
    Courant 控制；只有早期配对同时通过声明的 Courant 和速度门槛，才复筛至
    0.006 s 并延长到超过 0.04 s。只有 Hstar peak-to-peak 不超过 0.02 的候选
    才能开始完整 1.0 s hold。
11. 上述 `interpolateNormal=true` 硬 timestep 诊断已主动停止：
    - 即使 `maxCo=0.2`、`maxDeltaT=1e-5`，global Co 仍在约
      \(2.0\times10^{-5}\) s 重现 1.992；
    - 尖峰前实际 timestep 只有 \(4.30\times10^{-6}\) s，interface Co 只有
      0.0075，因此根因不是后期 timestep 放大，而是 near-interface mask 外的
      高度局部 pressure-corrected flux；
    - 首两个写出速度已为 1.239/1.207 m/s，也未保留旧运行的低峰值。
    不得继续缩小 `maxDeltaT` 挽救 true 路径。source 已物化 plicRDF
    iteration/tolerance 控制；下一步保留 `interpolateNormal=false`，筛选静态
    benchmark 的 `iterations=10`、`tol=1e-8`，先覆盖旧 false 候选约
    0.003 s 的速度峰值。
12. 上述严格 plicRDF 收敛筛选也已主动停止：
    - 0.00047 s 首个写出速度为 1.905 m/s，比默认 false 运行的全窗峰值高
      46.3%；
    - global/interface Co 均达到 0.487；
    - alpha 仍在 roundoff 范围内，无 rim 水量或 gas-entry。
    因峰值门槛已不可逆失败，运行在 0.0018 s 停止。不得继续增加 plicRDF
    iterations；下一项是 `CASEB_RECONSTRUCTION_SCHEME=isoAlpha` 配合
    `fitParaboloid`，先筛到超过旧 0.003 s 速度峰窗口。
13. `isoAlpha + fitParaboloid` 也已在约 0.001 s 主动停止：
    - early pure-cell global Co=1.206；
    - 后续 global/interface Co 均达到 0.705；
    - 写出速度从 0.00047 s 的 1.233 增至 0.00101 s 的 1.667 m/s；
    - alpha 仍 bounded，无 rim 水量或 gas-entry。
    该配对还失去 fitParaboloid 依赖的 plicRDF wall ghost contact-angle
    geometry，因此不再延长。
14. 保留真实 sigma 的 `constantCurvature=0` 机制诊断已正常运行至
    0.006 s：
    - global/interface Co 分别达到 0.393/0.214，仍超过 0.30/0.20 门槛；
    - 写出峰值和末帧速度为 1.437/0.770 m/s，热点仍在 tower 初始自由面；
    - alpha 约在
      \([-2.51\times10^{-11},1+5.18\times10^{-11}]\)，gas/total balance
      误差低于 \(3.9\times10^{-6}\%\)，无 rim 水量或 gas-entry；
    - 只有一个 pressure sample，Hstar=0 没有稳定性意义。
    移除 variable-curvature force 并未消除 startup imbalance，因此该非物理
    机制诊断不能延长或作为 full-run 模型。下一步筛缺失的
    `RDF + plicRDF + interpolateNormal=false` 物理组合；只有同时满足
    Courant 和速度门槛才可超过旧 0.04 s pressure-drift 窗口，且只有
    Hstar peak-to-peak 不超过 0.02 才能开始完整 1.0 s hold。
15. 上述 `RDF + plicRDF + interpolateNormal=false` 物理组合也已正常运行
    至 0.006 s，但不能延长：
    - global/interface Co 分别达到 0.365/0.306；
    - 0.006 s 写出速度为 1.220 m/s，热点仍在 tower 初始自由面；
    - alpha 仍 bounded，无 rim 水量或 gas-entry；
    - 两次超限后的 timestep 缩减精确符合 `limit/observed` 比例，确认是一步
      滞后的 Courant controller，而不是 pressure solver 发散。
    更重要的是，pre-solver 直接检查
    `p_rgh - (p + rho*9.81*y)` 得到 -453.5 至 +3946.5 Pa，最大值就在初始
    自由面。源码审计确认单次 `setExprFields` 预加载旧 `p` 后，各个新 `p`
    通过未注册临时对象写盘，最终 `p_rgh` 表达式仍读取缓存旧值。此前全部
    startup 筛选均受此缺陷影响。下一步必须拆分 absolute `p` 与 `p_rgh`
    为两个独立 `setExprFields` 进程，先证明 residual 接近 roundoff，再复筛
    `constantCurvature=0` 和物理 RDF；不得直接开始完整 hold。
16. commit `3a1777d` 已完成并验证上述压力初始化拆分：
    - 全部 1,903,549 个 cell 的
      `p_rgh - (p + rho*9.81*y)` 从 [-453.5,+3946.5] Pa 降至 [0,0] Pa；
    - 17 项 source/workflow tests 全部通过；
    - 修复后的 `constantCurvature=0` 已正常运行至 0.006 s；
    - 首个写出速度降至 0.00047 s 的 0.170 m/s，但延迟自由面热点仍在
      0.00454 s 达到 1.424 m/s；
    - global/interface Co 分别为 0.363/0.186，alpha 和质量界仍干净。
    因此旧 `p` 缓存确实制造了 startup impulse，但不是延迟热点的唯一来源。
    该非物理模型仍不能延长。下一步用修复后的初始化复筛
    `RDF + plicRDF + interpolateNormal=false`，再决定 timestep 或
    pressure-coupling sensitivity。
17. 修复初始化后的物理 RDF 复筛已正常运行至 0.006 s，但明确失败：
    - global/interface Co 为 0.366/0.303；
    - 首帧速度 1.516 m/s，写出峰值在 0.00349 s 达 1.775 m/s，仍位于
      初始自由面；
    - 末帧速度仍为 1.344 m/s；
    - 分析曲率为零的平面界面上，写出的 RDF `K_` 达数千 1/m，峰值时一个
      自由面 cell 为 +1608.6 1/m；
    - alpha 和质量界仍干净。
    RDF 相对 zero-curvature 的首帧差异确认 variable-curvature force 很强，
    而 zero-curvature 的延迟峰又独立确认 pressure-gravity 底噪。全局 K
    极值可能落在 alpha-CSF 不活跃区域，所以下一步不是直接再换模型，而是
    物化并记录实际 Umax cell 的 alpha/rho/K，以及 pressure-gravity、
    surface-tension 和 total face-force residual。根据同位机制证据只选择
    一个配对 sensitivity；不得直接延长到 0.04 s 或 1.0 s。
18. commit `771714e` 的同位诊断已通过 17 项测试、动态编译并执行。在
    0.006493 s 的最小 continuation 中：
    - Umax cell 为近纯气体：
      alpha.water=\(1.03\times10^{-5}\)、rho=1.215 kg/m3、
      K=337.5 1/m；
    - pressure-gravity residual 与 surface-tension force 的最大值为
      462/473 kPa/m，并位于同一个 tower 深部 wall-adjacent face
      (y=0.184 m)；
    - 最大 total residual 为 77.6 kPa/m，位于预期自由面附近；
    - 深部力热点旁的 alpha probe 从 time-zero 的 1.0 变成 0.006493 s
      的 0.437，距物理自由面 0.219 m，不可能是该时窗内的物理 advection。
    这确认 exact-radius tower cylinder 初始化/重构产生了非物理壁邻相界面。
    下一步把 tower-water initializer 扩入已知 2 mm 固体壁间隙（不得触及
    exterior fluid cells），先证明 y<0.403 的 tower fluid 全湿，再复筛
    zero-curvature 与 RDF。
19. commit `ebf034a` 已把 tower-water selector 半径改为 0.00735 m：
    比 0.00635 m fluid radius 大 1 mm，同时比 0.00835 m exterior-fluid
    内边界小 1 mm。18 项测试通过。修复后的 zero-curvature 复筛表明：
    - deep wall-adjacent interface/force hotspot 已消失；
    - 写出峰值/末值速度从 1.424/0.760 降到 1.031/0.687 m/s；
    - pressure-gravity residual 从 deep layer 的 462 kPa/m 降至自由面
      10.5 kPa/m；
    - interface Co 仅 0.121，但 deltaT 增长至 0.179 ms 后，一步滞后的
      global Co 达 0.419；
    - alpha 与质量界干净，Umax cell 是物理自由面上的近纯气体。
    下一步用 fully-wet initializer 复筛 physical RDF；若 surface force
    主导，则用 hard maxDeltaT 配对 `curvFromTr=false`，不得直接延长。
20. fully-wet physical RDF 已正常运行至 0.006 s，但仍未通过：
    - deep y=0.184 m interface/force hotspot 没有复现，全部热点都在
      y=0.403 m 物理自由面；
    - global/interface Co 最大值为 0.349/0.206，均超过 0.30/0.20；
    - 首帧、峰值和末帧速度为 1.338、1.805、1.056 m/s；
    - collocated pressure-gravity/surface-tension 最大值为 82/118 kPa/m，
      且位于同一 free-surface face；
    - alpha、质量界、rim water 和 gas-entry 仍干净。
    表面张力离散项是剩余 startup 的较强贡献。下一步必须在相同硬
    `maxDeltaT` 下配对 `curvFromTr=true|false`，只改变曲率离散；不得把
    当前结果延长为 0.04 s 或 1.0 s hold。
21. 硬 timestep 配对的 `curvFromTr=true` reference 已运行至 0.0035 s：
    - `maxDeltaT=1e-5` 下 global/interface Co 为 0.115/0.021；
    - 写出峰值/末值速度为 1.317/0.949 m/s；
    - surface-force 最大值仍在真实自由面，为 119 kPa/m；
    - 后期在 y=0.463 m exterior gas 出现 132 kPa/m pressure-gravity
      residual，且短暂成为 Umax 区域；
    - alpha 与质量界仍干净。
    这只证明硬 timestep 本身显著影响峰值，不能据此选择 curvature formula。
    下一步必须用完全相同控制运行 `curvFromTr=false`，完成配对后再决策。
22. 完全同控制的 `curvFromTr=false` 已完成，且被拒绝：
    - global/interface Co 为 0.125/0.024，仍在门槛内；
    - 首帧速度从 true 的 1.317 降至 1.121 m/s，但 0.0035 s 速度反而增长到
      1.549 m/s；true 同时刻仅 0.949 m/s；
    - y=0.463 m exterior-gas pressure residual 从 true 的 132 增至
      198 kPa/m；
    - 最大 surface force 小幅从 119 降至 113 kPa/m，不能抵消后期恶化。
    因此保留 source-default `curvFromTr=true`。先以同一 `maxDeltaT=1e-5`
    复跑至 0.006 s，确认 exterior-gas 热点有界；不得直接跨到 0.04 s。
23. 保留的 trace formula 已在同一硬 timestep 下复跑至 0.006 s，但热点
    没有保持有界：
    - global/interface Co 为 0.152/0.024，仍通过门槛；
    - 末帧速度是全程最大值 1.481 m/s，位于 y=0.463 m pure exterior gas；
    - 同位 curvature=0，surface force 不在该处；
    - pressure-gravity/total residual 在同一位置达到 208 kPa/m，已超过
      free-surface surface-force 最大值 119 kPa/m；
    - alpha、质量界、rim water 和 gas-entry 仍干净。
    该候选不得延长到 0.04 s。下一项是用 solver 相同 gravity-force operator
    构造离散 hydrostatic `p_rgh`，先验证 pre-solver face residual，再复筛
    source-default RDF。
24. 离散 hydrostatic initializer 已完成动态验证；补齐模型库、修正
    `fixedFluxPressure` 调用顺序并注册 `hRef` 后，20 项测试通过：
    - 10 次 corrector 把 pre-solver 最大 face residual 从 1.661 MPa/m
      降到 1.018 kPa/m，代数 `p/p_rgh` residual 为
      \(1.46\times10^{-11}\) Pa；但 face residual 尚非 roundoff；
    - adaptive RDF 复筛中，所有速度/力热点均在 y=0.403 m 物理自由面，
      y=0.463 m exterior-gas 热点未出现；
    - global/interface Co 为 0.365/0.204，仍双双超限；
    - 首帧、峰值、末帧速度为 1.279、1.798、1.056 m/s，与 analytic
      fully-wet RDF 的 1.338、1.805、1.056 m/s 实质相同；
    - dynamic pressure-gravity/surface force 最大值为 83/118 kPa/m；
    - alpha、质量界、rim water 和 gas-entry 干净。
    该 adaptive 候选被拒绝。下一项只改变 `maxDeltaT=1e-5`，与既有
    analytic hard-cap trace 运行直接配对，确认 discrete projection 是否
    消除其增长的 exterior-gas 热点；不得直接延长到 0.04 s。

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
- CASEB_MAX_CAPILLARY_NUM=0.5|1.0
- CASEB_VALVE_OPEN_TIME=0|0.10|0.25|0.50|1.0
- CASEB_ADVECTION_SCHEME=isoAdvection|MULESScheme
- CASEB_RECONSTRUCTION_SCHEME=plicRDF|isoAlpha|gradAlpha
- CASEB_RECONSTRUCTION_ITERATIONS=5|10
- CASEB_RECONSTRUCTION_TOL=1e-6|1e-8
- CASEB_INTERPOLATE_NORMAL=false|true
- CASEB_CURVATURE_MODEL=RDF|fitParaboloid|gradAlpha
- CASEB_C_ALPHA=0.5|1.0|1.5（仅与 MULESScheme 配合）
- CASEB_N_ALPHA_CORR=1
- CASEB_N_ALPHA_SUBCYCLES=2
- CASEB_N_OUTER_CORRECTORS=1
- CASEB_N_CORRECTORS=2
- CASEB_N_NON_ORTHOGONAL_CORRECTORS=0
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

