# 新 Cursor Cloud Agent 完整提示词

> **若你是本地 Agent（只做渲染/接手，不再 Cloud 复算）：**  
> 请改读 Case 根目录 [`../LOCAL_AGENT_HANDOFF.md`](../LOCAL_AGENT_HANDOFF.md)。  
> 本文件保留 Cloud 验收与硬约束；其中「禁止提交 postProcessing」已被本分支为本地渲染而 force-add 覆盖。

把本文件全文作为新 Agent 的任务上下文；任务名称使用：

`geysering test 3 caseA2`

---

继续完成 GitHub PR：

`https://github.com/brant123451/geysering/pull/9`

仓库：

`https://github.com/Brant123451/Geysering`

必须直接继续 PR head 分支：

`cursor/test3-a2-openfoam-1850`

不要从 `main` 重新实现，不要创建互相竞争的第二套实现。开始时先确认 PR
最新 head 包含交接提交 `62beed4`，然后阅读：

* `tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/openfoam/3d/HANDOFF.md`
* `tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/openfoam/3d/PAPER_AUDIT.md`
* `tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/openfoam/3d/README.md`
* PR 全部 diff、提交记录、说明和 CI 状态

指定底座提交为：

`867b2fccd591a9f44325a13c2042bbce32405087`

底座分支为：

`bootstrap/geysering-cases-20260711`

当前 PR 分支已经建立在指定底座上。验证祖先关系即可；禁止 reset、rebase、
force-push 或重新合并覆盖已有工作。

唯一允许修改的目录：

`tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser`

不要修改其他 Case、`paper/`、根 README、`.gitignore`、bootstrap 分支或
`main`。

## 任务目标

根据 Liu、Shao & Zhu (2020) 原论文，对 A2 建立并完成可复现的真实三维
OpenFOAM 模拟，验证实验中“不发生 geyser”的结果。现有 PR 中的实现已修复
大量 pilot 问题，但仍必须用运行证据核查，不能为了得到“不喷发”结论而调参，
也不能把尚未完成的结果报告为完成。

## 已完成并提交的工作

1. 已独立审计原论文、Case 文件、数字化曲线、扫描图、冻结的一维模型和旧
   三维 pilot。
2. 已用 OpenCASCADE/Gmsh 重建完整三维流体域，包括真实圆管、矩形交汇室、
   中心圆形竖管和独立大气开口。
3. 已建立 `base` 与 `refined` 两套确定性网格配置。
4. base 网格为 118,321 个四面体，已通过：

   `checkMesh -allGeometry -allTopology`

   最终输出为 `Mesh OK`。
5. 已修正 `p_rgh`、相分数、入口流量、大气边界、初始液位和下游固定水位
   等边界条件。
6. 对论文没有给出尺寸的下游堰，不得虚构几何；当前模型仅用已报告的
   `hd/Dd=1/4` 建立透明记录的固定水位等效边界。
7. 已完成 4 MPI ranks smoke run：初始 20 L/s 水流量基本守恒，PT3 约
   0.99 kPa，探针及守恒函数对象正常写出，最大 Co 小于 0.5。
8. 已实现 clean-clone 的 `Allrun`、`Allclean`、mesh、solve、resume 和
   后处理脚本。
9. 已实现实验、一维模型、三维 OpenFOAM 的压力、竖管高度、质量守恒和网格
   敏感性后处理。
10. 论文依据、模型假设和限制已经记录在 `PAPER_AUDIT.md`。

旧账号 VM 中可能仍有一个 base full run，但 Agent VM、进程、网格、日志、
`processor*` 和时间步目录都不会跨账号转移。新 Agent 不得假设可以恢复该
未提交运行；如果 Git clone 中没有完整 checkpoint，必须从已提交源文件重新
运行。

## 论文审计必须保持覆盖的参数

逐项用论文页码、图或表核对，并在发现现有记录错误时如实更正：

1. 上游管：长度 5.80 m、直径 0.20 m、坡度 1:100。
2. 交汇室：0.30 × 0.30 × 0.45 m；上下游管底高差 0.18 m。
3. 下游管：长度 5.95 m、直径 0.28 m、水平布置。
4. 竖管：直径 0.06 m、长度 1.22 m、顶部与大气连通。
5. 流量：`Q0=20 L/s`、`Q1=100 L/s`、阀门开启约 0.4 s。
6. 下游：明渠流、初始水深 `hd/Dd=1/4`、堰控制。
7. PT1：竖管壁，交汇室顶盖以上 0.80 m。
8. PT2：交汇室顶盖。
9. PT3：交汇室前壁、底部以上 0.02 m。
10. 实验结果：不发生 geyser；涌波约 1.20 s 到达交汇室；PT2 稳态约
    2.15 kPa；PT3 初始约 0.99 kPa、稳态约 4.99 kPa；首次竖管混合柱
    高度约 0.13 m。

若论文与 Case README 不一致，以论文为准并记录差异。禁止猜测缺失尺寸或
数据。0.13 m 是从论文图中数字化的首次混合柱标量，不是完整高度时程。

## 三维物理模型要求

1. 保持真实三维圆管面积、矩形交汇室、中心竖管及上下游管空间关系；不得
   将二维或薄层结果作为最终结果。
2. 核查入口流量换算、出口明渠/堰等效条件、初始水位、重力方向及所有大气
   边界。
3. 核查 OpenFOAM v2512 中 `interFoam` 与可压缩替代求解器的适用性。当前
   记录的结论是：通气、充水涌波和 no-geyser 分支可用 `interFoam` 比较平均
   压力与液面；它不能表达声学水锤、封闭气囊压缩或亚网格含气混合物。若运行
   证据推翻这一判断，应更正模型和文档，而非隐瞒。
4. 压力与实验比较必须使用重构的大气表压 `p`，不能直接把 `p_rgh` 当作
   传感器读数。
5. 模拟统一定义 `t=0` 为流量爬升开始。论文定义的 `t=0` 是阀门完全打开，
   因而比较时使用：

   `t_sim = t_paper + 0.4 s`

## 网格与数值验证

依次完成并保留可审计的紧凑指标：

1. 从干净状态生成 base 网格。
2. 运行完整：

   `checkMesh -allGeometry -allTopology`

3. 确认无负体积、泄漏边界、严重拓扑错误或错误 patch。
4. 完成 smoke run，确认流向、液位、下游明渠、竖管顶部大气边界和
   PT1/PT2/PT3 探针。
5. 完成 base 全瞬态：爬升前有稳定初始化，爬升后至少覆盖 14 s。
6. 完成 refined 网格的完整 `checkMesh`、smoke/full run。
7. 界面 Courant 数目标不超过 0.5；报告实际最大值、最小时间步和是否存在
   异常限步区域。
8. 比较两套网格的压力、涌波时间、竖管高度、是否喷发和守恒误差。
9. 报告网格数、求解器版本、MPI ranks、ClockTime、最小时间步、最大 Co 和
   液体质量守恒误差。
10. 运行若中断，先判断 checkpoint 是否完整，再使用现有
    `Allrun.resume`；不要伪造结束标记。

推荐在 Case 中执行：

```bash
cd tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/openfoam/3d/case
./Allrun base
./Allrun refined
```

已验证环境需要 OpenFOAM.com v2512、Gmsh Python API 4.15.2、
`libGLU.so.1`、NumPy、Matplotlib 和 4 个 MPI ranks。若环境不同，先核查
版本和命令，不得静默改变数值设置。

## 实验和一维模型对比

使用已有数字化数据：

* `data/digitized/fig3_PT1.csv`
* `data/digitized/fig3_PT2.csv`
* `data/digitized/fig3_PT3.csv`

分别标识实验、已有一维模型和新三维 OpenFOAM，提取并比较：

* PT1、PT2、PT3 压力时程；
* 涌波到达时间；
* PT2/PT3 初值和稳态值；
* 竖管水/气混合界面高度；
* 最大竖管柱高；
* 是否达到 1.22 m 管顶；
* 是否有竖管顶部液体/混合物排出；
* 液体质量守恒；
* base/refined 网格敏感性。

竖管高度必须区分代码中已实现的 water-equivalent、contiguous 和
mixture-front 定义，不能把单点壁面液膜误判成充满截面的混合柱。

## 必须形成的紧凑成果

仅在当前 Case 内提交：

* `openfoam/3d/PAPER_AUDIT.md`
* `openfoam/3d/README.md`
* 完整源算例
* 可复现的 Allrun/Allclean/恢复脚本
* `outputs/openfoam_3d_pressure_series.csv`
* `outputs/openfoam_3d_riser_series.csv`
* `outputs/openfoam_3d_metrics.json`
* `outputs/openfoam_3d_pressure_comparison.png`
* `outputs/openfoam_3d_riser_comparison.png`
* Case 根 `README.md` 的最终状态说明

禁止提交：

* `processor*`
* `postProcessing/`
* `constant/polyMesh/`
* 数值时间步目录
* 大型网格
* 日志
* 帧序列
* Python 或工具缓存

## 完成标准

只有同时满足下列条件才能宣称完成：

1. 几何已与论文逐项核对。
2. base/refined 的完整 `checkMesh` 通过。
3. 两套真实三维完整实验窗口运行完成。
4. A2 对 geyser/no-geyser 的预测有运行证据。
5. 已对 PT1/PT2/PT3、涌波时刻及竖管高度定量比较。
6. 已报告质量守恒和网格敏感性。
7. 已如实报告误差、模型不足和求解器限制。
8. 所有源文件和紧凑结果已提交并推送到同一 PR 分支。
9. `git status` 中不得残留应提交源文件，也不得误提交大型运行产物。

提交前检查 Case 范围 diff、输出文件可重建性和文档数据一致性。每个逻辑修改
使用清晰 commit，正常 push；禁止 force-push。更新 PR 说明中的最终运行状态。

最终回复必须给出：

1. 分支名；
2. 最终 commit SHA；
3. OpenFOAM 求解器和版本；
4. base/refined 网格规模；
5. 完整运行命令；
6. 主要实验—模拟误差；
7. 是否发生 geyser；
8. 尚未解决的物理或数值问题。
