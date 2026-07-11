# Cursor Cloud Agent handoff

Use the following prompt when starting the replacement Cloud Agent under the
new Cursor account.

```text
任务名：geysering test 1case A 3D

你正在接管一个已经完成主体开发和验证的 OpenFOAM Case。先检查现状，
再根据用户的新请求继续；不要从头重建，也不要自行重复完整 9 s 模拟。

仓库：
https://github.com/Brant123451/Geysering

基础分支：
main

继续使用的现有分支：
cursor/organize-case-a-1dfe

现有 PR：
https://github.com/Brant123451/Geysering/pull/3

唯一允许修改的目录：
Vasconcelos_Wright_2011_Geysering/caseA_Dt57p1_Ha0305_Yfs0356

禁止修改：
- 其他 Case
- paper
- 根 README
- 根 .gitignore

开始工作前：
1. 确认当前分支是 cursor/organize-case-a-1dfe，并与 origin 同步。
2. 检查 git status，不覆盖未提交的人类修改。
3. 阅读本 Case 的 README.md、HANDOFF.md、manifest.yaml、config/case.yaml、
   reference/README.md 和 outputs/README.md。
4. 阅读 PR #3 的标题和描述。
5. 检查当前云环境是否安装 OpenFOAM v2512；不要假设新账号的环境、
   密钥、MCP 授权或硬件配置会从旧账号转移。

已经完成的工作：
- 建立了可独立复跑的 Case 目录：config、data、model、scripts、
  reference、outputs。
- 保留并验证了原有二维平面模型。
- 新增真实圆形水平管和圆形竖井的三维模型
  model/openfoam_3d_caseA。
- 三维几何由闭合的布尔并集 STL 表示；OpenFOAM surfaceCheck 确认其
  闭合、单连通且无自相交。
- 使用 blockMesh + snappyHexMesh 生成 138292 单元网格；标准
  checkMesh 报告 Mesh OK。
- 论文 Case A 的已公开几何和初始参数已对齐：
  * 水平管直径 0.094 m
  * 上游气室长度 0.546 m
  * 中间充水段长度 2.970 m
  * 下游封闭段长度 0.490 m
  * 竖井直径 0.0571 m
  * 竖井高出管冠 0.610 m
  * 初始竖井水位高出管冠 0.356 m
  * 初始气压水头 0.305 m
  * 大气压 101325 Pa
  * 气室绝对压力 104311.7 Pa
  * 水平管中心线初始水压 105271.3 Pa
- 物理气室目标体积为 0.00378912 m3；离散网格体积为
  0.00376901825 m3，误差 0.53%。
- 已使用 OpenFOAM v2512 compressibleInterFoam 和 4 路 MPI 完成
  0--9.0 s 三维计算，日志正常结束于 Finalising parallel run。
- 紧凑结果、曲线和指标已提交到 outputs/。

三维结果与论文试验的对比：
- 平均压力平台 H*=0.698；试验目标 0.54，仍高约 29%。
- 压力 RMSE=0.148；二维结果为 0.206。
- 最大自由液面 Y*=0.651；试验目标约 0.63。
- 自由液面 RMSE=0.0319；二维结果为 0.0122。
- 界面开始抬升 T*=7.448；试验重复值为 7.3、7.8、7.9。
- 界面追上自由液面 T*=8.337；试验约 8.4。
- 界面 RMSE=0.207；二维结果为 0.347。
- 模拟和试验均未发生 geysering。
- 结论必须表述为“部分匹配”：三维明显改善压力 RMSE 和界面时序，
  但压力幅值仍未定量匹配，自由液面 RMSE 反而比二维大。

明确限制：
- 模型使用瞬时开阀近似；试验为人工在 1 s 内开启。
- 当前三维核心网格尺度约 6 mm。
- 当前模型采用层流设置。
- 壁面润湿和接触角资料不足，未做试验标定。
- 不得声称模型与试验完全一致。

主要命令（从 Case 根目录执行）：
- 三维快速验证：./scripts/validate_3d.sh
- 三维完整计算：OPENFOAM_NP=4 ./scripts/run_3d.sh
- 继续中断计算：./scripts/resume_3d.sh
- 清理三维生成物：./scripts/clean_3d.sh
- 三维后处理：python3 scripts/postprocess_compare.py --model 3d
- 二维快速验证：./scripts/validate.sh
- 二维完整计算：OPENFOAM_NP=4 ./scripts/run.sh

除非用户明确要求，不要重新运行完整 9 s 模拟。重新运行会生成约
133 MB 的临时 OpenFOAM 数据，并且在旧的 4 核 CPU 云环境中记录到
约 6209 s 的 OpenFOAM ExecutionTime。

不得提交：
- OpenFOAM 时间目录
- constant/polyMesh
- processor*
- postProcessing 原始探针
- 帧序列
- 大日志

可以提交：
- 必要的 OpenFOAM 字典和初始场
- 脚本与文档
- 紧凑 CSV、JSON、PNG、PDF 对比结果

如需修改：
1. 只修改指定 Case。
2. 按风险执行验证。
3. 每个逻辑改动单独 commit。
4. 使用 git push -u origin cursor/organize-case-a-1dfe 推送。
5. 更新现有 PR #3，不要合并 main，不要 force push。
6. 最终报告分支、commit、测试结果、未提交的大文件和恢复命令。
```

Conversation history, VM state, personal secrets, MCP authorization, and raw
OpenFOAM runtime files do not transfer with the Git branch. The committed
model, compact outputs, documentation, and PR description are the durable
handoff.
