# 本地 Agent 接手说明 — Case A2（Liu 2020）

本文档是给**本地 Cursor Agent** 的单一入口。先读完本节再改代码或渲染。

## 1. 身份与范围

| 项 | 值 |
|---|---|
| 论文工况 | Liu, Shao & Zhu (2020) **Series A / Case A2** |
| 流量 | `Q0=20 → Q1=100 L/s`，阀开 0.4 s |
| 下游 | 明渠 + 水箱/活动圆堰（**no-geyser**） |
| 求解器 | OpenFOAM.com **v2512** `interFoam`，4 MPI ranks |
| 唯一可改目录 | `tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/` |
| PR | https://github.com/brant123451/geysering/pull/9 |
| 分支 | `cursor/test3-a2-openfoam-1850`（base=`main`） |

**禁止**：改其他 Case、`paper/`、根 README、根 `.gitignore`、`main`；禁止为凑 no-geyser / 压力曲线去调堰顶或其它参数。

## 2. 当前状态（已完成）

替换模型（学位论文水箱+堰，`z_crest=0.031 m` 仅由 Q0/`hd` 校准；`Dr=0.057 m`；无 headbox；`t=-12…14.4 s`）的 **base 与 refined 全瞬态均已跑完并提交**。

| 指标 | 实验/目标 | base | refined |
|---|---:|---:|---:|
| 四面体 | — | 158,507 | 251,664 |
| 爬升前入口/堰 (L/s) | 20 / 20 | 19.981 / 20.079 | 19.453 / 20.106 |
| 涌波到达（ramp 时钟, s） | 1.60 | 1.538 | 1.525 |
| PT2 / PT3（论文 7–14 s 均值, kPa） | 2.15 / 4.99 | 0.517 / 2.562 | 0.417 / 2.750 |
| 喷发 | 否 | 否 | 否 |

结论要如实写：**涌波时刻与 no-geyser 分支可复现；稳态 PT2/PT3 与首柱高度仍偏低**，不能宣称已完整定量复现试验压力。

旧「固定下游水位 + headbox」表格已 superseded，见 `openfoam/3d/HANDOFF.md`；以 tank/weir 替换模型结果为准。

## 3. 克隆与 LFS（渲染前必做）

```bash
git fetch origin cursor/test3-a2-openfoam-1850
git checkout cursor/test3-a2-openfoam-1850
git lfs install
git lfs pull
```

大文件（Git LFS）：

* `openfoam/3d/case/postProcessing/riserAlpha/-12/alpha.water`（~108 MB）
* `openfoam/3d/case/postProcessing/tankLevel/-12/alpha.water`（~30 MB）
* `openfoam/3d/case/VTK/**/internal.vtu`（每个 ~15–16 MB）

未 `git lfs pull` 时这些文件只是 pointer，渲染会失败或得到空图。

## 4. 目录地图

```
A2_Q20to100_openchannel_nogeyser/
├── LOCAL_AGENT_HANDOFF.md     ← 你在这里（本地接手入口）
├── README.md                  ← Case 总说明（含 1D 模型与 3D 摘要）
├── config/case.json
├── data/digitized/            ← 论文数字化曲线
├── outputs/                   ← 紧凑指标、对比图、正视渲染产物
├── model/                     ← 冻结的一维复合域模型（勿与 3D 混谈）
├── scripts/                   ← 1D/数字化脚本
└── openfoam/3d/
    ├── SIMDATA.md             ← 已上传模拟数据清单
    ├── HANDOFF.md             ← Cloud Agent 历史交接（英文，含完整指标表）
    ├── NEW_AGENT_PROMPT.md    ← 原 Cloud 任务约束（仍有效的硬约束见下文）
    ├── PAPER_AUDIT.md         ← 论文逐项核对
    ├── README.md              ← 3D 复现命令与结果说明
    ├── render_front_water_air.py
    ├── postprocess_compare.py
    ├── make_geometry.py / make_gmsh_mesh.py / prepare_runtime.py
    └── case/                  ← refined 运行产物（已 force-add）
        ├── VTK/               ← t≈-12,12,13,14 体积场（purgeWrite=3）
        ├── -12/ 12/ 13/ 14/   ← 重建串行场
        ├── constant/polyMesh/
        ├── postProcessing/    ← 全时段探针/湿面积/通量
        ├── system/            ← controlDict、探针坐标等
        └── 0.orig/ + Allrun*
```

数据细节见 [`openfoam/3d/SIMDATA.md`](openfoam/3d/SIMDATA.md)。

**未上传**：`processor*`（与串行 4 时刻重复）、`log.interFoam.full`（~488 MB）。

## 5. 本地渲染（主要任务）

依赖：`python3` + `numpy` + `matplotlib` + `pyvista` + `Pillow`。

```bash
cd tests/test_03_liu2020/cases/A2_Q20to100_openchannel_nogeyser/openfoam/3d
python3 render_front_water_air.py
```

脚本会写出到 `../outputs/`（以及若存在则写 `/opt/cursor/artifacts/`）：

| 产物 | 含义 |
|---|---|
| `openfoam_3d_refined_front_water_air.png` | 四时刻 VTK `y=0` 正视拼图 |
| `openfoam_3d_refined_front_full_t*.png` | 单时刻双面板（全域 + 管段放大） |
| `openfoam_3d_refined_front_complete_strip.png` | 四时刻 strip |
| `openfoam_3d_refined_front_complete_motion.gif` | ~133 帧完整正视动图 |
| `openfoam_3d_refined_front_motion_*.png/gif` | 探针重建时间轴 |

说明：

* **体积场只保留 4 个时刻**（`purgeWrite=3`）。动图中间帧用 `postProcessing` 探针重建。
* 下游是明渠，`hd≈0.07–0.19 m`；全高正视里像细线，必须看 **pipe zoom** 面板。
* 右侧管道**有水**；不是空管。

ParaView：直接打开 `case/VTK/case.vtm.series` 或各 `case_*.vtm`，切片 `y=0`，着色 `alpha.water`。

## 6. 硬约束（接手后仍必须遵守）

1. 比较压力用大气表压 **`p`（kPa）**，不要直接拿 `p_rgh` 对论文。
2. 时间对齐：`t_sim = t_paper + 0.4 s`（论文 `t=0` = 阀全开；模拟 `t=0` = 开始爬升）。
3. **禁止**用瞬变压力、竖管高度或 no-geyser 结果反调 `z_crest`、入口、初始液位等。
4. 堰顶 `z_crest=0.031 m` 只允许由报告的 `(Q0=20 L/s, hd=0.070 m)` 水力校准得出。
5. PT3 文献自相矛盾（0.99 kPa vs「0.10 m 水深」）；初液面 `z=0.12 m` 来自 0.99 kPa 字面，松弛后约 0.79–0.82 kPa，属已知源问题。
6. 下游水深必须用管内湿面积换算，**不能**用水箱水位冒充 `hd`；测点 `x=0.60/3.25/6.00 m`。

## 7. 复跑三维（通常不需要）

仅当本地要重算网格/求解时：

```bash
cd openfoam/3d/case
NP=4 ./Allrun refined    # 或 base
# 中断后续跑：
NP=4 ./Allrun.resume
```

环境：OpenFOAM.com v2512、Gmsh Python API、NumPy、Matplotlib、4 ranks。全时段墙钟量级：base ~1 天、refined 更长。

后处理紧凑对比：

```bash
cd openfoam/3d
python3 postprocess_compare.py
```

## 8. 建议本地 Agent 优先做的事

1. `git lfs pull`，确认 `case/VTK/*/internal.vtu` 与 `postProcessing/riserAlpha` 是实体文件而非 130 字节 pointer。
2. 跑 `render_front_water_air.py`，在本地检查正视/动图；按需改进相机、配色或导出 ParaView 状态。
3. 若要更密的体积动画：需在新 run 中增大 `writeInterval` 或关闭/放宽 `purgeWrite` 后重算（**不得**为此改物理参数凑结果）。
4. 文档/指标有出入时以 `outputs/openfoam_3d_refined_metrics.json` 与 `openfoam/3d/HANDOFF.md` 表格为准。

## 9. 相关文档阅读顺序

1. **本文** `LOCAL_AGENT_HANDOFF.md`
2. `openfoam/3d/SIMDATA.md` — 数据清单
3. `openfoam/3d/HANDOFF.md` — 完整指标与 Cloud 历史
4. `openfoam/3d/PAPER_AUDIT.md` — 论文几何/IC/BC 核对
5. `openfoam/3d/README.md` — 复现命令
6. `openfoam/3d/NEW_AGENT_PROMPT.md` — 原验收标准（其中「禁止提交 postProcessing」已被本分支为本地渲染而 force-add 覆盖；**其它物理/范围约束仍有效**）

## 10. 一句话给本地 Agent

> 这是已完成的 A2 三维 no-geyser 验证 Case；物理结果与紧凑输出已在 PR 分支。你的职责是拉取 LFS 模拟数据、本地渲染/可视化或文档整理；不要改参数去「修好」偏低的 PT2/PT3，也不要动本目录之外的文件。
