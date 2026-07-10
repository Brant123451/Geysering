# Geysering 文件与论文目录重组设计

## 1. 目标

将 `E:\Geysering` 整理为清晰的三层结构：

1. 三个实验 Test/Campaign 各自独立；
2. 每个实际复现或已产出数据的 Test-Case 各自拥有可独立理解、复跑和审阅的子目录；
3. 中英文论文采用平行工程结构，并能追溯每张论文图所对应的 Test-Case。

整理后不再依赖仓库根目录、Test 根目录或历史 `reproduction` 树中的 Case 专属文件。

## 2. 范围

### 2.1 Test 1 — Vasconcelos & Wright (2011)

- `A_Dt57p1_Ha0305_Yfs0356`
- `B_Dt12p7_Ha0610_Yfs0356`
- `Fig10_Dt57p1_Ha0305_Yfs0254`
- `Fig11_Dt12p7_Ha0305_Yfs0254`

### 2.2 Test 2 — Cong, Chan & Lee (2017)

- `BH1_Dr16_H066_L061`
- `BH2_Dr21_H066_L061`
- `BH3_Dr26_H066_L061`
- `BH4_Dr31_H066_L061`
- `BH5_Dr36_H066_L061`
- `BH6_Dr41_H066_L061`
- `BH7_Dr46_H066_L061`
- Campaign 级系列扫描保留为 `studies/criterion_map`，不伪装成单独 Case。

### 2.3 Test 3 — Liu, Shao & Zhu (2020)

- `A2_Q20to100_openchannel_nogeyser`
- `B3_Q20to100_fullpipe_geyser`
- `C9_Q25to40_hr03_airpocket`

不为只有论文参数、但本项目没有代码或产出数据的实验矩阵行创建空目录。Liu C9 phase-2 当前未计算，因此记录在 Case README 的范围说明中，不另建空 Case。

## 3. 目标目录

```text
E:\Geysering\
├─ tests\
│  ├─ test_01_vw2011\
│  │  ├─ README.md
│  │  ├─ _shared\
│  │  │  ├─ reference\
│  │  │  ├─ metadata\
│  │  │  └─ tools\
│  │  └─ cases\
│  │     ├─ A_Dt57p1_Ha0305_Yfs0356\
│  │     ├─ B_Dt12p7_Ha0610_Yfs0356\
│  │     ├─ Fig10_Dt57p1_Ha0305_Yfs0254\
│  │     └─ Fig11_Dt12p7_Ha0305_Yfs0254\
│  ├─ test_02_cong2017\
│  │  ├─ README.md
│  │  ├─ _shared\
│  │  ├─ studies\criterion_map\
│  │  └─ cases\BH1_... 至 BH7_...
│  └─ test_03_liu2020\
│     ├─ README.md
│     ├─ _shared\
│     └─ cases\A2_...、B3_...、C9_...
├─ paper\
│  ├─ README.md
│  ├─ en\
│  │  ├─ main.tex
│  │  └─ sections\
│  ├─ zh\
│  │  ├─ main.tex
│  │  └─ sections\
│  ├─ shared\
│  │  ├─ figures\
│  │  ├─ bibliography\
│  │  └─ styles\
│  ├─ build\
│  │  ├─ en\
│  │  └─ zh\
│  └─ archive\
├─ references\
├─ tools\
└─ docs\
```

## 4. Case 目录契约

每个 Case 至少包含：

```text
README.md
manifest.yaml
config\
data\
model\
scripts\
reference\
outputs\
```

仅需要 OpenFOAM 的 Case 增加 `openfoam\`。

- `README.md`：物理工况、论文对应关系、运行命令、输出说明、当前验证状态。
- `manifest.yaml`：Test、Case ID、参数、来源论文、入口脚本、模型版本、关键输出和状态。
- `config/`：Case 参数和运行配置。
- `data/`：数字化实验数据、输入数据和 Case 专属测量表。
- `model/`：能够独立复跑该 Case 的冻结模型代码。
- `scripts/`：取图、数字化、运行、后处理和报告生成入口。
- `reference/`：该 Case 对应的论文截图、图表裁剪和局部参考资料。
- `outputs/`：CSV、JSON、PNG、PDF、GIF、HTML 等可审阅结果。
- `openfoam/`：网格、初始条件、常量、系统配置和必要的后处理脚本；不保存大体积时间步和分区运行目录到 Git。

Case 不依赖另一个 Case 的相对路径，也不从仓库根目录导入临时模型副本。

## 5. 共享资产

- Test 原始论文、参数矩阵和跨 Case 工具进入对应 Test 的 `_shared/`。
- 跨三个 Test 的参考文献进入根目录 `references/`。
- 通用数字化、目录验证和构建工具进入根目录 `tools/`。
- Test 2 的 Series B 与 63 构型扫描进入 `studies/criterion_map/`；BH1–BH7 各自从扫描结果中获得独立参数、数据切片和说明。
- 每个 Case 保留冻结模型副本；Test `_shared` 可以保留模型来源和版本说明，但不能成为 Case 运行时的隐藏依赖。

## 6. 论文框架

- 英文和中文工程分别放在 `paper/en` 与 `paper/zh`。
- 共享图件、参考文献和样式放在 `paper/shared`。
- 英文和中文最终 PDF 分别输出到 `paper/build/en` 与 `paper/build/zh`。
- 历史 `v3`、`v4`、`rev` 等版本进入 `paper/archive`，不再与当前 PDF 混放。
- LaTeX 的 `.aux`、`.log`、`.out`、`.spl` 等构建缓存不归档。
- `paper/shared/figures/manifest.yaml` 记录每个发布图件的源 Case、源输出路径和生成脚本；发布用快照是有意保留的论文资产，不按误重复删除。

## 7. 清理与去重

- 对完全相同的重复文件先计算 SHA-256，再保留语义最正确位置的一份。
- 删除记录写入 `docs/file-layout/deletion-manifest.csv`，至少包含原路径、大小、SHA-256、保留副本或删除原因。
- `_tmp_*`、`_dev_*`、`_slugdbg*`、`.bak*`、缓存和纯调试日志在确认不被正式入口引用后删除。
- 唯一且仍有解释价值的历史实现进入所属 Test 的 `_archive/legacy`。
- `caseB/_caseA_unified_check` 等交叉污染目录拆回正确 Test/Case；无价值输出删除。
- `reproduction` 中与正式 Case 字节相同的资产删除；唯一的历史实现归档后移除平行目录。

## 8. Git 收录策略

整理后 Git 收录：

- 源代码、配置、README、manifest；
- 参数表、数字化数据、CSV、JSON；
- 报告 HTML；
- 可审阅的 PNG、PDF、GIF 和关键动画；
- 中英文论文源文件和最终 PDF。

Git 排除：

- OpenFOAM 数值时间步；
- `processor*` 分区目录；
- 大量逐帧图片目录；
- Python/LaTeX 缓存；
- 临时文件和运行日志；
- 单文件超过 25 MiB 的生成物，除非在清单中明确白名单。

不使用 `git add -f .` 绕过规则；`.gitignore` 改为按生成物类别排除，而不是当前的全局 `*` 加少量白名单。

## 9. 迁移与回滚

迁移按以下顺序执行：

1. 记录迁移前文件清单、大小和哈希；
2. 建立目标骨架与 Case manifest；
3. 迁移 Test 1、Test 2、Test 3；
4. 拆分 Test 2 BH2–BH5、BH7 的数据和配置；
5. 整理论文中英文工程；
6. 去重并清理临时文件；
7. 更新所有绝对路径、相对路径、README 和 `.gitignore`；
8. 执行结构、导入、轻量运行和论文构建验证。

在整个迁移通过验证前保留迁移清单，不提交也不推送。若验证失败，依据迁移清单恢复原路径。

## 10. 验收标准

- 三个 Test 均有 README、`_shared` 和 `cases`。
- 上述 14 个目标 Case 均有独立目录和完整 Case 契约。
- 每个 Case 的主要运行入口不依赖其他 Case 或旧根路径。
- 根目录和 Test 根目录不再存在 Case 专属 `_tmp`、`_dev`、备份或模糊 `outputs`。
- 除有清单的论文发布快照外，不存在已知字节级重复输出。
- BH1–BH7 均能从独立 Case 目录识别参数、输入、模型和结果。
- 英文与中文论文均能从新目录构建，最终 PDF 位于 `paper/build`。
- 全仓搜索不到已迁移的旧绝对路径。
- `.gitignore` 能收录全部约定资产，并排除大体积运行状态。
- `git status` 中的变化可按 Test、论文和基础设施清楚审阅。
