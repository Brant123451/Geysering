# Geysering 应用论文（LaTeX 工程）

应用型论文：用模型论文（`E:\Research\论文\mixed_flow_submission_en_clean_20260526_1930`，
解耦两流体 cut-cell 有限体积框架）复现 geysering 试验。与方法学论文互为配套：
方法学论文讲模型与算法，本篇讲对试验的定量复现与工程适用性。

## 结构

```text
main.tex                    主文件（elsarticle preprint）
sections/introduction.tex   引言
sections/model.tex          模型框架概述 + 竖井分支 / 结点闭合 / 数值配置
sections/tests.tex          Tests 章节（核心）
sections/discussion.tex     Discussion + Conclusions
sections/bibliography.tex   参考文献（thebibliography，按引用顺序）
figures/                    结果图（从复现文件夹拷贝的快照）
zh/                         中文稿主文件与中文构建产物
```

## Tests 章节现状（对比论文最终定为 3 篇）

选择逻辑：分支判别与时程（VW2011）→ 判据地图与参数扫描（Cong2017）→ 原型尺度工程对照（Liu2020）。
`references/` 里其余论文（wright2011 现场札记、muller2017 大尺度/3D 标定、qian2020 抑制措施、
leon2018 竖管 violent geyser、zhou2019 Taylor 气泡判据）只作引用佐证，不做复现对象。

- **Campaign 1 = Vasconcelos & Wright (2011, JHE)**，已完成：
  - **Test A**：大塔 `Dt/D=0.607`（不喷发分支），图 = `caseA_pressure/levels/snapshots.png`；
  - **Test B**：小塔 `Dt/D=0.135`（喷发分支），图 = `caseB_pressure/levels.png`。
  - 图与指标来源：`tests/test_01_vw2011/cases/A_*/outputs/` 与
    `tests/test_01_vw2011/cases/B_*/outputs/`
    （2026-07-06 冻结版求解器输出；数值表引用各 `*_comparison_metrics.json`）。
- **Campaign 2 = Cong, Chan & Lee (2017, JHE)**，已完成（判据地图 + B-H 系列）：
  - 图 = `cong2017_bh_series.png`（B-H1..B-H7 分类/到达时间/速度对照）、
    `cong2017_signature.png`（B-H1/B-H6 签名工况）、`cong2017_criterion_map.png`（63 例判据地图）。
  - 数据来源：`tests/test_02_cong2017/cases/BH1_*/`、
    `tests/test_02_cong2017/cases/BH6_*/` 与
    `tests/test_02_cong2017/studies/criterion_map/outputs/`。
  - 与 Campaign 1 使用同一冻结求解器和全同步 T 接口耦合；真实盲测成绩为
    Series B **5/7**、63 构型 **39/63**，24 处分歧均为漏报喷发、无假阳性。
    旧分级耦合的 7/7、62/63 结果仅保留为历史对照，不再作为论文结论。
- **Campaign 3 = Liu, Shao & Zhu (2020, JHE)**，已完成：
  - A2/B3 下游边界分支对（明渠不喷发 / 满管喷发）；
  - C9 第一阶段瞬变振荡喷发及 Eq.(7) 对照；
  - 数据与 HTML 报告位于 `tests/test_03_liu2020/cases/A2_*/`、
    `tests/test_03_liu2020/cases/B3_*/`、`tests/test_03_liu2020/cases/C9_*/`。

## 编译

```powershell
cd paper
pdflatex -interaction=nonstopmode main.tex   # 跑 2~3 遍解决交叉引用
```

产物 `main.pdf`。文中红色 `[TODO: ...]` 为待定内容（目标期刊、配套论文 DOI、
Campaign 2/3 正文），用 `\todo{}` 命令标记，全文检索 `todo` 即可定位。

## 结果更新流程

复现结果更新后：重跑对应 case 的 `caseX_digitize_and_compare.py` →
把新的 `outputs\*.png` 拷入 `figures\`（文件名对应关系见上）→
按新的 `*_comparison_metrics.json` 更新 `sections/tests.tex` 中
Table 2 / Table 3 的数值与正文事件时刻 → 重新编译。
