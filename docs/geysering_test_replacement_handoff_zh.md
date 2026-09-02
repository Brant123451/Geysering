# Geysering 论文新增/替换 Test 讨论交接

> 用途：把本窗口关于“是否新增第四组 geysering 验证，或替换现有 Test 2”的讨论、检索结果、筛选逻辑和未完成工作交给另一个 Codex 窗口继续。
>
> 状态：**讨论与候选筛选记录，不是已经批准的论文修改方案。** 截至 2026-08-08，尚未决定最终替换论文，也没有删除现有 Test 2 或 Test 3。

## 1. 当前论文的三组验证是什么

当前英文论文以 `paper/sections/tests.tex` 为准，包含三组互补验证：

1. **Campaign/Test 1 — Vasconcelos & Wright (2011)**
   - 单根通风竖管，包含不喷发 Test A 和喷发 Test B。
   - 作用：标准基准；对比压力、气水界面、自由水面和喷发分支。
   - 结论：必须保留。它是本文最基础、最可识别的 geysering benchmark。

2. **Campaign/Test 2 — Cong, Chan & Lee (2017)**
   - 水平管＋单根竖管；改变竖管直径、初始气囊体积和外部水头。
   - 当前论文不只是三个工况：已有 7 个 Series-B 工况和 63 个参数构型的 1D 分支扫描；OpenFOAM 受算力限制，只能重点做约三个代表工况。
   - 作用：唯一明确测试“喷发/不喷发分类能力”和参数空间判据的 campaign。

3. **Campaign/Test 3 — Liu, Shao & Zhu (2020)**
   - 交汇室＋竖井＋上下游管道，来自 Edmonton 原型的 1:20 实体模型。
   - 当前重点工况 A2、B3、C9，包含下游明流/满管切换、瞬变压力驱动和气囊驱动喷发。
   - 作用：拓扑和机理比 Test 1/2 更复杂，更接近工程交汇结构。

## 2. 为什么开始讨论新增或替换

用户的核心担忧是：

- 论文看起来只与三篇试验论文比较，是否显得验证对象太少；
- Test 1 和 Test 2 都是“水平管＋单竖管＋气囊到达竖管”的简化结构，视觉上重复；
- 希望增加更复杂、能明显体现 1D 网络模型优势的 geysering 构型，理想情况是**两根竖管/竖井**；
- Test 2 的 OpenFOAM 预算只允许约三个代表工况，担心无法形成足够强的 1D/2D 对比；
- 论文篇幅已经较长，不希望简单再堆一个完整 Test。

讨论中形成的初步判断：

- 论文长并不是因为“引用了三篇论文”，而是每个 campaign 下有多工况、曲线、分支分类、1D/2D 比较和机理讨论。
- **不建议直接新增第四个完整 campaign。** 若新文献足够强，优先“替换一个”，避免结果章节进一步膨胀。
- Test 1 必须保留；Test 3 的工程拓扑和机理复杂性通常强于 Test 2，因此当时的替换倾向是：**保留 Test 1 和 Test 3，考察是否用新论文替换 Test 2。**
- 但需要注意：删除 Test 2 会失去当前论文唯一的 7 工况＋63 构型喷发分类证据。一个只有单个猛烈喷发工况的新 Test，可能“画面更复杂”但“验证维度反而更窄”。因此替换必须做证据收益比较，不能只看装置图。

## 3. 筛选条件如何逐步收紧

### 第一轮理想条件

希望候选同时满足：

1. 有真实实体试验，不是只做数值模拟；
2. 至少两根竖管/竖井；
3. 明确发生水气混合物喷发，而不是只研究跌水、排气或气囊形成；
4. 是正式期刊论文；
5. 期刊不是 *Water*；
6. 论文提供足够的管径、长度、初始水位、气囊、阀门/流量和边界条件，可搭 OpenFOAM；
7. 工况复杂性明显高于现有 Test 1/2；
8. 用户明确排除 Lokhandwala（对话中曾写作 “Okhandwala”）2026。

### 严格筛选后的结论

目前检索到的公开文献中，**没有一篇正式期刊论文同时满足“实体试验＋双竖井＋真实喷发＋非 Water＋参数足够复现”全部条件。**

最接近的 Lu et al. (2025) 确实声称做了单井和双井 geyser 试验，但目前公开材料只是 IAHR 世界大会扩展摘要，缺少完整装置和工况参数，不能直接承担可复现的论文验证。

### 后续放宽条件

由于双竖井严格条件没有合格对象，讨论后来改为：

- 允许只有一根竖管；
- 但必须是真实喷发试验、非 *Water* 期刊；
- 工况或边界物理必须比当前单纯“气囊经过竖管”更强；
- 必须有足够数据支持 OpenFOAM 和本文 1D 模型复现。

## 4. 候选论文与筛选结果

| 候选 | 实体试验 | 两根竖向结构 | 真实喷发 | 正式非 Water 期刊 | 参数可复现性 | 当前结论 |
|---|---:|---:|---:|---:|---:|---|
| Lokhandwala et al. (2026), *Initiation of Air Trapping between Two Manholes...* | 否/核心为分析与建模 | 是，两检查井 | 否，只研究气囊形成前提 | 是，JHE | 中 | 用户明确排除；不是喷发验证 |
| Lu et al. (2025), *Systematical Experimental Investigations...* | 是 | **是** | **是** | **否，IAHR 扩展摘要** | 低 | 物理目标最匹配，但材料不足，暂不能模拟 |
| Ye, Zheng & Ma (2025), *Effect of Downstream Water Level...* | 是，大型物模 | 是：上游跌水井＋下游排气井 | 否 | 是，TAML | 较高 | 研究跌水携气/排气，不是 geyser 喷发，不适合替换 |
| Allasia et al. (2023), *Experimental Study of Geysering in an Upstream Vertical Shaft* | 是 | 否 | 是 | **否，Water** | 高 | 因期刊条件排除 |
| León, Elayeb & Tang (2019), *An Experimental Study on Violent Geysers...* | **是** | 否 | **是，猛烈连续喷发** | **是，JHR** | 高 | 放宽为单竖管后最强物理候选之一；OpenFOAM 成本很高 |
| Wang, Qian & Chen (2019), *Experimental Study on Geysers Induced...* | **是** | 否 | **是，两类 geyser** | 是，Applied Sciences | **高** | 可复现，但拓扑仍与 Test 1/2 接近，复杂性增益有限 |
| Zhang et al. (2022), *Experimental Study on Geysers in Covered Manholes...* | **是** | 否 | **是** | **是，JHE** | 待全文审计 | 井盖、通气口、固定/活动盖带来新边界物理；值得重点复核 |
| Li et al. (2023), *Modeling Geysers Triggered by an Air Pocket...* | 核心不是新实体试验 | 否 | 数值上有 | 是，Physics of Fluids | 数值几何较清楚 | 使用 ANSYS Fluent 2022R1 的 3D VOF；不满足“真实试验”硬条件 |

## 5. 重点候选的详细判断

### 5.1 Lu et al. (2025)：唯一明确提到双竖井喷发试验，但不是完整期刊论文

- 题名：*Systematical Experimental Investigations of Air-Water Interactions in Geysers*。
- 作者：Yanqing Lu, Ling Zhou, David Ferras, Capucine Dupont, Qianuxun Chen, Saber Nasaoui。
- 出处：41st IAHR World Congress，Book of Extended Abstracts，2025；公开页面没有 DOI。
- 摘要明确写到：
  - 上游水快速充入空的封闭末端管道；
  - 先研究缓倾管混合流，再研究单竖直流；
  - 最后做单竖井和上下游双竖井 geyser 试验；
  - 单井时两相混合物猛烈进入竖井并喷发；
  - 上下游双井会明显降低上游压力峰值；
  - 竖井内主要是 churn flow，并伴随 Taylor bubble 和气泡合并/破碎。
- 这是目前最贴合“双竖井＋试验＋喷发”的材料。
- 致命问题：公开内容只是扩展摘要；没有足够完整的管径、管长、两井间距、阀门开启、初始水气分布和逐工况边界数据。我们此前看示意信息后得出的结论是：**仅凭现有公开摘要无法可靠搭建 OpenFOAM，也不应猜参数。**
- 后续窗口可以做的唯一有效动作：找完整论文、会议全文、作者预印本或博士论文；若仍只有摘要，则终止该候选。
- 官方页面：https://www.iahr.org/library/infor?pid=39265

### 5.2 Ye, Zheng & Ma (2025)：两根竖向结构，但本质是跌水井—隧洞—排气井系统

- 题名：*Effect of Downstream Water Level on Operation Features of a Dropshaft–Tunnel System*。
- 期刊：*Theoretical and Applied Mechanics Letters*, 15 (2025), 100600。
- DOI：https://doi.org/10.1016/j.taml.2025.100600
- 装置：大型物理模型，包含上游进水跌水井（#1）、地下水平隧洞、下游排气井/airshaft（#2）和下游水箱。
- 研究量：跌水携气、管内波状流或气塞流、气压分布、下游淹没水位对排气阻力的影响。
- 我们讨论“中间/下游竖管里会发生什么”时的判断：主要发生气体排放、气水塞状/波状流和压力响应；论文没有把该排气井作为水气混合物冲出地面的 geyser 喷发竖井。
- 因此它满足“实体试验＋两根竖向结构＋非 Water”，但不满足“真实喷发”，不适合替换 geysering 验证。它更适合未来做 deep-tunnel air-management 应用，而不是本文 geyser campaign。

### 5.3 León, Elayeb & Tang (2019)：FIU 暴烈 geyser 试验

- “FIU” 指 **Florida International University（佛罗里达国际大学）**，不是软件，也不是某个数值模型；Arturo S. León 当时的单位是 FIU。
- 题名：*An Experimental Study on Violent Geysers in Vertical Pipes*。
- 期刊：*Journal of Hydraulic Research*, 57(3), 283–294。
- DOI：https://doi.org/10.1080/00221686.2018.1494052
- 试验真实性与数据：
  - 透明 PVC 水平管和竖管，内径均为 152.4 mm；
  - 上游约 1.7 m³ 加压水箱；
  - 竖管长度/初始水深为 3、6、9、12 m；
  - 两种初始水量；
  - 8 个主构型，每个重复 15 次，共 120 次；
  - 还测试竖管底部孔板，孔径比 1/2、1/4、1/8；
  - 喷发由高压气体和水平管快速变化压力梯度驱动，可能出现数次连续猛烈喷发，换算高度可超过 30 m。
- 与现有 Test 的区别：
  - 几何拓扑仍是单水平管＋单竖管，并不比 Liu 2020 的交汇室更复杂；
  - 但边界与物理强度明显更复杂：高压可压缩气源、长竖直水柱、破口后的连续喷发、孔板控制和外部自由射流。
- OpenFOAM 难点：大尺度三维外部喷流、长竖管、高速可压缩气体、压力罐耦合以及多次喷发；如果完整模拟到 30 m 自由射流，计算域和网格成本很高。
- 可行的降阶复现方案应事先约定：只算管内＋竖管口通量/压力，不追踪完整 30 m 外部喷流；或者只选 3 m/6 m 和一个孔板工况。但这种裁剪会削弱“猛烈喷发高度”的核心证据，必须在论文中披露。
- FIU 公开论文页：https://discovery.fiu.edu/display/pub135853

### 5.4 Allasia et al. (2023)：试验和参数都好，但用户排除 Water

- 题名：*Experimental Study of Geysering in an Upstream Vertical Shaft*。
- 期刊：*Water*, 15(9), 1740。
- DOI：https://doi.org/10.3390/w15091740
- 实体试验；两种竖管构型、两种水头、25/45 L 气囊、快速/渐开阀门，共 16 个条件。
- 装置是一根上游端部竖管，不是双竖井；参数较完整，模拟可行性强。
- 用户明确说“Water 就算了”，因此不作为替换对象。

### 5.5 Lokhandwala et al. (2026)：两检查井，但研究的是气囊如何形成

- 正确拼写是 **Lokhandwala**；题名：*Initiation of Air Trapping between Two Manholes in a Stormwater Conduit*。
- 期刊：*Journal of Hydraulic Engineering*, 152(5), 2026。
- DOI：https://doi.org/10.1061/JHEND8.HYENG-14514
- 它研究两检查井之间相向充水波如何截留气体，是 geyser 风险的前置条件，不是实体喷发 benchmark。
- 用户已明确排除，后续无需重新推荐。

### 5.6 Wang, Qian & Chen (2019)：参数最完整，但与现有 Test 2 相似

- 题名：*Experimental Study on Geysers Induced by the Release of Trapped Air in Storage Tunnel Systems*。
- 期刊：*Applied Sciences*, 9(24), 5326。
- DOI：https://doi.org/10.3390/app9245326
- 实体装置：2.00×1.60×1.20 m 水箱；上游管 8 m、下游管 6 m；水平管直径 0.15 m；竖管高 1.35 m、直径 0.05–0.15 m；气囊长 0.60–2.40 m；阀门约 0.25 s 打开；有 8 组无外压和 64 组有外压试验。
- 区分 gas-flow geyser 与 surge-type geyser，视频、相界面和压力资料比较完整。
- 优点：最容易重新搭建 1D/OpenFOAM，数据量也大。
- 缺点：仍是标准单竖管 capsule experiment，与 V&W/Cong 的核心拓扑高度重复，难以证明“新 Test 更复杂”。若替换 Cong，可能只是换了另一组相似参数扫描。
- 原文：https://doi.org/10.3390/app9245326

### 5.7 Zhang et al. (2022)：建议下一窗口重点复核的现实候选

- 题名：*Experimental Study on Geysers in Covered Manholes during Release of Air Pockets in Stormwater Systems*。
- 期刊：*Journal of Hydraulic Engineering*, 148(5), 06022003。
- DOI：https://doi.org/10.1061/(ASCE)HY.1943-7900.0001978
- 实体试验考察固定与活动井盖、井径和通风开口，属于真实 geyser，且不是 *Water*。
- 它虽然仍是单 manhole，但引入了盖板运动/受限排气，是现有三个 campaign 没有的边界物理；从“新机制”角度可能比单纯追求双竖井更有价值。
- 当前未完成：尚未系统提取全文装置尺寸、初边值、工况表、可数字化曲线和 OpenFOAM 计算域，因此不能直接确定替换。

### 5.8 Li et al. (2023)：复杂，但不满足真实试验硬条件

- 题名：*Modeling Geysers Triggered by an Air Pocket Migrating with Running Water in a Pipeline*。
- 期刊：*Physics of Fluids*, 35, 045126。
- DOI：https://doi.org/10.1063/5.0138342
- 使用 ANSYS Fluent 2022R1 的瞬态 3D VOF，讨论 air-releasing、rapid-filling 和 hybrid geyser；模型用已有试验做验证，但该论文的主要新证据是商业 CFD 参数扫描。
- 因用户要求“必须是真实做实验的候选”，它不能作为新的实验 validation campaign；可以作为机理和 3D CFD 对照文献。

## 6. 到目前为止的决策结论

### 已基本确定

1. **不直接增加第四个完整 Test。** 当前篇幅和证据已经很多，优先替换或把新构型放补充材料。
2. **Test 1 保留。**
3. **Test 3 倾向保留。** 它的交汇室、上下游边界和两阶段驱动比单竖管 benchmark 更接近工程系统。
4. 若必须替换，历史讨论倾向替换 Test 2；但必须先评估失去 63 构型分类图的代价。
5. 严格“双竖井＋试验＋喷发＋非 Water 期刊”目前无合格对象。
6. Lu 2025 不能在缺少全文参数时开算；Ye 2025 不是 geyser 喷发；Lokhandwala 2026 已排除。

### 尚未最终确定

1. 放宽为单竖管后，是选择 FIU/León 2019 的猛烈喷发，还是 Zhang 2022 的带盖 manhole。
2. 新候选是否真能用本文冻结 1D 模型表示，而不是必须引入外部三维喷流、活动盖板或额外结构动力学。
3. 新候选能否提供至少：
   - 一张装置图；
   - 一张动态/多帧对比图；
   - 两类定量曲线（压力、界面/水位、喷发高度或流量）；
   - 一个跨工况表格；
   - 可审计的初始条件和边界条件。
4. 替换后论文的核心贡献是否更强。不能仅以“装置更复杂”作为理由，必须说明新增了哪一种现有三组没有覆盖的模型能力。

## 7. 给下一个窗口的推荐工作顺序

1. **不要立刻删除 Test 2。** 先做一页 candidate scorecard。
2. 优先精读并提取两篇全文：
   - Zhang et al. (2022) covered manholes；
   - León et al. (2019) violent geysers。
3. 对每篇建立同样的证据表：几何、材料、初始水位、气体状态、阀门/流量、边界、测点、采样率、工况数、主要图表、可数字化数据。
4. 评估本文 1D 模型能否表示：
   - León：压力罐边界、长竖管、孔板、喷口后的质量流出；
   - Zhang：受限通气、井盖压力/运动。若活动井盖必须做结构动力学，先判断是否超出论文模型范围。
5. 为两篇各选 1–3 个代表工况，估算 OpenFOAM 网格规模、物理时间、是否需要可压缩多相、是否必须计算外部喷流。
6. 将候选与 Cong 2017 对比，不只比较画面：
   - 现有 Cong 提供 7 工况＋63 构型分类；
   - 新候选必须提供新的机理维度或更强定量证据，才能值得替换。
7. 最终给用户三个明确选项：
   - 保留现有三组；
   - 用某篇替换 Test 2；
   - 保留三组，仅把新候选做补充/演示，不进入主文。

## 8. 可直接粘贴给另一个窗口的任务说明

```text
请先阅读 E:\Geysering\docs\geysering_test_replacement_handoff_zh.md，继续“是否用新的复杂 geysering 实验替换现有 Test 2”的研究。不要删除或修改现有 paper/sections/tests.tex，也不要启动 OpenFOAM。

优先获取并精读：
1) Zhang et al. (2022), Experimental Study on Geysers in Covered Manholes during Release of Air Pockets in Stormwater Systems, JHE, DOI 10.1061/(ASCE)HY.1943-7900.0001978；
2) León et al. (2019), An Experimental Study on Violent Geysers in Vertical Pipes, JHR, DOI 10.1080/00221686.2018.1494052。

对两篇分别提取：完整装置尺寸、初始/边界条件、工况表、测点与采样、可数字化曲线、喷发照片/多帧、是否能由当前 1D 模型表达、OpenFOAM 计算成本与必须简化的部分。然后与 Cong 2017 当前 7 工况＋63 构型分类证据做定量 scorecard，给出“保留/替换/补充材料”三选一建议。所有未从全文确认的信息标为待核实，不得猜参数。
```

## 9. 本文档使用的项目内证据

- `paper/sections/tests.tex`：现有三组 campaign、工况和论文叙事。
- `paper/sections/bibliography.tex`：V&W 2011、Cong 2017、Liu 2020、Zhang 2022、Qian 2020、León 2019 等正式文献信息。
- `tests/test_01_vw2011/`、`tests/test_02_cong2017/`、`tests/test_03_liu2020/`：现有算例与输出。
- `C:\Users\Administrator\.codex\skills\geysering-paper\references\project-map.md`：当前三组 campaign 的项目映射和证据优先级。

## 10. 重要提醒

- 这次讨论改变的是论文验证策略，属于**科学内容与篇章结构同时变化**，不是单纯排版。
- 任何替换都会要求同步修改英文 `paper/sections/tests.tex`、Introduction、Discussion、Abstract、Conclusions 和中文 `paper/zh/main_zh.tex`。
- 新论文的装置图只是展示材料；定量结论必须追溯到全文工况表、曲线或原始数据。
- 不要把会议摘要、商业 CFD 结果或缺失参数的装置包装成“实验验证”。
- 当前工作区有大量用户生成和未提交文件；后续窗口必须保持就地修改，不清理、不重置、不覆盖现有算例。
