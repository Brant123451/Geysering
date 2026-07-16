# Liu, Shao & Zhu (2020), JHE 146(2):04019055 — apparatus & experimental parameters
Source: references/liu2020.pdf (parsed). DOI: 10.1061/(ASCE)HY.1943-7900.0001660

原型：Edmonton 27 m 深竖井，交汇室连接 3.5 m 直径进水管与 4 m 跌落的出水管。
实验室 1:20 概化模型。

## Apparatus (Fig. 2)

| component | value | note |
|---|---|---|
| upstream pipe | L=5.80 m, D=0.20 m, slope 1:100 | 由承压水箱供水，阀门阶跃增流 |
| junction chamber | 0.30 × 0.30 × 0.45 m (L×W×H) | 透明 PVC 长方体 |
| invert drop in chamber | 0.18 m | 上下游管底高差 |
| downstream pipe | Ld=5.95 m, Dd=0.28 m, horizontal | 末端溢流堰/尾门控制 |
| riser | dr=0.06 m, length 1.22 m | 接交汇室顶盖中心，顶部敞开 |
| valve opening time Tv | ~0.4 s | 解析模型 Eq.(7) 用 |

## Pressure transducers（OMEGA, 130 kPa, 0.2%）

| PT | position |
|---|---|
| PT1 | 竖管壁，交汇室顶盖以上 0.80 m |
| PT2 | 交汇室顶盖（**主对标点**） |
| PT3 | 交汇室前壁，底以上 0.02 m |
| PT4 | 上游管顶，交汇室上游 0.30 m |

## Test matrix（Table 1，36 组，均为流量阶跃 Q0 → Q1）

- **Series A**（12 组）：下游明渠流（hd/Dd = 1/4，堰控）。
  Q0 = 20/30/40/50 L/s；Q1 = 80/100/120 L/s。
  编号规律 A1=(20,80), A2=(20,100), A3=(20,120), A4=(30,80), ...
  **全系列无 geyser**（下游泄流能力足够），竖管内混合物最大高度 h < 竖管高（Fig. 4）。
- **Series B**（12 组）：下游满管流。
  B1=(20,60), B2=(20,80), B3=(20,100), B4=(30,60), B5=(30,80), B6=(30,100),
  B7=(40,60), B8=(40,80), B9=(40,100), B10=(50,60), B11=(50,80), B12=(50,100)。
  交汇室压力涌升 → 单发（single-shoot）geyser。
- **Series C**（12 组）：下游满管（尾门控制）+ 上游管**人为封气囊** + 竖管初始水柱 hr0。
  Q0 = 15/20/25/30 L/s，Q1 = 40 L/s（固定）；hr0 = 0.1/0.2/0.3 m。
  C1=(15,0.1), C2=(15,0.2), C3=(15,0.3), C4=(20,0.1), ..., C9=(25,0.3), ..., C12=(30,0.3)。
  两阶段剧烈 geyser：第一阶段瞬变压力波触发（第 1、2 次喷发），
  第二阶段封闭气囊释放触发（第 4 次喷发等）。

## Key comparison targets

- Fig. 3：Series A 典型压力（PT1/PT2/PT3）——不喷发分支
- Fig. 4：Series A 竖管混合物最大高度 h
- Fig. 5：**Case B3** 流动过程照片 + 压力时程（与 A2 仅差下游条件——分支对照对）
- Fig. 6：B3/B6/B9/B12（Q1=100）PT2 压力对比
- Fig. 7(a)：喷发高度—峰压线性关系 h = 0.6943·PMax/ρg + 0.3086（R²=0.97）；
  Fig. 7(b)：触发喷发的临界 ΔQ
- Fig. 8/9：**Case C9** 快照 + 两阶段压力时程（A→G 点，多次喷发 P1m/P2m/P4m）
- Fig. 10：Series C 全部工况的峰压
- Fig. 11：h–PMax、PFinal–PMax 关系
- Eq. (7)/(8) + Fig. 13：交汇室—竖管质量振荡解析模型（周期
  T = 2π·sqrt(Ld·Ar/(g·Ad) + hr0/g)），Series C 三工况对照
- Table 2：典型工况溢出水量（3 次重复，V1/V2/V3/Vavg，重复性 ±15%）

## 与模型论文 Campaign 3 的对应

计划对齐项（论文 sec:tests:liu2020 占位所列）：
最大压力—喷发高度关系（Fig. 7a / Fig. 11）、压力振荡幅值与周期 vs 解析模型
（Eq. 7/8, Fig. 13）、溢出水量（Table 2）。
