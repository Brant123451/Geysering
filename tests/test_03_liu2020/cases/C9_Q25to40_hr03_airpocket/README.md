# Case C — Liu, Shao & Zhu (2020) Case C9（上游封气囊，两阶段剧烈喷发）

论文 Campaign 3 的核心工况：Series C 的 **C9**——论文详细分析工况
（Fig. 8 快照 + Fig. 9 两阶段压力分解）。

## 工况参数

| item | value |
|---|---:|
| 初始流量 Q0 | 25 L/s |
| 终了流量 Q1 | 40 L/s（Series C 固定） |
| 竖管初始水柱 hr0 | 0.3 m |
| 下游初始条件 | 满管（尾门控制） |
| 上游初始条件 | 满管 + 封闭气囊（试验）；模型含全水/封气两变体 |
| 试验结果 | 两阶段 geyser：phase-1 瞬变驱动（第 1–2 次）；phase-2 气囊抵达后（第 3–8 次，t≈6.46 s） |

## 本轮完成范围（phase-1）

- ✅ 求解器：`model/liu2020_network_twofluid.py`（自 B3 扩展 `series_c`、尾门 `A_gate`、封气楔）
- ✅ 数字化 Fig.9 + 三方对比图 `outputs/caseC_phase1_threeway.png`
- ✅ 全记录压力 `caseC_comparison_pressure.png`、竖管柱 `caseC_riser_column.png`
- ✅ 指标 `caseC_metrics.json`、`report.html`
- ⬜ phase-2（气囊抵达交汇室后的喷发 3–8）：需完整 RTRP/气核输运，声明为范围外
- ⬜ Fig.8 快照定性对照、Table 2 溢出体积定量对标（可选）

## 关键结果（全水变体，Eq.7 假设集）

| 量 | 试验 | 模型 |
|---|---:|---:|
| P1m @ t | 10.69 kPa @ 0.50 s | 11.5 kPa @ 0.54 s |
| 水面到顶 | 0.73 s | 0.93 s |
| 振荡周期 | 1.45 s | 1.13 s（解析 Eq.7：1.52 s） |
| PT2/3/4 终值 | 8.79 / 12.76 / 9.25 kPa | 10.6 / 15.4 / 11.1 kPa |

封气变体：首峰缓冲定性一致，~4 s 后降阶气楔模型漂移（见 report）。

## 运行

```bash
python scripts/caseC_digitize_and_compare.py
```
