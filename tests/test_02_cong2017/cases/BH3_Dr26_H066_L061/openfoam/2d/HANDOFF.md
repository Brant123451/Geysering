# BH3 2-D fine-mesh 交接（base_nominal_2d）

> 状态：**COMPLETED**（t≈13.0/13 s；更新 2026-07-24T08:02:10Z） Ta_2d=None geyser=False  
> 论文：Cong, Chan & Lee (2017) Series B **Run B-H3**  
> 路径：`tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/2d/`  
> 分支：`cursor/test2-bh3-3d-e294`  
> 运行目录：`/tmp/bh3-2d-study/base_nominal_2d`（云端临时）

## 1. 最终判决

**13 s 事件窗已算完。** 看门狗（20 min 巡检 + 自愈）盯到结束。

与论文对照（需诚实）：

| 量 | 2-D 结果 | 实验 / 论文 |
|---|---|---|
| 事件窗 | **13 s 完成** | 13 s |
| `geysering` | **false**（液面未到 rim 1.85 m） | true |
| `Ta_2d` | **null**（未达 rim） | 8.18 s |
| `Yfs_max` | **≈ 0.725 m** | 喷发过 rim |
| 纯计算墙钟 | ExecutionTime ≈ **3.4 h**（12378 s） | — |
| 日历墙钟 | ClockTime ≈ **12.9 h**（含云休眠） | — |

结论：尺寸/初边值按论文搭好并跑满窗，但 **2-D 平面未能复现论文 geyser 形态**（最大自由面仅略高于 `H0=0.66 m`）。主要嫌疑是平面无法保住圆管面积比 `(Dr/D)²`，以及 laminar + 瞬时阀简化。

## 2. 与论文一致的设置（已用）

| 量 | 值 |
|---|---|
| `D` / `Dr` / `H0` / `L0` | 0.050 / 0.026 / 0.66 / 0.61 m |
| 管长 / Tee / 阀 | 6.59 / 3.47 / 5.98 m |
| rim / 计算顶 | 1.85 / 3.0 m |
| 入口 | 定水头 `p_rgh=107786.651`，α=1 |
| 大气顶 | `totalPressure` p0=101325 |
| 阀 | 瞬时打开（论文 CFD baseline） |
| 网格 | ~2 mm，**111 435** cells |
| 求解器 | `compressibleInterFoam` v2512，laminar，4 MPI |

## 3. 产物位置

| 路径 | 内容 |
|---|---|
| `outputs/base_nominal_2d_metrics.json` | 指标摘要 |
| `outputs/base_nominal_2d_timeseries.csv` | `Yfs(t)` |
| `outputs/base_nominal_2d_progress.json` | COMPLETED 进度快照 |
| `outputs/base_nominal_2d_handoff/` | FO 探针、日志尾、system 快照、MANIFEST |
| `outputs/base_nominal_2d_live/` | 同步镜像 |

**无** VTK/`processor*` 入库（体积大；场在 `/tmp`，环境重置即失）。

## 4. 本地用法

```bash
git pull
cd tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/2d
# 看曲线
python3 - <<'PY'
import csv
rows=list(csv.DictReader(open('outputs/base_nominal_2d_timeseries.csv')))
print(rows[0], rows[len(rows)//2], rows[-1])
PY
# 原始探针
ls outputs/base_nominal_2d_handoff/functionObjects/
```

## 5. 若要更接近论文 geyser

可选后续（未在本轮做）：

1. 按面积比改立管宽度 `w = D*(Dr/D)^2`（牺牲几何直径一致）  
2. 加 `kEpsilon`、阀门 0.2 s 平滑打开  
3. 回到 3-D（更慢但契约几何）

## 6. 看门狗

- tmux：`bh3-2d-watch`（INTERVAL=1200 s，死进程/FATAL/停滞自愈）  
- 已修正误报：忽略日志中的 `FOAM_SIGFPE` 启用提示  
