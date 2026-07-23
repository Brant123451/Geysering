# BH3 2-D fine-mesh 交接（base_nominal_2d）

> 状态：**RUNNING**（约 42.4%；t≈5.506/13 s；更新 2026-07-23T23:05:52Z；看门狗 20 min 自愈已启用）  
> 论文：Cong, Chan & Lee (2017) Series B **Run B-H3**  
> 路径：`tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/2d/`  
> 分支：`cursor/test2-bh3-3d-e294`  
> 运行目录：`/tmp/bh3-2d-study/base_nominal_2d`

## 1. 为何改 2-D

3-D `base_nominal` 在云环境重置后丢失体场，且墙钟过慢。2-D 细网格用于在可接受时间内得到与论文**大致可比**的时序（`Ta`、液面上升、是否 geyser）。

## 2. 与论文一致的量

| 量 | 值 |
|---|---|
| `D` | 0.050 m |
| `Dr` | 0.026 m |
| `H0` | 0.66 m（自由面 / 上游水头） |
| `L0` | 0.61 m（气囊，阀门平面 x=5.98 → 封闭端 6.59） |
| 管长 / Tee / 阀 | 6.59 / 3.47 / 5.98 m |
| 物理 rim | y=1.85 m；计算顶 y=3.0 m |
| `T0` | 296.15 K；σ=0.072；ρw≈998 |
| 入口 | 定水头 `p_rgh=107786.651 Pa`，α=1 |
| 大气顶 | `totalPressure` p0=101325；入流 α=0 |
| 阀门 | **瞬时打开**（论文 CFD baseline 0 s；实验名义 0.2 s） |
| 目标对照 | 实验 `Ta=8.18 s`，geyser |

## 3. 网格与数值

- 竖直面 2-D（`empty` front/back），~**2 mm** 细网格，**~111 435 cells**
- `compressibleInterFoam` v2512，**laminar**（加速；非 3-D kEpsilon）
- `maxCo=0.25`，`maxAlphaCo=0.15`，`maxDeltaT=5e-4`，`endTime=13`
- 4 MPI（`scotch`）

**2-D 局限（必读）：** 平面宽度比 `Dr/D=0.52` ≠ 圆管面积比 `(Dr/D)²=0.27`，不能同时保直径与面积比。结果是形态/时序诊断，不是几何精确替代。

## 4. 预计墙钟时间

短时基准（本机 4 核）：**0.05 sim-s ≈ 64 wall-s** → 约 **2.8 sim-s / 墙钟小时**。

| 目标 | 估计墙钟 |
|---|---|
| 完整 **13 s** 事件窗 | **约 5–8 小时**（若 geyser 段 Courant 收紧，上限可到 ~10–12 h） |
| 到达实验 `Ta≈8.18 s` 量级 | **约 3–5 小时** |

云环境空闲休眠会拉长墙钟；以 `log.compressibleInterFoam` 中的 `Time =` 为准。

## 5. 如何跑 / 续跑

```bash
cd tests/test_02_cong2017/cases/BH3_Dr26_H066_L061/openfoam/2d
chmod +x Allrun
OPENFOAM_NP=4 ./Allrun
```

运行产物在 `/tmp/bh3-2d-study/base_nominal_2d/`。结束后 `postprocess_2d.py` 写入 `outputs/base_nominal_2d_metrics.json` 与 timeseries。

## 6. 本地看结果

1. `git pull` 本分支  
2. 看 `outputs/base_nominal_2d_metrics.json`（`Ta_2d_s`、`geysering`）  
3. 曲线：`outputs/base_nominal_2d_timeseries.csv` 或 live FO  
4. ParaView：在运行目录对 `*.foam` / `foamToVTK`（需本地有完整时间目录）

## 7. 成功判据（大致匹配论文）

- 立管液面显著上升并接近 / 越过 rim（geyser 形态）  
- `Ta_2d` 与实验 8.18 s **同量级**（目标相对误差最好 <~30%，2-D 面积比偏差下更严苛）  
- 未要求为标签重调物性  
