# B-H2 二维平面 OpenFOAM（细网格）

Cong, Chan & Lee (2017) Series B / Run **B-H2** 的快速二维中心切片模型。

- 求解器：OpenFOAM v2512 `compressibleInterIsoFoam`
- 几何：论文审计尺寸（与 `../3d/PAPER_AUDIT.md` 一致）
- 网格：`blockMesh` 结构化细网格，约 **1.37×10⁵** 单元
- 阀门：`cyclicACMI` 有效面积 0→1，0.2 s
- 湍流：laminar（实验室尺度；避免壁面函数干扰喷发）

## 论文对齐参数

| 量 | 值 |
|---|---|
| D / Dr | 0.050 / 0.021 m |
| 主管长 / 三通 / 阀 | 6.59 / 3.47 / 5.98 m |
| H0 / 自由面 z / L0 | 0.66 / 0.635 / 0.61 m |
| 立管 rim | z=1.825 m |
| 入口总压头 | 107534.4665 Pa |
| 气袋初压 | 101325 Pa |
| T / ρw / σ | 296.15 K / 998 / 0.072 |

实验：geyser，`Ta=7.84 s`，`vfs=0.768`，`vint=1.022`。

## 运行

```bash
BH2_NP=4 ./Allrun smoke baseline
BH2_NP=4 ./Allrun solve baseline
python3 postprocess.py --run fine_baseline
```

## 预计墙钟时间（4 核）

细网格 ~1.37e5 单元，`maxCo=maxAlphaCo=0.25`，`endTime=13 s`：

- 乐观（平均 Δt≈1–2×10⁻⁴）：约 **3–6 小时**
- 偏保守（界面段 Δt 掉到 ~5×10⁻⁵）：约 **8–14 小时**

相对原三维 refined（~58 万单元）可缩短一个数量级以上。
