# C9 OpenFOAM **2D** fine-mesh case (Liu et al. 2020)

中心线平面（x–z）细网格替代 3D，用于在有限墙钟时间内对齐论文相位/喷发行为。

## 与论文一致性

| 项 | 论文 | 本 2D 模型 |
|----|------|-----------|
| 上游管 | 5.80 m，D=0.20 m，坡 1:100 | 长度/直径一致；**坡度展平为 invert z=0.18**（共形网格） |
| 跌水 | 0.18 m | 一致 |
| 腔室 | 0.30×0.45 m（长×高） | 一致（宽度折叠） |
| 下游管 | 5.95 m，D=0.28 m | 一致 |
| 竖管 | d=0.06 m，L=1.22 m，hr0=0.30 m | 一致 |
| Q | 25→40 L/s，0.40 s | 平均流速匹配；2D 体积流量按厚度缩放 |
| 尾门 | 等效开度 | 高度按 A_eff/A_pipe×D |
| EOS / 湍流 | 305 m/s 管波速，`kOmegaSST` | 与 3D 同源 |

## 网格

- ~**217,000** cells（细网格，管段约 4 mm）
- `MESH_META.json` / `prepare_case_2d.py`

## 运行

```bash
source /usr/lib/openfoam/openfoam2512/etc/bashrc
cd case
./Allrun.mesh          # 若尚无 polyMesh
./Allrun.initialize    # → 0.25
./Allrun.resume smoke  # → 1.25
./Allrun.resume phase1 # → 6.75
./Allrun.resume full   # → 20.25
```

## 墙钟时间估算（4 核）

初始化末段实测约 **2.9 sim-s / 墙钟小时**。  
到 `endTime=20.25` 约剩 **20 sim-s** → 粗估 **约 7–12 墙钟小时**（喷发阶段若 Δt 变小可到 **12–20 h**）。  
对比原 3D（~481k，~0.12 sim-s/h）约快 **1–2 个数量级**。

## 状态

- 初始化已完成（solver 0.25，单连通 Mesh OK）
- **Phase 2 and eight eruptions have not yet been reproduced.**（全时段仍在跑）

详见 `HANDOFF.md`。
