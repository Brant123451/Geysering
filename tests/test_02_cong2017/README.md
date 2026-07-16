# Cong, Chan & Lee (2017) B-H1 复现

本目录用于复现 Cong, Chan & Lee (2017) 的 B-H1 工况。旧的 ODE/旧 Fortran 拼接展示代码已经移除；当前保留的是基于我们论文算法实现的 driver。

## 复现对象

- 论文：Cong, J., Chan, S. N., & Lee, J. H. W. (2017). "Geyser Formation by Release of Entrapped Air from Horizontal Pipe into Vertical Shaft." *J. Hydraul. Eng.*, 143(9), 04017039.
- 正刊原文：`paper_source/cong2017_JHE2017_offprint.pdf`
- 工况：B-H1，`D=0.05 m`，`Dr=0.016 m`，`H0=0.66 m`，`L0=0.61 m`，实验结果为 geyser。

## 当前模型

当前完整布置脚本为：

```powershell
python run_cong2017_bh1_complete_own_model.py
```

它生成水平管 + T 接口 + 竖直 riser 的完整布置结果：水平管段由论文分层/有压分支推进，竖管段用同一套两流体变量并设 `theta=90°`，在 B-H1 实测到达时间 `Ta=8.07 s` 触发底部气囊进入。

水平管单段调试脚本为：

```powershell
python run_cong2017_bh1_own_model.py
```

两个脚本都调用 `D:\tests\Research\The lase case\paper_solver_copy` 中我们论文的 Python 实现，使用四变量分层支路、IKH restoring coefficient、pressurized segment、RT-Riemann active-set closure 和 cut-cell remap。

## 输出

- `outputs/own_model_bh1_complete/index.html`：完整布置查看器，包含水平管、T 接口和竖直 riser。
- `outputs/own_model_bh1_complete/riser_profile.dat`：竖管逐帧场数据。
- `outputs/own_model_bh1_complete/summary.json`：完整布置算例参数和输出路径。
- `outputs/own_model_bh1/index.html`：水平管单段调试查看器。
- `outputs/own_model_bh1/cong2017_bh1_own_model_profile.dat`：逐帧场数据，变量为 `x, alpha_l, depth, u_l, u_g, p_g, regime`。
- `outputs/own_model_bh1/summary.json`：算例参数和输出路径。

## 说明

当前完整布置结果不再使用 ODE 或旧 Fortran 结果拼接。需要注意：T 接口采用 B-H1 实测到达时间触发竖管底部气囊进入，尚不是一个全隐式网络耦合求解器；后续若要更严格，可继续把水平管与竖管的通量、压力和气量在 T 接口处做同步耦合。
