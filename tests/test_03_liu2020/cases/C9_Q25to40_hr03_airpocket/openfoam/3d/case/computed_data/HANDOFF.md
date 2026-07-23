# C9 计算结果交接（`case/computed_data`）

本目录是 **已上传到 Git 的计算结果归档**，供下一任 Cloud Agent / 本地渲染使用。

## 一句话结论

- **Phase1（至 solver 6.75）三维场 + 探针时程：已在本目录，经 Git LFS 跟踪。**
- **Phase2 现场检查点（曾跑到 ~t=8.04 / full≈36%）：因 VM 重置丢失，无法从此续跑。**
- **Phase 2 and eight eruptions have not yet been reproduced.**

## 路径

```text
tests/test_03_liu2020/cases/C9_Q25to40_hr03_airpocket/openfoam/3d/case/computed_data/
```

| 子路径 | 内容 |
|--------|------|
| `phase1_render/` | Phase1 完成时的 VTK / 重构场 / 网格 / postProcessing / 日志摘录 |
| `PROGRESS_TRACK.md` | Phase1→Phase2 进度表（最后记录到 2026-07-17T13:29Z，t≈8.0347） |
| `MANIFEST.json` | 机器可读总清单 |
| `.gitattributes` | `*.tar.xz` 等走 Git LFS |

旧路径 `openfoam/3d/artifacts/phase1_render/` 已迁移到这里；见 `../../artifacts/README.md`。

## Git

- 仓库：`https://github.com/brant123451/geysering`
- 分支：`cursor/c9-openfoam-3d-bf97`
- PR：`https://github.com/brant123451/geysering/pull/8`

克隆后必须：

```bash
git lfs install
git lfs pull
```

大文件约：`VTK.tar.xz` ~454 MB，`reconstructed_render_fields.tar.xz` ~290 MB，`mesh/polyMesh.tar.xz` ~20 MB。

## 还在 / 已丢

| 数据 | 状态 |
|------|------|
| Phase1 可用时刻体场（见 `phase1_render/MANIFEST.json`） | **已上传** |
| Phase1 全时段 `postProcessing/` 探针与通量 | **已上传** |
| Phase1 守恒门 JSON（`phase1_render/results/`） | **已上传** |
| 进度表至 t≈8.04（百分比记录） | **已上传** |
| Phase2 `processor*` / `log.full` / t≈7–8 体场 | **丢失（未进 Git，VM 重置）** |
| 八次喷发 / 论文 20 s 完整轨迹 | **尚未再现** |

## 模拟摘要

| 项 | 值 |
|----|----|
| 工况 | Liu et al. (2020) Case C9 |
| 求解器 | OpenFOAM v2512 `compressibleInterFoam` |
| 网格 | 481,874 cells |
| 并行 | 4 ranks |
| Phase1 | `1.2289420474` → `6.75`（完成） |
| Phase2 曾达 | ~`8.04` / `20.25`（full≈35.8%，p2≈9.5%），随后环境丢失 |
| 守恒门 | `bal_rel ~ 1.7e-4`（通过） |

`purgeWrite 4`：phase1 体场在约 1.23–5.98 有缺口；探针曲线仍完整。

## 本地渲染

详见 `phase1_render/RENDER_HANDOFF.md`。最短路径：

```bash
cd case/computed_data/phase1_render
mkdir -p VTK && tar -xJf VTK.tar.xz -C .
paraview VTK/case.vtm.series
```

## 以后如何避免再丢（强制约定）

运行时目录 `case/processor*`、`case/[0-9]*`、`case/log.*` 仍默认 gitignore（体积太大）。

**每个阶段结束或每推进 ~0.5–1 sim-s，必须打包上传到本目录：**

```bash
# 示例：归档最新 processor 检查点（LFS）
cd case
tar -cJf computed_data/checkpoints/processor_latest_TXXXX.tar.xz processor[0-3]/XXXX
git add computed_data/checkpoints/processor_latest_TXXXX.tar.xz
git commit -m "Archive C9 checkpoint at TXXXX"
git push
```

同时更新本目录 `MANIFEST.json` 与顶层 `HANDOFF.md`。

## 下一任 Agent 续跑

1. 安装 OpenFOAM v2512 + 依赖；`git lfs pull`。
2. **不能**从丢失的 t≈8.04 processor 续跑。
3. 从提交源码重新：`Allrun.mesh` → `Allrun.initialize` → `Allrun.resume smoke|phase1|full`。
4. 可用本目录 phase1 包做渲染/曲线对照，但 **不是** 可重启的完整求解器状态（缺完整 processor 分解场与 phase2 增量）。
5. 重跑后按上面约定把检查点写入 `computed_data/checkpoints/`。

再次强调：**Phase 2 and eight eruptions have not yet been reproduced.**
