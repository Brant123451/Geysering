# frontview-3d — 本地正视渲染数据

真三维 `compressibleInterFoam`（base 网格）在 **y=0** 中面的 `alpha.water` 切割面。

- `cuttingPlane/<time>/yMid.vtp` — 130 帧（t≈0.1…13 s）
- `vtp_manifest.json` — 时刻清单
- `probes/` — 立管/羽流中心线与 PT1/PT2
- `system/` — 运行时 `controlDict` / `frontViewCut` 快照
- `config/` — 几何与阀门调度审计快照

完整说明见上级目录 [`../../HANDOFF.md`](../../HANDOFF.md)。
