# Case B 3D simulation data archives

Curated archives under this folder are tracked in git (large `*.tar.xz` via LFS).
Runtime trees under `openfoam/3d/` remain gitignored.

| Archive | Status | Path | Start here |
|--------|--------|------|------------|
| maxCo=0.10, 0.12 s closed-valve rim-onset | **rejected** (rim gas hotspot) | `maxco010_0p12_screen/` | `RENDER_HANDOFF.md` |
| refined mesh, Relocate optimizer | **mesh gate failed**; no solver | `refined_mesh_relocate/` | `MESH_HANDOFF.md` |

Case-level Chinese handoff for local AI continuation:

`../LOCAL_HANDOFF.md`

```bash
git lfs pull
```
