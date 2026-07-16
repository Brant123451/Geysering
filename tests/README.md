# Experimental validation tests

Each Test contains campaign-level shared material and a `cases/` directory.
Each Case must satisfy the contract checked by `tools/validate_layout.py`:

- `README.md` and `manifest.yaml`
- `config/`
- `data/`
- `model/`
- `scripts/`
- `reference/`
- `outputs/`
- optional `openfoam/`

`_shared/` is reserved for the source paper, parameter tables, and tools used
by multiple Cases. `_archive/` contains historical implementations that are
not the current reproducibility path. Cloud Agents must never use another Case
as an undocumented runtime dependency.
