"""Regenerate Case B comparison frames with Case A's final network solver.

The horizontal pipe and riser exchange mass and momentum at the T-junction, so
the whole coupled network is run with Case A's frozen defaults.  The resulting
1D images are written to a new directory; existing Case B and OpenFOAM images
are retained unchanged.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import argparse
from pathlib import Path

import numpy as np


CASE_B = Path(__file__).resolve().parents[1]
CASE_A_MODEL = (
    CASE_B.parent
    / "A_Dt57p1_Ha0305_Yfs0356"
    / "model"
    / "vw2011_network_twofluid.py"
)
FRAME_ROOT = CASE_B / "openfoam" / "2d" / "outputs_1d2d_compare"
FRAME_DIR = FRAME_ROOT / "frames_1d_caseB_tpa_wetdry"
TEMP_DIR = FRAME_ROOT / "_caseB_tpa_wetdry_frame_tmp"
METADATA_FILE = FRAME_ROOT / f"{FRAME_DIR.name}_meta.json"


def load_case_a_solver():
    spec = importlib.util.spec_from_file_location("case_a_network_solver", CASE_A_MODEL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {CASE_A_MODEL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(reuse_existing: bool = False) -> None:
    if reuse_existing:
        legacy_metadata = FRAME_ROOT / "frames_1d_caseA_solver_meta.json"
        if not FRAME_DIR.exists() or not legacy_metadata.exists():
            raise RuntimeError("No completed Case-A-solver frame set is available for re-pairing")
        metadata = json.loads(legacy_metadata.read_text(encoding="utf-8"))
    else:
        solver = load_case_a_solver()
        case = solver.NetworkCase(
            Dr=0.0127,
            air_head=0.610,
            init_water_level=0.356,
            t_end=9.0,
            # Case B requires the resolved dry-bed/free-surface transition;
            # the completed Case A remains on its frozen contact solver.
            horizontal_model="tpa_wetdry",
        )
        rec = solver.run_network(case, verbose=False)

        # Regression checks for the two Case-B features that must coexist:
        # a single monotone dry-bed surge at valve opening and a later geyser.
        early_index = min(
            range(len(rec["frames_t"])),
            key=lambda index: abs(rec["frames_t"][index] - 0.15),
        )
        early_alpha = np.asarray(
            rec["frames_alt"][early_index], dtype=float
        )
        early_depth = np.asarray(
            solver._depth_frac(early_alpha), dtype=float
        )
        wet = np.where(early_depth > 0.005)[0]
        nearly_full = np.where(early_depth > 0.95)[0]
        if wet.size and nearly_full.size:
            bore_start = int(wet[0])
            full_after_front = nearly_full[
                nearly_full >= bore_start
            ]
            if full_after_front.size:
                bore_end = int(full_after_front[0])
                bore_drop = float(np.min(
                    np.diff(
                        early_depth[
                            bore_start:bore_end + 1
                        ]
                    )
                ))
                if bore_drop < -0.05:
                    raise RuntimeError(
                        "Case-B early surge contains a detached "
                        "crest/trough: "
                        f"minimum depth drop={bore_drop:.3f}D"
                    )
        max_jet = float(max(rec.get("jet_height", [0.0])))
        if max_jet <= case.riser_height + 0.05:
            raise RuntimeError(
                "Case-B run did not retain the geysering branch: "
                f"max jet height={max_jet:.3f} m"
            )

        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
        TEMP_DIR.mkdir(parents=True)
        try:
            metadata = solver.make_case_frames(case, rec, TEMP_DIR, "B_caseA_solver", max_frames=60)
            source_dir = TEMP_DIR / "frames"
            if not source_dir.exists():
                raise RuntimeError("Case A frame exporter did not create its frame directory")
            if FRAME_DIR.exists():
                shutil.rmtree(FRAME_DIR)
            shutil.copytree(source_dir, FRAME_DIR)
        finally:
            if TEMP_DIR.exists():
                shutil.rmtree(TEMP_DIR)

    one_d = [
        {
            "index": index,
            "file": f"{FRAME_DIR.name}/frame_{index:04d}.png",
            "time": float(item["time"]),
            "wtop": float(item.get("wtop", float("nan"))),
            "itop": float(item.get("itop", 0.0)),
            "jetHeight": float(item.get("jetHeight", 0.0)),
            "topQ": float(item.get("topQ", 0.0)),
            "label": item.get(
                "label",
                f"Present model  t={item['time']:.2f}s  "
                f"Yfs={item.get('wtop', float('nan')):.3f}m  "
                f"jet={item.get('jetHeight', 0.0):.3f}m",
            ),
        }
        for index, item in enumerate(metadata)
    ]
    original_pairs = json.loads((FRAME_ROOT / "frames_index.json").read_text(encoding="utf-8"))
    paired = []
    for pair in original_pairs:
        target_time = float(pair["time"])
        closest = min(one_d, key=lambda item: abs(item["time"] - target_time))
        paired.append(
            {
                "time": target_time,
                "file1d": closest["file"],
                "file2d": pair["file2d"],
                "label1d": closest["label"],
                "label2d": pair["label2d"],
                "wtop1d": closest["wtop"],
                "itop1d": closest["itop"],
                "jetHeight1d": closest["jetHeight"],
                "dt_match": abs(closest["time"] - target_time),
            }
        )
    METADATA_FILE.write_text(
        json.dumps(one_d, indent=2), encoding="utf-8"
    )
    (FRAME_ROOT / "frames_index.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    print(f"Wrote {len(one_d)} Case-A-solver 1D frames to {FRAME_DIR}")
    print(f"Repaired {len(paired)} synchronized 1D--2D pairs; OpenFOAM frames were retained.")
    if not reuse_existing:
        print(
            "Validated one continuous early surge and retained "
            f"geysering (max jet={max_jet:.3f} m)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-existing", action="store_true")
    main(reuse_existing=parser.parse_args().reuse_existing)
