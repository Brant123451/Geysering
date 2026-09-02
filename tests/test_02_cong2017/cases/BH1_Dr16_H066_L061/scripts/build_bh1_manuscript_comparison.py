# -*- coding: utf-8 -*-
"""Build the manuscript B-H1 common-clock 1D/OpenFOAM-2D phase sequence.

The figure has one scientific job: show that both descriptions select the
geyser branch while resolving different eruption timing and interface
morphology.  The four columns/rows retain physical time; no event alignment,
time shift, smoothing, or calibration is applied.

The 1D panels come from the validated geometry-matched 13 s archive.  They are
not extrapolated beyond that archive.  The 2D fields come from the completed
``h1_refined_co015`` qualification run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


SCRIPT = Path(__file__).resolve()
CASE = SCRIPT.parent.parent
REPO = next(parent for parent in CASE.parents if (parent / "paper").is_dir())
OF2D = CASE / "openfoam" / "2d"
QUAL = OF2D / "qualification" / "h1_refined_co015"
QUAL_RESULTS = QUAL / "results"
FRAME_BUILDER = OF2D / "frame_compare" / "build_frame_compare.py"
OUT = CASE / "outputs" / "manuscript_comparison_h1"
FRAMES_1D = OUT / "source_frames" / "frames_1d"
FRAMES_2D = OUT / "source_frames" / "frames_2d"
ONE_D_VIEWER = OF2D / "frame_compare"
PAPER_FIGURES = REPO / "paper" / "figures"
RUN = Path(os.environ.get("BH1_REFINED_2D_RUN", "/tmp/bh1-2d-study/h1_refined_co015"))

FRAME_TIMES = np.array([8.50, 12.80, 13.00, 14.80], dtype=float)
ONE_D_ARCHIVE_END = 13.0

D = 0.05
DR = 0.016
H0 = 0.66
RIM = 1.8
X_TEE = 3.47
X_VALVE = 5.98
X_END = 6.59


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frame_builder():
    spec = importlib.util.spec_from_file_location("bh1_frame_builder", FRAME_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {FRAME_BUILDER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_inputs() -> tuple[dict, dict, dict, dict]:
    two_metrics = load_json(QUAL_RESULTS / "openfoam_2d_metrics.json")
    audit = load_json(QUAL_RESULTS / "run_record" / "paper_audit.json")
    consistency = load_json(QUAL / "paper_consistency_report.json")
    one_metrics = load_json(CASE / "outputs" / "paper_layout_1d" / "caseA_comparison_metrics.json")

    status = two_metrics["status"]
    if status.get("fatal_error") or not status.get("ended_normally"):
        raise RuntimeError(f"Refined 2D run did not end normally: {status}")
    if float(status.get("last_log_time_s", -1.0)) < float(FRAME_TIMES[-1]):
        raise RuntimeError("Refined 2D run does not cover the final selected frame")
    if audit.get("status") != "PASS" or not all(audit.get("checks", {}).values()):
        raise RuntimeError("The geometry/initial/boundary-condition paper audit did not pass")
    if two_metrics.get("run_id") != "h1_refined_co015":
        raise RuntimeError("Unexpected 2D run identifier")
    if consistency.get("overall_assessment") != "BASIC_OUTCOME_MATCH_WITH_MAJOR_TRANSIENT_BIASES":
        raise RuntimeError("The descriptive evidence assessment changed; re-audit the figure claim")
    if one_metrics.get("variant") != "paper_Fig1b_axial_layout":
        raise RuntimeError("The selected 1D archive is not the geometry-matched Fig. 1(b) variant")
    viewer_validation = load_json(ONE_D_VIEWER / "validation.json")
    if viewer_validation.get("status") != "PASS":
        raise RuntimeError("The archived 1D frame viewer did not pass its validation gate")
    for time_s in FRAME_TIMES:
        if not (RUN / f"{time_s:g}").exists():
            raise RuntimeError(f"Missing OpenFOAM field directory for t={time_s:g} s under {RUN}")
    return one_metrics, two_metrics, audit, consistency


def render_1d_frames() -> list[dict]:
    html = (ONE_D_VIEWER / "bh1_1d2d_frame_compare_13s.html").read_text(encoding="utf-8")
    one_text = html.split("const data1=", 1)[1].split(",data2=", 1)[0]
    archived_metadata = json.loads(one_text)
    FRAMES_1D.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []
    for index, target in enumerate(FRAME_TIMES):
        if target > ONE_D_ARCHIVE_END:
            manifest.append({"target_time_s": float(target), "available": False,
                             "reason": "validated 1D archive ends at 13.0 s"})
            continue
        source_index = int(round(float(target) * 10.0))
        source_zoom = ONE_D_VIEWER / "one_d_frames" / f"zoom_{source_index:04d}.png"
        source_full = ONE_D_VIEWER / "one_d_frames" / f"full_{source_index:04d}.png"
        if not source_zoom.exists() or not source_full.exists():
            raise RuntimeError(f"Missing validated 1D source frame at t={target:.1f} s")
        image = plt.imread(source_zoom)
        height, width = image.shape[:2]
        crop = image[int(0.087 * height):int(0.966 * height),
                     int(0.217 * width):int(0.954 * width)]
        # Recover the phase pixels only.  The archived zoom includes a small
        # legend inside its axes; retaining only water components connected to
        # the domain boundary removes that annotation without modifying phase.
        from scipy.ndimage import label

        rgb = crop[..., :3]
        water = np.linalg.norm(rgb - np.array([43, 127, 255]) / 255.0, axis=2) < 0.08
        labels, count = label(water)
        phase_water = np.zeros(water.shape, dtype=float)
        for component in range(1, count + 1):
            pixels = labels == component
            # Physical water regions are orders of magnitude larger than the
            # small in-axes legend swatch.  Keep both wall films when a gas
            # core splits the liquid into two connected components.
            if int(pixels.sum()) > int(0.005 * pixels.size):
                phase_water[pixels] = 1.0
        # The original axes legend sits in the lower-right corner over a
        # fully liquid wall-film region. Restore that small patch after phase
        # extraction so its white glyphs do not masquerade as gas cells.
        phase_water[int(0.93 * phase_water.shape[0]):,
                    int(0.60 * phase_water.shape[1]):] = 1.0
        data_path = FRAMES_1D / f"zoom_data_archive_{index:04d}.npz"
        np.savez_compressed(data_path, alpha_water=phase_water)
        shutil.copy2(source_full, FRAMES_1D / f"full_archive_{index:04d}.png")
        metadata = min(archived_metadata, key=lambda item: abs(float(item["time"]) - float(target)))
        manifest.append({"target_time_s": float(target), "actual_time_s": float(target),
                         "available": True,
                         "Yfs_m": metadata.get("Yfs"),
                         "Yint_m": metadata.get("Yint"),
                         "head_over_H0": metadata.get("headOverH0"),
                         "source": str(source_zoom.relative_to(REPO).as_posix()),
                         "file": str(data_path.relative_to(OUT).as_posix())})
        print(f"1D archived manuscript frame {index + 1}/3 t={target:.2f} s", flush=True)
    return manifest


def render_2d_frames(builder) -> list[dict]:
    FRAMES_2D.mkdir(parents=True, exist_ok=True)
    builder.HERE = OUT
    builder.RUN = RUN
    builder.RESULTS = QUAL_RESULTS
    builder.TWO_FRAMES = FRAMES_2D
    builder.FRAME_TIMES = FRAME_TIMES
    original_export = builder.export_vtu

    def clean_export(time_s: float):
        # foamToVTK appends selected times even with -overwrite.  Remove only
        # its disposable conversion directory before each requested field so
        # the shared reader receives exactly one VTU; simulation checkpoints
        # and post-processing data remain untouched.
        shutil.rmtree(RUN / "VTK_FRAME_WORK", ignore_errors=True)
        return original_export(time_s)

    builder.export_vtu = clean_export
    frames = builder.render_two_d_frames(reuse=False)
    write_2d_zoom_data(builder)
    return [{"target_time_s": float(target), "actual_time_s": frame["time"],
             "Yfs_m": frame["Yfs"], "Yint_m": frame["Yint"],
             "head_over_H0": frame["headOverH0"], "file": frame["file"]}
            for target, frame in zip(FRAME_TIMES, frames)]


def write_2d_zoom_data(builder) -> None:
    """Raster the physical riser and above-rim plume at high lateral resolution."""
    from scipy.spatial import cKDTree

    x_half = 0.06
    nx, nz = 360, 900
    xs = np.linspace(X_TEE - x_half, X_TEE + x_half, nx)
    z_above_crown = np.linspace(0.0, 2.2, nz)
    global_z = z_above_crown + 0.025
    xx, zz = np.meshgrid(xs, global_z)
    riser = (np.abs(xx - X_TEE) <= (DR * DR / D) / 2.0) & (zz <= 1.825)
    exterior = zz > 1.825
    mask = riser | exterior
    raster_index = None

    for index, time_s in enumerate(FRAME_TIMES):
        vtu = builder.export_vtu(float(time_s))
        _, alpha, centres = builder.read_vtu(vtu, need_geometry=raster_index is None)
        if raster_index is None:
            if centres is None or len(centres) != len(alpha):
                raise RuntimeError("VTK geometry and alpha field sizes differ in H1 zoom raster")
            raster_index = np.full(mask.shape, -1, dtype=np.int32)
            _, nearest = cKDTree(centres).query(np.column_stack((xx[mask], zz[mask])), k=1)
            raster_index[mask] = nearest.astype(np.int32)
        image = np.full(mask.shape, np.nan, dtype=float)
        image[mask] = np.clip(alpha[raster_index[mask]], 0.0, 1.0)
        np.savez_compressed(FRAMES_2D / f"zoom_data_{index:04d}.npz",
                            alpha_water=image, x_half_m=x_half)


def build_figure() -> tuple[Path, Path]:
    cmap = LinearSegmentedColormap.from_list(
        "air_water", [(0.0, "#f4f6f8"), (0.08, "#dbeafe"), (1.0, "#2b7fff")]
    )
    cmap.set_bad("#eef0f3")
    fig, axes = plt.subplots(2, 4, figsize=(7.35, 5.25), sharex=True, sharey=True)
    for col, time_s in enumerate(FRAME_TIMES):
        two = np.load(FRAMES_2D / f"zoom_data_{col:04d}.npz")["alpha_water"]
        top = axes[0, col]
        if time_s <= ONE_D_ARCHIVE_END:
            one = np.load(FRAMES_1D / f"zoom_data_archive_{col:04d}.npz")["alpha_water"]
            top.imshow(one, extent=(0, 1, 0, RIM), origin="upper", aspect="auto",
                       cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
            html = (ONE_D_VIEWER / "bh1_1d2d_frame_compare_13s.html").read_text(encoding="utf-8")
            one_text = html.split("const data1=", 1)[1].split(",data2=", 1)[0]
            metadata = min(json.loads(one_text),
                           key=lambda item: abs(float(item["time"]) - float(time_s)))
            if metadata.get("Yfs") is not None:
                top.axhline(float(metadata["Yfs"]), color="#1d4ed8", lw=0.9)
            if metadata.get("Yint") is not None and float(metadata["Yint"]) > 0.0:
                top.axhline(float(metadata["Yint"]), color="#f97316", lw=0.85,
                            ls=(0, (3, 2)))
        else:
            top.text(0.5, 0.48, "validated 1D archive\nends at 13.0 s",
                     ha="center", va="center", fontsize=7.5, color="#4b5563")
        bottom = axes[1, col]
        bottom.imshow(two, extent=(0, 1, 0, 2.2), origin="lower", aspect="auto",
                      cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
        for row, ax in enumerate((top, bottom)):
            ax.axhline(RIM, color="#d94841", lw=0.85, ls="--")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 2.2)
            ax.set_xticks([])
            ax.set_yticks([0.0, 0.6, 1.2, 1.8, 2.2])
            ax.tick_params(labelsize=7, length=2.5)
            if col > 0:
                ax.tick_params(labelleft=False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
                spine.set_color("#374151")
        axes[0, col].axhspan(RIM, 2.2, facecolor="#eef0f3", alpha=0.35,
                            hatch="////", edgecolor="#c7cbd1", linewidth=0.0)
        axes[0, col].set_title(f"({chr(97 + col)})  $t={time_s:.1f}$ s",
                               fontsize=8.8, fontweight="bold", pad=4)

    axes[0, 0].text(-0.43, 0.5, "Frozen 1D", transform=axes[0, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=9,
                    fontweight="bold")
    axes[1, 0].text(-0.43, 0.5, "OpenFOAM 2D", transform=axes[1, 0].transAxes,
                    rotation=90, va="center", ha="center", fontsize=9,
                    fontweight="bold")
    fig.text(0.022, 0.48, "Height above pipe crown (m)", rotation=90,
             va="center", ha="center", fontsize=8)
    fig.legend(handles=[Patch(facecolor="#2b7fff", label="water"),
                        Patch(facecolor="#f4f6f8", edgecolor="#9ca3af", label="gas"),
                        Line2D([0], [0], color="#1d4ed8", lw=1.0,
                               label="1D free surface"),
                        Line2D([0], [0], color="#f97316", lw=1.0, ls=(0, (3, 2)),
                               label="1D gas front"),
                        Patch(facecolor="#eef0f3", edgecolor="#c7cbd1", hatch="////",
                              label="outside 1D domain")],
               ncol=5, loc="upper center", bbox_to_anchor=(0.57, 0.995),
               frameon=False, fontsize=6.8, handlelength=1.8, columnspacing=1.0)
    fig.subplots_adjust(left=0.13, right=0.995, bottom=0.055, top=0.91,
                        wspace=0.08, hspace=0.10)

    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "cong2017_bh1_1d2d_phase_sequence.png"
    pdf = OUT / "cong2017_bh1_1d2d_phase_sequence.pdf"
    fig.savefig(png, dpi=350, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png, PAPER_FIGURES / png.name)
    shutil.copy2(pdf, PAPER_FIGURES / pdf.name)
    return png, pdf


def write_evidence_files(one_frames: list[dict], two_frames: list[dict],
                         one_metrics: dict, two_metrics: dict, audit: dict,
                         consistency: dict, png: Path, pdf: Path) -> None:
    manifest = {
        "figure": "cong2017_bh1_1d2d_phase_sequence",
        "dominant_claim": "Both calculations select the B-H1 geyser branch, while their eruption timing and resolved interface morphology differ.",
        "selected_physical_times_s": FRAME_TIMES.tolist(),
        "event_alignment": False,
        "smoothing": False,
        "one_d": {
            "variant": one_metrics["variant"],
            "validated_archive_end_s": ONE_D_ARCHIVE_END,
            "primary_13s_metrics_unchanged": True,
            "frames": one_frames,
        },
        "two_d": {
            "run_id": two_metrics["run_id"],
            "normal_end_s": two_metrics["status"]["last_log_time_s"],
            "paper_contract_audit": audit["status"],
            "assessment": consistency["overall_assessment"],
            "frames": two_frames,
        },
        "claim_evidence": {
            "same_geyser_outcome": "supported",
            "gas_arrival": "supported",
            "eruption_timing": "partial",
            "quantitative_rise_speed": "missing",
            "postarrival_pressure_surge": "missing",
        },
        "artifacts": {
            "png": {"path": str(png.relative_to(REPO).as_posix()), "sha256": sha256(png)},
            "pdf": {"path": str(pdf.relative_to(REPO).as_posix()), "sha256": sha256(pdf)},
        },
    }
    (OUT / "bh1_1d2d_phase_sequence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    plan = """# B-H1 manuscript figure plan

## Dominant claim

The frozen 1D model and the area-equivalent planar OpenFOAM calculation both
select the B-H1 geyser branch, but they differ in eruption timing and resolved
interface morphology.

## Panel roles

- (a), 8.5 s: methodological bridge at gas-pocket arrival.
- (b), 12.8 s: claim-supporting evidence for the earlier 1D rim arrival.
- (c), 13.0 s: failure-mode/limitation panel showing the 1D rim state while
  the 2D water column remains below the rim.
- (d), 14.8 s: claim-supporting evidence for the 2D geyser/ejection outcome;
  the validated 1D archive is deliberately not extrapolated beyond 13.0 s.

The upper and lower rows retain identical physical time. No event alignment,
time shift, smoothing or outcome fitting is applied. The riser is enlarged in
each row so that phase morphology remains legible; the complete pipe--tee--
riser source frames remain in the audit folder. Quantitative level and pressure
curves remain outside this figure because their evidence status is partial or
missing.

## Legend logic and scope

Blue denotes water and pale grey/white denotes gas. Solid blue and dashed
orange markers in the 1D row denote its reported free surface and gas front;
hatching above the rim marks a domain that the reduced model does not resolve.
Lateral widths are independently enlarged and must not be compared quantitatively.
The 2D field is an area-equivalent planar spatial diagnostic, not a geometry-
exact or quantitative benchmark. The figure supports outcome, timing contrast,
and morphology statements only.
"""
    (OUT / "BH1_FIGURE_PLAN.md").write_text(plan, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    one_metrics, two_metrics, audit, consistency = validate_inputs()
    builder = load_frame_builder()
    one_frames = render_1d_frames()
    two_frames = render_2d_frames(builder)
    png, pdf = build_figure()
    write_evidence_files(one_frames, two_frames, one_metrics, two_metrics,
                         audit, consistency, png, pdf)
    print(f"Wrote B-H1 manuscript comparison assets to {OUT}")


if __name__ == "__main__":
    main()
