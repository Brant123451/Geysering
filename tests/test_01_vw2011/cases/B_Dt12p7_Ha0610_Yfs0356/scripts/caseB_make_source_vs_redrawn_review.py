from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


CASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = CASE_ROOT / "outputs" / "source_matched_overlays"
REDRAW_DIR = CASE_ROOT / "outputs" / "vector_traced_figures"
OUT_DIR = CASE_ROOT / "outputs" / "source_vs_redrawn_review"

PRESSURE_SOURCE = SOURCE_DIR / "source_fig6_caseB_centre_panel.png"
LEVEL_SOURCE = SOURCE_DIR / "source_fig8_caseB_centre_panel.png"
PRESSURE_REDRAW = REDRAW_DIR / "caseB_vw2011_fig6_traced_pressure_1d2d.png"
LEVEL_REDRAW = REDRAW_DIR / "caseB_vw2011_fig8_traced_levels_1d2d.png"


def _show_panel(ax, path: Path, title: str, panel_letter: str) -> None:
    image = mpimg.imread(path)
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, fontsize=17, fontweight="semibold", pad=10)
    ax.text(
        0.012,
        0.975,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "black", "pad": 3.0},
    )


def _save_pair(
    source: Path,
    redraw: Path,
    quantity: str,
    filename: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.6), constrained_layout=False)
    _show_panel(
        axes[0],
        source,
        f"Published Case B panel - {quantity}",
        "(a)",
    )
    _show_panel(
        axes[1],
        redraw,
        "Independent redraw + present 1D/2D",
        "(b)",
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.88, bottom=0.105, wspace=0.055)
    fig.text(
        0.5,
        0.035,
        "The published raster is shown only for visual audit; manuscript artwork uses the independent redraw.",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#404040",
    )
    fig.savefig(OUT_DIR / filename, dpi=240, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _save_four_panel() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.4), constrained_layout=False)
    _show_panel(
        axes[0, 0],
        PRESSURE_SOURCE,
        "Published Case B pressure panel",
        "(a)",
    )
    _show_panel(
        axes[0, 1],
        PRESSURE_REDRAW,
        "Redrawn pressure + present 1D/2D",
        "(b)",
    )
    _show_panel(
        axes[1, 0],
        LEVEL_SOURCE,
        "Published Case B level panel",
        "(c)",
    )
    _show_panel(
        axes[1, 1],
        LEVEL_REDRAW,
        "Redrawn levels + present 1D/2D",
        "(d)",
    )
    fig.suptitle(
        "Case B visual audit: published panels versus independent redraws",
        fontsize=21,
        fontweight="bold",
        y=0.975,
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.915,
        bottom=0.075,
        hspace=0.15,
        wspace=0.055,
    )
    fig.text(
        0.5,
        0.025,
        "Source rasters are retained only in this audit sheet; the redrawn panels use digitized coordinates and simulation data.",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#404040",
    )
    fig.savefig(
        OUT_DIR / "caseB_vw2011_source_vs_redrawn_4panel.png",
        dpi=240,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    required = [PRESSURE_SOURCE, LEVEL_SOURCE, PRESSURE_REDRAW, LEVEL_REDRAW]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source artifacts:\n" + "\n".join(missing))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _save_pair(
        PRESSURE_SOURCE,
        PRESSURE_REDRAW,
        "pressure head",
        "caseB_vw2011_pressure_source_vs_redrawn.png",
    )
    _save_pair(
        LEVEL_SOURCE,
        LEVEL_REDRAW,
        "free surface and air-water interface",
        "caseB_vw2011_levels_source_vs_redrawn.png",
    )
    _save_four_panel()

    for path in sorted(OUT_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
