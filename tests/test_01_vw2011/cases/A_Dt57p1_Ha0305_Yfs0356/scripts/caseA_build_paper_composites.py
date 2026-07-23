"""Build publication composites for the two Case-A comparisons missing in the paper."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
PAPER_FIGURES = HERE.parents[1] / "paper" / "figures"


def read_rgb(path: Path):
    image = mpimg.imread(path)
    return image[..., :3] if image.ndim == 3 else image


def build_fig4() -> None:
    photo = read_rgb(HERE / "paper_scans" / "fig4_photo_strip.png")
    model = read_rgb(OUT / "caseA_fig4_snapshots.png")

    # The source crop contains the first line of the paper caption at the bottom.
    photo = photo[: int(photo.shape[0] * 0.945)]

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.0))
    for ax, image, title in (
        (axes[0], photo, "(a) Experiment: Vasconcelos and Wright (2011), Fig. 4"),
        (axes[1], model, "(b) Present two-fluid model at the same 0.14 s spacing"),
    ):
        ax.imshow(image)
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")
    fig.tight_layout(pad=0.8)
    for suffix in ("png", "pdf"):
        target = OUT / f"caseA_fig4_experiment_model.{suffix}"
        fig.savefig(target, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            PAPER_FIGURES / target.name,
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


def build_fig10() -> None:
    published = read_rgb(HERE / "paper_scans" / "fig10_full.png")
    model = read_rgb(OUT / "caseA_fig10_model_panels.png")

    # Remove the partial source caption while retaining all five published panels.
    published = published[: int(published.shape[0] * 0.965)]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.4))
    for ax, image, title in (
        (
            axes[0],
            published,
            "(a) Published experiment and TPA model\n(Vasconcelos and Wright, 2011, Fig. 10)",
        ),
        (axes[1], model, "(b) Present model on the same five axes (no time shift)"),
    ):
        ax.imshow(image)
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")
    fig.tight_layout(pad=0.7)
    for suffix in ("png", "pdf"):
        target = OUT / f"caseA_fig10_full_comparison.{suffix}"
        fig.savefig(target, dpi=300 if suffix == "png" else None, bbox_inches="tight")
        PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            PAPER_FIGURES / target.name,
            dpi=300 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    build_fig4()
    build_fig10()
    print(OUT / "caseA_fig4_experiment_model.pdf")
    print(OUT / "caseA_fig10_full_comparison.pdf")
