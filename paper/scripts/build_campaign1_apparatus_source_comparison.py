"""Build a review composite of the Campaign 1 source and redraw.

The upper panel is cropped directly from page 2 of Vasconcelos and Wright
(2011).  The lower panel is the current manuscript redraw.  This review
artifact is not inserted into the manuscript.
"""

from __future__ import annotations

from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PDF = (
    ROOT
    / "tests"
    / "test_01_vw2011"
    / "_shared"
    / "reference"
    / "Vasconcelos_Wright_2011_JHE.pdf"
)
REDRAW = ROOT / "paper" / "figures" / "campaign1_apparatus_redrawn.png"
TMP_DIR = ROOT / "tmp" / "pdfs" / "campaign1_apparatus_compare"
RENDER_STEM = TMP_DIR / "vw2011_p2"
RENDERED_PAGE = TMP_DIR / "vw2011_p2.png"
SOURCE_CROP = TMP_DIR / "vw2011_fig2_crop.png"
OUTPUT = (
    ROOT
    / "paper"
    / "figures"
    / "campaign1_apparatus_original_vs_redrawn.png"
)

# Page 2 rendered at 360 dpi.  The box contains Fig. 2 and its published
# caption while excluding the surrounding article text and footer rule.
FIG2_CROP_BOX = (500, 2650, 2540, 3698)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    path = Path("C:/Windows/Fonts") / name
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)


def _fit_width(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            "2",
            "-l",
            "2",
            "-r",
            "360",
            "-png",
            "-singlefile",
            str(SOURCE_PDF),
            str(RENDER_STEM),
        ],
        check=True,
    )

    with Image.open(RENDERED_PAGE) as page:
        original = page.convert("RGB").crop(FIG2_CROP_BOX)
    original.save(SOURCE_CROP)
    with Image.open(REDRAW) as image:
        redraw = image.convert("RGB")

    canvas_width = 3200
    panel_width = 3000
    margin_x = (canvas_width - panel_width) // 2
    title_height = 92
    outer_margin = 84
    panel_gap = 118
    original = _fit_width(original, panel_width)
    redraw = _fit_width(redraw, panel_width)
    canvas_height = (
        outer_margin
        + title_height
        + original.height
        + panel_gap
        + title_height
        + redraw.height
        + outer_margin
    )

    canvas = Image.new("RGB", (canvas_width, canvas_height), "#f5f5f5")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(48, bold=True)
    border = "#b8b8b8"
    text = "#202020"

    y = outer_margin
    draw.text(
        (margin_x, y),
        "(a) Original: Vasconcelos and Wright (2011), Fig. 2",
        font=title_font,
        fill=text,
        anchor="la",
    )
    y += title_height
    canvas.paste(original, (margin_x, y))
    draw.rectangle(
        (margin_x - 1, y - 1, margin_x + panel_width, y + original.height),
        outline=border,
        width=2,
    )

    y += original.height + panel_gap
    draw.line((margin_x, y - panel_gap // 2, margin_x + panel_width, y - panel_gap // 2), fill="#8a8a8a", width=3)
    draw.text(
        (margin_x, y),
        "(b) Strict vector redraw used in the present manuscript",
        font=title_font,
        fill=text,
        anchor="la",
    )
    y += title_height
    canvas.paste(redraw, (margin_x, y))
    draw.rectangle(
        (margin_x - 1, y - 1, margin_x + panel_width, y + redraw.height),
        outline=border,
        width=2,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, dpi=(300, 300), optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
