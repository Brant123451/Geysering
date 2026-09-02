#!/usr/bin/env python3
"""Compose the active Case-1 curve figures with their published source panels.

The active comparison PDF is embedded directly so that its vector curves and
text remain vector objects.  The source-paper panels are necessarily raster
images because the locally archived reference is a scanned reproduction.

This script writes review candidates only; it does not replace the figures
referenced by the manuscript.
"""
from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import fitz
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "paper" / "figures"
CASE_A = ROOT / "tests" / "test_01_vw2011" / "cases" / "A_Dt57p1_Ha0305_Yfs0356"
CASE_B = ROOT / "tests" / "test_01_vw2011" / "cases" / "B_Dt12p7_Ha0610_Yfs0356"


def _png_stream(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def _split_case_a_source() -> list[tuple[str, str, bytes, float]]:
    path = CASE_A / "data" / "digitized" / "caseA_reference_fig5_fig7_combined.png"
    with Image.open(path) as source:
        source = source.convert("RGB")
        width, height = source.size
        # The archived source montage contains the two exact experimental
        # panels separated at its midpoint.  Trim only the narrow centre gap.
        gap = max(2, int(round(width * 0.004)))
        midpoint = width // 2
        pressure = source.crop((0, 0, midpoint - gap, height))
        levels = source.crop((midpoint + gap, 0, width, height))
        return [
            ("Pressure panel", "Vasconcelos and Wright (2011), Fig. 5", _png_stream(pressure), pressure.width / pressure.height),
            ("Free-surface / interface panel", "Vasconcelos and Wright (2011), Fig. 7", _png_stream(levels), levels.width / levels.height),
        ]


def _case_b_sources() -> list[tuple[str, str, bytes, float]]:
    source_dir = CASE_B / "outputs" / "source_matched_overlays"
    items = []
    for title, citation, filename in (
        ("Pressure panel", "Vasconcelos and Wright (2011), Fig. 6", "source_fig6_caseB_centre_panel.png"),
        ("Free-surface / interface panel", "Vasconcelos and Wright (2011), Fig. 8", "source_fig8_caseB_centre_panel.png"),
    ):
        path = source_dir / filename
        with Image.open(path) as source:
            source = source.convert("RGB")
            items.append((title, citation, _png_stream(source), source.width / source.height))
    return items


def _fit_rect(box: fitz.Rect, aspect: float) -> fitz.Rect:
    """Return a centred rectangle with the requested width/height aspect."""
    box_aspect = box.width / box.height
    if aspect >= box_aspect:
        height = box.width / aspect
        y0 = box.y0 + (box.height - height) / 2.0
        return fitz.Rect(box.x0, y0, box.x1, y0 + height)
    width = box.height * aspect
    x0 = box.x0 + (box.width - width) / 2.0
    return fitz.Rect(x0, box.y0, x0 + width, box.y1)


def build_composite(
    case_label: str,
    curve_pdf: Path,
    sources: list[tuple[str, str, bytes, float]],
    output_stem: str,
) -> dict[str, object]:
    curve_document = fitz.open(curve_pdf)
    curve_page = curve_document[0]
    curve_rect = curve_page.rect

    page_width = curve_rect.width
    outer = 8.0
    section_title_height = 22.0
    source_label_height = 28.0
    inter_section_gap = 12.0
    source_gap = 8.0
    source_width = (page_width - 2.0 * outer - source_gap) / 2.0
    source_height = max(source_width / item[3] for item in sources)
    page_height = (
        outer
        + section_title_height
        + curve_rect.height
        + inter_section_gap
        + section_title_height
        + source_label_height
        + source_height
        + outer
    )

    output_document = fitz.open()
    page = output_document.new_page(width=page_width, height=page_height)
    page.insert_textbox(
        fitz.Rect(outer, outer, page_width - outer, outer + section_title_height),
        f"{case_label}: present experiment-1D-2D comparison",
        fontname="Times-Bold",
        fontsize=11.0,
        align=fitz.TEXT_ALIGN_CENTER,
        color=(0, 0, 0),
    )

    curve_target = fitz.Rect(0, outer + section_title_height, page_width, outer + section_title_height + curve_rect.height)
    page.show_pdf_page(curve_target, curve_document, 0, keep_proportion=True)

    source_header_y = curve_target.y1 + inter_section_gap
    page.insert_textbox(
        fitz.Rect(outer, source_header_y, page_width - outer, source_header_y + section_title_height),
        "Published experimental source panels",
        fontname="Times-Bold",
        fontsize=11.0,
        align=fitz.TEXT_ALIGN_CENTER,
        color=(0, 0, 0),
    )
    label_y = source_header_y + section_title_height
    image_y = label_y + source_label_height

    for index, (title, citation, image_bytes, aspect) in enumerate(sources):
        x0 = outer + index * (source_width + source_gap)
        x1 = x0 + source_width
        label_box = fitz.Rect(x0, label_y, x1, label_y + source_label_height)
        page.insert_textbox(
            label_box,
            f"{title}\n{citation}",
            fontname="Times-Roman",
            fontsize=8.0,
            lineheight=1.05,
            align=fitz.TEXT_ALIGN_CENTER,
            color=(0.08, 0.08, 0.08),
        )
        image_box = fitz.Rect(x0, image_y, x1, image_y + source_height)
        fitted = _fit_rect(image_box, aspect)
        page.insert_image(fitted, stream=image_bytes, keep_proportion=True)
        page.draw_rect(fitted, color=(0.55, 0.55, 0.55), width=0.45)

    pdf_path = FIGURES / f"{output_stem}.pdf"
    png_path = FIGURES / f"{output_stem}.png"
    output_document.set_metadata(
        {
            "title": f"{case_label} curve comparison with published source panels",
            "author": "Geysering manuscript figure workflow",
            "subject": "Review candidate; published panels reproduced with attribution",
        }
    )
    output_document.save(pdf_path, garbage=4, deflate=True)
    output_document.close()
    curve_document.close()

    rendered = fitz.open(pdf_path)
    pixmap = rendered[0].get_pixmap(matrix=fitz.Matrix(3.0, 3.0), alpha=False)
    pixmap.save(png_path)
    rendered.close()

    return {
        "case": case_label,
        "status": "review candidate; manuscript includegraphics not changed",
        "active_comparison_pdf": str(curve_pdf.relative_to(ROOT)).replace("\\", "/"),
        "published_panels": [citation for _, citation, _, _ in sources],
        "output_pdf": str(pdf_path.relative_to(ROOT)).replace("\\", "/"),
        "output_png": str(png_path.relative_to(ROOT)).replace("\\", "/"),
        "layout": "active comparison above; attributed published source panels below",
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    records = [
        build_composite(
            "Test A",
            FIGURES / "caseA_experiment_1d2d_curves.pdf",
            _split_case_a_source(),
            "caseA_experiment_1d2d_curves_with_source",
        ),
        build_composite(
            "Test B",
            FIGURES / "caseB_experiment_1d2d_curves.pdf",
            _case_b_sources(),
            "caseB_experiment_1d2d_curves_with_source",
        ),
    ]
    manifest = {
        "purpose": "Pair the active Case-1 comparison figures with the exact published experimental panels used for digitisation.",
        "source_reference": "Vasconcelos and Wright (2011)",
        "scientific_data_changed": False,
        "manuscript_references_changed": False,
        "artifacts": records,
    }
    (FIGURES / "case1_curve_source_composites_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    for record in records:
        print(ROOT / record["output_pdf"])
        print(ROOT / record["output_png"])


if __name__ == "__main__":
    main()
