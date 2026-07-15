#!/usr/bin/env python3
"""Build LNCS/BDA-style standalone composite figure PNGs.

The paper LaTeX should include these finished PNGs directly. Panel labels and
subcaptions are embedded inside the image canvas, so LaTeX does not need
overpic/minipage labels.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PAPER_FIGURES = ROOT / "paper_assets" / "figures"
ENHANCED_FIGURES = ROOT / "figures_enhanced"
FINAL_FIGURES = ROOT / "figures_final"
DPI = 1200
WHITE = (255, 255, 255)
TEXT = (20, 20, 20)


@dataclass(frozen=True)
class Panel:
    filename: str
    caption: str
    aliases: tuple[str, ...] = ()


def source_dir() -> Path:
    """Prefer figures_enhanced if present, otherwise use paper_assets/figures."""
    if ENHANCED_FIGURES.exists():
        return ENHANCED_FIGURES
    return PAPER_FIGURES


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def panel_path(panel: Panel) -> Path:
    src = source_dir()
    names = (panel.filename, *panel.aliases)
    for name in names:
        path = src / name
        if path.exists():
            return path
    tried = ", ".join(str(src / name) for name in names)
    raise FileNotFoundError(f"Missing panel image. Tried: {tried}")


def contain_size(src_w: int, src_h: int, box_w: int, box_h: int) -> tuple[int, int]:
    scale = min(box_w / src_w, box_h / src_h)
    return max(1, int(src_w * scale)), max(1, int(src_h * scale))


def trim_outer_white(image: Image.Image, *, tolerance: int = 248, padding: int = 28) -> Image.Image:
    """Trim inherited white margins while keeping a small protective padding."""
    rgb = image.convert("RGB")
    # In the mask, non-white pixels become 255 and near-white background becomes 0.
    mask = rgb.convert("L").point(lambda value: 255 if value < tolerance else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return rgb
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(rgb.width, right + padding)
    bottom = min(rgb.height, bottom + padding)
    return rgb.crop((left, top, right, bottom))


def paste_panel(
    canvas: Image.Image,
    panel: Panel,
    box: tuple[int, int, int, int],
    caption_y: int,
    font: ImageFont.ImageFont,
) -> None:
    x, y, w, h = box
    with Image.open(panel_path(panel)) as src:
        image = trim_outer_white(src.convert("RGB"))
        new_w, new_h = contain_size(image.width, image.height, w, h)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    paste_x = x + (w - new_w) // 2
    paste_y = y + (h - new_h) // 2
    canvas.paste(image, (paste_x, paste_y))

    draw = ImageDraw.Draw(canvas)
    label_bbox = draw.textbbox((0, 0), panel.caption, font=font)
    label_w = label_bbox[2] - label_bbox[0]
    draw.text((x + (w - label_w) // 2, caption_y), panel.caption, fill=TEXT, font=font)


def make_figure_1() -> Image.Image:
    canvas = Image.new("RGB", (6000, 2250), WHITE)
    font = load_font(68)
    margin_x = 70
    top = 60
    gutter = 70
    caption_gap = 30
    caption_h = 95
    bottom = 40
    panel_w = (canvas.width - 2 * margin_x - gutter) // 2
    panel_h = canvas.height - top - caption_gap - caption_h - bottom
    panels = [
        Panel("figure_15_good_vs_bad_prescription_examples.png", "(a) Visual-quality examples"),
        Panel("figure_02_dataset_composition.png", "(b) Dataset composition"),
    ]
    for idx, panel in enumerate(panels):
        x = margin_x + idx * (panel_w + gutter)
        paste_panel(canvas, panel, (x, top, panel_w, panel_h), top + panel_h + caption_gap, font)
    return canvas


def make_figure_2() -> Image.Image:
    canvas = Image.new("RGB", (6000, 5200), WHITE)
    font = load_font(68)
    margin_x = 150
    top = 115
    gutter = 120
    caption_gap = 46
    caption_h = 105
    row_gap = 150
    panel_w = (canvas.width - 2 * margin_x - gutter) // 2
    top_panel_h = 1750
    bottom_panel_h = canvas.height - top - top_panel_h - caption_gap - caption_h - row_gap - caption_gap - caption_h - 120

    top_panels = [
        Panel("figure_05_ocr_performance.png", "(a) Raw OCR performance"),
        Panel("figure_13_benchmark_leaderboard.png", "(b) Structured extraction leaderboard"),
    ]
    for idx, panel in enumerate(top_panels):
        x = margin_x + idx * (panel_w + gutter)
        paste_panel(canvas, panel, (x, top, panel_w, top_panel_h), top + top_panel_h + caption_gap, font)

    bottom_y = top + top_panel_h + caption_gap + caption_h + row_gap
    bottom_panel = Panel("figure_10_models_x_metrics_heatmap.png", "(c) Cross-metric comparison")
    paste_panel(
        canvas,
        bottom_panel,
        (margin_x, bottom_y, canvas.width - 2 * margin_x, bottom_panel_h),
        bottom_y + bottom_panel_h + caption_gap,
        font,
    )
    return canvas


def make_figure_3() -> Image.Image:
    canvas = Image.new("RGB", (6000, 5200), WHITE)
    font = load_font(66)
    margin_x = 150
    top = 115
    gutter_x = 120
    gutter_y = 150
    caption_gap = 42
    caption_h = 100
    panel_w = (canvas.width - 2 * margin_x - gutter_x) // 2
    panel_h = (canvas.height - top - 120 - gutter_y - 2 * (caption_gap + caption_h)) // 2
    panels = [
        Panel("figure_08_department_model_heatmap.png", "(a) Department-wise scores"),
        Panel(
            "figure_09_runtime_vs_accuracy.png",
            "(b) Runtime versus accuracy",
            aliases=("figure_09_runtimevs_accuracy.png",),
        ),
        Panel("figure_11_bootstrap_confidence_intervals.png", "(c) Bootstrap confidence intervals"),
        Panel("figure_12_wilcoxon_holm.png", "(d) Wilcoxon-Holm comparisons"),
    ]
    for idx, panel in enumerate(panels):
        row = idx // 2
        col = idx % 2
        x = margin_x + col * (panel_w + gutter_x)
        y = top + row * (panel_h + caption_gap + caption_h + gutter_y)
        paste_panel(canvas, panel, (x, y, panel_w, panel_h), y + panel_h + caption_gap, font)
    return canvas


def regenerate_panels() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import generate_publication_assets as assets  # noqa: PLC0415

    assets.ensure_dirs()
    assets.generate_figures()


def save(canvas: Image.Image, filename: str) -> Path:
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    path = FINAL_FIGURES / filename
    canvas.save(path, dpi=(DPI, DPI), optimize=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regenerate-panels",
        action="store_true",
        help="Regenerate individual source panels before composing final figures.",
    )
    args = parser.parse_args()

    if args.regenerate_panels:
        regenerate_panels()

    outputs = [
        save(make_figure_1(), "figure_1_dataset_overview.png"),
        save(make_figure_2(), "figure_2_main_results.png"),
        save(make_figure_3(), "figure_3_extended_analysis.png"),
    ]

    print(f"Source panel directory: {source_dir()}")
    for path in outputs:
        with Image.open(path) as image:
            print(f"{path}: {image.width}x{image.height} px, {DPI} dpi")


if __name__ == "__main__":
    main()
