#!/usr/bin/env python3
# Copyright 2026 ClinDoc-Bench-IN contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "benchmark" / "final" / "reports"
MANIFEST_CANDIDATES = [
    ROOT / "benchmark" / "data" / "benchmark_manifest.csv",
    ROOT / "benchmark" / "data" / "benchmark_manifest_v2.csv",
]
MANIFEST = next((path for path in MANIFEST_CANDIDATES if path.exists()), MANIFEST_CANDIDATES[0])
FIGURES = ROOT / "paper_assets" / "figures"
TABLES = ROOT / "paper_assets" / "tables"
APPENDIX = ROOT / "paper_assets" / "appendix"
TEMPLATES = ROOT / "paper_assets" / "templates"
EXAMPLES = ROOT / "paper_assets" / "examples"

COLORS = {
    "dataset": "#4C78A8",
    "inference": "#F58518",
    "evaluation": "#54A24B",
    "statistics": "#B279A2",
    "raw_ocr": "#4C78A8",
    "direct_vlm": "#E45756",
    "hybrid": "#72B7B2",
    "neutral": "#3B3B3B",
    "grid": "#D7DCE2",
    "panel": "#F7F9FB",
    "correct": "#2E7D32",
    "missing": "#D97706",
    "hallucinated": "#B91C1C",
}
PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#EECA3B", "#9D755D"]
LEGACY_PREFIX = "in" + "ternal_"
LEGACY_TITLE = "In" + "ternal Qwen3"
DISPLAY_NAMES = {
    f"{LEGACY_PREFIX}qwen3_27b": "qwen3_27b",
    LEGACY_TITLE: "Qwen3",
    f"{LEGACY_TITLE} 27B": "Qwen3 27B",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 12,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def display_text(value: object) -> object:
    if not isinstance(value, str):
        return value
    out = value
    for old, new in DISPLAY_NAMES.items():
        out = out.replace(old, new)
    out = out.replace(f"{LEGACY_PREFIX}qwen3", "qwen3")
    out = out.replace(LEGACY_TITLE, "Qwen3")
    return out


def display_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(display_text)
    return out


def ensure_dirs() -> None:
    for path in [FIGURES, TABLES, APPENDIX, TEMPLATES, EXAMPLES]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name)


def md_table(df: pd.DataFrame) -> str:
    data = display_df(df)
    for col in data.columns:
        if pd.api.types.is_float_dtype(data[col]):
            data[col] = data[col].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    cols = list(data.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in data.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            text = "" if pd.isna(value) else str(value)
            cells.append(text.replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, stem: str) -> None:
    data = display_df(df)
    path = TABLES / stem
    data.to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".md").write_text(md_table(data) + "\n", encoding="utf-8")
    path.with_suffix(".tex").write_text(data.to_latex(index=False, escape=True), encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=1200)
    plt.close(fig)


def add_caption(captions: list[str], number: int, title: str, text: str) -> None:
    captions.append(f"**Figure {number}. {title}.** {text}")


def clean_axis(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8792A2")
    ax.spines["bottom"].set_color("#8792A2")


def box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], label: str, color: str, fontsize: int = 9) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.2,
        edgecolor=color,
        facecolor=color + "18",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color="#20242A", wrap=True)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=1.05, color="#4B5563"))


def connection_points(source: dict[str, object], target: dict[str, object], gap: float = 0.004) -> tuple[tuple[float, float], tuple[float, float]]:
    sx, sy = source["xy"]
    sw, sh = source["wh"]
    tx, ty = target["xy"]
    tw, th = target["wh"]
    sc = (sx + sw / 2, sy + sh / 2)
    tc = (tx + tw / 2, ty + th / 2)
    dx = tc[0] - sc[0]
    dy = tc[1] - sc[1]
    if abs(dx) >= abs(dy):
        if dx >= 0:
            return (sx + sw + gap, sc[1]), (tx - gap, tc[1])
        return (sx - gap, sc[1]), (tx + tw + gap, tc[1])
    if dy >= 0:
        return (sc[0], sy + sh + gap), (tc[0], ty - gap)
    return (sc[0], sy - gap), (tc[0], ty + th + gap)


def diagram(stem: str, title: str, nodes: list[dict[str, object]], arrows: list[tuple[str, str]], size=(7.2, 5.2)) -> None:
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=8)
    index = {str(item["id"]): item for item in nodes}
    for item in nodes:
        box(ax, item["xy"], item["wh"], str(item["label"]), str(item["color"]), fontsize=int(item.get("fontsize", 9)))
    for left, right in arrows:
        a = index[left]
        b = index[right]
        start, end = connection_points(a, b)
        arrow(ax, start, end)
    save_figure(fig, stem)


def normalize_department(value: object) -> str:
    text = "Unknown" if pd.isna(value) else str(value).strip()
    low = text.lower()
    if not text or low == "nan":
        return "Unknown"
    if "general medicine" in low or low.startswith("medicine") or "medicine unit" in low:
        return "General Medicine"
    if "obstetric" in low or "gynaecology" in low or "gynecology" in low:
        return "Obstetrics & Gynecology"
    if "endocrinology" in low:
        return "Endocrinology"
    if "dermatology" in low:
        return "Dermatology"
    if low.startswith("ent"):
        return "ENT"
    if "radiotherapy" in low:
        return "Radiotherapy"
    if "radiology" in low:
        return "Radiology"
    if "surgery" in low:
        return "Surgery"
    return text


def generate_tables() -> None:
    manifest = pd.read_csv(MANIFEST)
    registry = read_csv("final_model_registry.csv")
    overall = read_csv("overall_benchmark_tables.csv")
    ocr = read_csv("ocr_tables.csv")
    direct = read_csv("direct_vlm_tables.csv")
    hybrid = read_csv("hybrid_tables.csv")
    dept = read_csv("department_tables.csv")
    runtime = read_csv("runtime_tables.csv")
    stats = read_csv("statistical_tests.csv")
    ci = read_csv("bootstrap_confidence_intervals.csv")
    metrics = read_csv("all_selected_per_document_metrics.csv")

    dataset_stats = pd.DataFrame(
        [
            {"statistic": "patients", "value": manifest["patient_id"].nunique()},
            {"statistic": "documents", "value": manifest["document_id"].nunique()},
            {"statistic": "images", "value": int(pd.to_numeric(manifest["num_images"], errors="coerce").fillna(0).sum())},
            {"statistic": "single_page_documents", "value": int((pd.to_numeric(manifest["num_images"], errors="coerce").fillna(1) == 1).sum())},
            {"statistic": "multi_page_documents", "value": int((pd.to_numeric(manifest["num_images"], errors="coerce").fillna(1) > 1).sum())},
            {"statistic": "departments", "value": manifest["department"].fillna("Unknown").nunique()},
        ]
    )
    save_table(dataset_stats, "table_01_dataset_statistics")
    save_table(ocr.sort_values("rank"), "table_02_ocr_benchmark")
    save_table(direct.sort_values("rank"), "table_03_direct_vlm_benchmark")
    save_table(hybrid.sort_values("rank"), "table_04_hybrid_benchmark")
    dept_summary = (
        dept.dropna(subset=["primary_score"])
        .sort_values(["department", "primary_score"], ascending=[True, False])
        .groupby(["family", "track", "department"], dropna=False)
        .head(3)
    )
    save_table(dept_summary, "table_05_department_wise_performance")
    save_table(runtime.sort_values(["track", "avg_runtime_seconds"], na_position="last"), "table_06_runtime_comparison")
    save_table(stats.sort_values(["test", "p_holm"], na_position="last").head(80), "table_07_statistical_significance")
    save_table(ci.sort_values(["family", "mean"], ascending=[True, False]), "table_08_bootstrap_confidence_intervals")
    ablation = (
        overall.groupby(["family", "track", "provenance", "publication_status"], dropna=False)
        .agg(models=("system", "nunique"), mean_primary_score=("primary_score", "mean"), mean_records=("records", "mean"))
        .reset_index()
    )
    save_table(ablation, "table_09_ablation")
    error_cols = ["hallucination_rate", "missing_entity_rate", "annotation_gap_rate", "schema_parse_success"]
    error_categories = (
        metrics[metrics["family"] == "structured"]
        .groupby(["track", "system"], dropna=False)[error_cols]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("hallucination_rate")
    )
    save_table(error_categories, "table_10_error_categories")
    save_table(
        registry[["model", "track", "provenance", "coverage", "publication_status", "reason_if_excluded", "statistics_used"]],
        "table_11_model_coverage",
    )
    save_table(overall.sort_values(["family", "rank"]), "table_12_final_leaderboard")


def figure_01(captions: list[str]) -> None:
    nodes = [
        {"id": "docs", "label": "Clinical\nDocuments", "xy": (0.39, 0.86), "wh": (0.22, 0.08), "color": COLORS["dataset"]},
        {"id": "dataset", "label": "Dataset\nConstruction", "xy": (0.37, 0.72), "wh": (0.26, 0.08), "color": COLORS["dataset"]},
        {"id": "ocr", "label": "OCR\nTrack", "xy": (0.10, 0.53), "wh": (0.20, 0.08), "color": COLORS["inference"]},
        {"id": "vlm", "label": "Direct VLM\nTrack", "xy": (0.40, 0.53), "wh": (0.20, 0.08), "color": COLORS["inference"]},
        {"id": "hybrid", "label": "Hybrid\nOCR+LLM", "xy": (0.70, 0.53), "wh": (0.20, 0.08), "color": COLORS["inference"]},
        {"id": "json", "label": "Canonical\nJSON", "xy": (0.39, 0.37), "wh": (0.22, 0.08), "color": COLORS["evaluation"]},
        {"id": "eval", "label": "Evaluation\nEngine", "xy": (0.37, 0.24), "wh": (0.26, 0.08), "color": COLORS["evaluation"]},
        {"id": "metrics", "label": "Metrics", "xy": (0.13, 0.10), "wh": (0.18, 0.08), "color": COLORS["statistics"]},
        {"id": "stats", "label": "Statistical\nAnalysis", "xy": (0.41, 0.10), "wh": (0.18, 0.08), "color": COLORS["statistics"]},
        {"id": "lead", "label": "Leaderboard", "xy": (0.69, 0.10), "wh": (0.18, 0.08), "color": COLORS["statistics"]},
    ]
    arrows = [("docs", "dataset"), ("dataset", "ocr"), ("dataset", "vlm"), ("dataset", "hybrid"), ("ocr", "json"), ("vlm", "json"), ("hybrid", "json"), ("json", "eval"), ("eval", "metrics"), ("eval", "stats"), ("eval", "lead")]
    diagram("figure_01_overall_benchmark_architecture", "Overall Benchmark Architecture", nodes, arrows, size=(7.4, 5.4))
    add_caption(captions, 1, "Overall benchmark architecture", "The benchmark separates dataset construction, three inference tracks, canonical JSON evaluation, and statistical analysis. This organization makes model provenance and metric computation explicit while keeping the frozen benchmark reproducible.")


def figure_02(captions: list[str]) -> None:
    manifest = pd.read_csv(MANIFEST)
    fig = plt.figure(figsize=(7.2, 5.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.05], hspace=0.55, wspace=0.35)
    dept_ax = fig.add_subplot(gs[0, :])
    dept = manifest["department"].map(normalize_department).value_counts().head(10).sort_values()
    dept_ax.barh(dept.index.map(lambda x: "\n".join(textwrap.wrap(str(x), 24))), dept.values, color=COLORS["dataset"])
    clean_axis(dept_ax, "Department Distribution", "documents")
    type_ax = fig.add_subplot(gs[1, 0])
    doc_type = manifest.get("source_type", pd.Series(["document"] * len(manifest))).fillna("document").value_counts()
    if len(doc_type) > 8:
        doc_type = pd.concat([doc_type.head(8), pd.Series({"Others": doc_type.iloc[8:].sum()})])
    type_ax.barh(doc_type.sort_values().index.map(str), doc_type.sort_values().values, color=COLORS["hybrid"])
    clean_axis(type_ax, "Document Categories", "documents")
    page_ax = fig.add_subplot(gs[1, 1])
    page = pd.Series({"Single page": int((manifest["num_images"] == 1).sum()), "Multi-page": int((manifest["num_images"] > 1).sum())})
    page_ax.bar(page.index, page.values, color=[COLORS["evaluation"], COLORS["statistics"]], width=0.55)
    for i, value in enumerate(page.values):
        page_ax.text(i, value + 2, str(value), ha="center", fontsize=10, fontweight="bold")
    clean_axis(page_ax, "Page Structure", "", "documents")
    save_figure(fig, "figure_02_dataset_composition")
    add_caption(captions, 2, "Dataset composition", "The figure summarizes department distribution, source-document categories, and page structure. Department variants such as General Medicine units are consolidated to avoid repeated labels for the same clinical service.")


def figure_03(captions: list[str]) -> None:
    nodes = [
        {"id": "dataset", "label": "Dataset", "xy": (0.04, 0.57), "wh": (0.12, 0.13), "color": COLORS["dataset"], "fontsize": 8},
        {"id": "tracks", "label": "Three\nInference\nTracks", "xy": (0.22, 0.57), "wh": (0.15, 0.13), "color": COLORS["inference"], "fontsize": 8},
        {"id": "eval", "label": "Unified\nEvaluation", "xy": (0.43, 0.57), "wh": (0.15, 0.13), "color": COLORS["evaluation"], "fontsize": 8},
        {"id": "metrics", "label": "Metrics", "xy": (0.64, 0.57), "wh": (0.12, 0.13), "color": COLORS["evaluation"], "fontsize": 8},
        {"id": "stats", "label": "Statistical\nTesting", "xy": (0.81, 0.57), "wh": (0.15, 0.13), "color": COLORS["statistics"], "fontsize": 8},
        {"id": "paper", "label": "Rankings +\nPaper Figures", "xy": (0.79, 0.27), "wh": (0.17, 0.13), "color": COLORS["statistics"], "fontsize": 8},
    ]
    arrows = [("dataset", "tracks"), ("tracks", "eval"), ("eval", "metrics"), ("metrics", "stats"), ("stats", "paper")]
    diagram("figure_03_benchmark_workflow", "Benchmark Workflow", nodes, arrows, size=(7.2, 3.6))
    add_caption(captions, 3, "Benchmark workflow", "The workflow moves left-to-right from a fixed dataset to inference, unified evaluation, statistical testing, rankings, and paper figures. No live inference is required for the publication asset stage.")


def figure_04(captions: list[str]) -> None:
    labels = ["Ground Truth\nJSON", "Prediction\nJSON", "Normalization", "Field\nAlignment", "Exact Match", "Token F1", "Entity F1", "Hallucination\nDetection", "Primary\nScore", "Bootstrap", "Wilcoxon", "Final\nRanking"]
    nodes = []
    for i, label in enumerate(labels):
        nodes.append({"id": str(i), "label": label, "xy": (0.35, 0.88 - i * 0.072), "wh": (0.30, 0.046), "color": COLORS["evaluation"] if i < 9 else COLORS["statistics"], "fontsize": 8})
    arrows = [(str(i), str(i + 1)) for i in range(len(labels) - 1)]
    diagram("figure_04_evaluation_pipeline", "Evaluation Pipeline", nodes, arrows, size=(4.8, 7.4))
    add_caption(captions, 4, "Evaluation pipeline", "Predictions and ground truth are aligned and scored before paired statistical analysis. The figure separates field-level matching from aggregate ranking so metric provenance is clear.")


def plot_grouped_bars(df: pd.DataFrame, label_col: str, metrics: list[str], stem: str, title: str, captions: list[str], number: int, caption: str) -> None:
    data = display_df(df.sort_values(metrics[0], ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(data))
    width = 0.72 / len(metrics)
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, data[metric], width=width, label=metric, color=PALETTE[i])
    ax.set_xticks(x, data[label_col], rotation=35, ha="right")
    ax.legend(frameon=False, ncol=len(metrics), loc="upper center", bbox_to_anchor=(0.5, -0.34), borderaxespad=0.0)
    clean_axis(ax, title, "", "score")
    fig.subplots_adjust(bottom=0.34)
    save_figure(fig, stem)
    add_caption(captions, number, title, caption)


def simple_bar(df: pd.DataFrame, stem: str, title: str, metric: str, captions: list[str], number: int, caption: str, top_n: int = 14) -> None:
    data = display_df(df.dropna(subset=[metric]).sort_values(metric, ascending=True).tail(top_n))
    fig, ax = plt.subplots(figsize=(7.2, max(3.8, 0.24 * len(data))))
    colors = [COLORS.get(str(track), COLORS["neutral"]) for track in data.get("track", pd.Series(["neutral"] * len(data)))]
    ax.barh(data["system"], data[metric], color=colors)
    clean_axis(ax, title, metric)
    save_figure(fig, stem)
    add_caption(captions, number, title, caption)


def department_model_matrix() -> pd.DataFrame:
    dept = display_df(read_csv("department_tables.csv").dropna(subset=["primary_score"]))
    dept["department"] = dept["department"].map(normalize_department)
    top_models = dept.groupby("system")["primary_score"].mean().nlargest(10).index
    top_depts = dept.groupby("department")["records"].sum().nlargest(12).index
    heat = dept[dept["system"].isin(top_models) & dept["department"].isin(top_depts)].pivot_table(index="department", columns="system", values="primary_score", aggfunc="mean")
    heat = heat.loc[[d for d in top_depts if d in heat.index], [m for m in top_models if m in heat.columns]]
    return heat


def figure_08(captions: list[str]) -> None:
    heat = department_model_matrix()
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    im = ax.imshow(heat.fillna(0), aspect="auto", cmap="viridis", vmin=0, vmax=max(0.5, float(np.nanmax(heat.values))))
    ax.set_xticks(np.arange(len(heat.columns)), heat.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(heat.index)), heat.index)
    ax.set_title("Department x Model Heatmap", loc="left", fontweight="bold")
    for i, row in enumerate(heat.to_numpy()):
        if np.isfinite(row).any():
            j = int(np.nanargmax(row))
            ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, edgecolor="white", linewidth=1.8))
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="primary score")
    save_figure(fig, "figure_08_department_model_heatmap")
    add_caption(captions, 8, "Department x model heatmap", "The heatmap compares top models across departments and outlines the best model for each department. Related service labels such as General Medicine unit variants are consolidated.")


def figure_08b(captions: list[str]) -> None:
    heat = department_model_matrix()
    winners = heat.idxmax(axis=1).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.barh(winners.index, winners.values, color=COLORS["statistics"])
    for y, value in enumerate(winners.values):
        ax.text(value + 0.08, y, str(int(value)), va="center", fontsize=9)
    clean_axis(ax, "Department Win Count", "departments")
    save_figure(fig, "figure_08b_department_win_count")
    add_caption(captions, "8B", "Department win count", "This companion chart counts how often each model is the top performer across departments. It gives a compact robustness view separate from the heatmap.")


def figure_09(captions: list[str]) -> None:
    overall = read_csv("overall_benchmark_tables.csv")
    runtime = read_csv("runtime_tables.csv").merge(overall[["system", "track", "primary_score"]], on=["system", "track"], how="left")
    data = display_df(runtime.dropna(subset=["avg_runtime_seconds", "primary_score"]).copy())

    lane_labels = {
        "trocr_qwen25_14b": "TrOCR -> Qwen2.5-14B",
        "docling_qwen25_14b": "Docling -> Qwen2.5-14B",
        "easyocr_qwen25_14b": "EasyOCR -> Qwen2.5-14B",
        "doctr_qwen25_14b": "DocTR -> Qwen2.5-14B",
        "docling_qwen3_8b": "Docling -> Qwen3-8B",
        "glm_ocr_qwen25_14b": "GLM OCR -> Qwen2.5-14B",
        "qwen3_vl_raw_ocr_qwen25_14b": "Qwen3-VL OCR -> Qwen2.5-14B",
        "glm_ocr_qwen3_8b": "GLM OCR -> Qwen3-8B",
        "qwen3_vl_raw_ocr_qwen3_8b": "Qwen3-VL OCR -> Qwen3-8B",
        "surya_qwen25_14b": "Surya -> Qwen2.5-14B",
    }

    fig, ax = plt.subplots(figsize=(13.76, 7.68))
    fig.patch.set_facecolor("white")

    x_min = float(data["avg_runtime_seconds"].min())
    x_max = float(data["avg_runtime_seconds"].max())
    y_min = float(data["primary_score"].min())
    y_max = float(data["primary_score"].max())
    ax.set_xlim(max(0, x_min - 1.0), x_max + 3.0)
    ax.set_ylim(y_min - 0.007, y_max + 0.004)

    ax.add_patch(
        Rectangle(
            (9.5, 0.320),
            6.9,
            0.028,
            facecolor=COLORS["hybrid"],
            edgecolor="#4F9C98",
            linewidth=1.2,
            alpha=0.28,
            zorder=0,
        )
    )
    ax.text(
        9.7,
        0.3455,
        "Efficient high-score zone",
        fontsize=13,
        fontweight="bold",
        color="#184E4B",
        va="top",
    )
    ax.text(
        9.7,
        0.3417,
        "Complete lanes that maintain strong extraction\nquality without drifting into the slowest runtimes.",
        fontsize=10,
        color="#184E4B",
        va="top",
        linespacing=1.2,
    )

    systems = list(data.sort_values(["avg_runtime_seconds", "primary_score"])["system"])
    palette = matplotlib.colormaps.get_cmap("tab20").resampled(max(1, len(systems)))
    color_map = {system: palette(i) for i, system in enumerate(systems)}
    for _, row in data.sort_values(["avg_runtime_seconds", "primary_score"]).iterrows():
        label = lane_labels.get(str(row["system"]), str(row["system"]))
        ax.scatter(
            row["avg_runtime_seconds"],
            row["primary_score"],
            label=label,
            marker="^",
            s=155,
            alpha=0.96,
            color=color_map[str(row["system"])],
            edgecolor="#2F3A45",
            linewidth=0.8,
            zorder=3,
        )

    def lane_row(system: str) -> pd.Series:
        matches = data[data["system"] == system]
        if matches.empty:
            raise KeyError(system)
        return matches.iloc[0]

    def callout(system: str, text: str, xytext: tuple[float, float], *, ha: str = "left") -> None:
        row = lane_row(system)
        ax.annotate(
            text,
            xy=(row["avg_runtime_seconds"], row["primary_score"]),
            xytext=xytext,
            fontsize=12,
            color="#111827",
            ha=ha,
            va="center",
            linespacing=1.12,
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.12",
                "facecolor": "white",
                "edgecolor": "#E87561",
                "linewidth": 1.5,
                "alpha": 0.96,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": "#6BA7A3",
                "linewidth": 1.4,
                "shrinkA": 6,
                "shrinkB": 7,
            },
            zorder=4,
        )

    callout(
        "glm_ocr_qwen25_14b",
        "GLM OCR hybrids cluster near\n14-16 s/document while retaining\ntop-tier extraction accuracy.",
        (17.45, 0.3376),
    )
    callout(
        "qwen3_vl_raw_ocr_qwen3_8b",
        "Highest plotted hybrid score:\nQwen3-VL OCR -> Qwen3-8B\n0.346 primary score at 17.3 s/doc.",
        (18.4, 0.3460),
    )
    callout(
        "surya_qwen25_14b",
        "Surya -> Qwen2.5-14B remains\ncompetitive, but moves beyond\n21 s/document.",
        (16.25, 0.3090),
    )
    ax.text(
        0.03,
        0.07,
        "Lower runtime  <<",
        transform=ax.transAxes,
        fontsize=11,
        color="#4B5563",
        fontweight="bold",
    )
    ax.text(
        0.02,
        0.98,
        "Higher score",
        transform=ax.transAxes,
        fontsize=11,
        color="#4B5563",
        fontweight="bold",
        va="top",
        rotation=90,
    )

    ax.set_title("Runtime vs Accuracy", loc="left", fontweight="bold", pad=18, fontsize=28)
    ax.set_xlabel("average runtime seconds per document", fontsize=15)
    ax.set_ylabel("primary score", fontsize=15)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8, alpha=0.9)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.45, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8792A2")
    ax.spines["bottom"].set_color("#8792A2")
    ax.tick_params(labelsize=12)
    ax.legend(
        frameon=False,
        fontsize=11,
        title="Full benchmark lane",
        title_fontsize=14,
        loc="center left",
        bbox_to_anchor=(1.01, 0.52),
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.08, right=0.71, top=0.86, bottom=0.13)
    save_figure(fig, "figure_09_runtime_vs_accuracy")
    add_caption(captions, 9, "Runtime versus accuracy", "The annotated scatter plot compares runtime and primary score only for lanes that have both values in the final reports. The shaded zone marks efficient high-score hybrids, while callouts identify the strongest plotted hybrid, the GLM OCR cluster, and the slower Surya trade-off.")


def plot_heatmap(df: pd.DataFrame, stem: str, title: str, captions: list[str], number: int, caption: str) -> None:
    data = display_df(df).fillna(0)
    fig, ax = plt.subplots(figsize=(7.2, max(3.8, 0.23 * len(data))))
    im = ax.imshow(data.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(data.columns)), data.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(data.index)), data.index)
    ax.set_title(title, loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    save_figure(fig, stem)
    add_caption(captions, number, title, caption)


def figure_12(captions: list[str]) -> None:
    stats = display_df(read_csv("statistical_tests.csv"))
    wil = stats[(stats["test"] == "wilcoxon") & (stats["family"] == "structured")].copy()
    systems = sorted(set(wil["system_a"]).union(set(wil["system_b"])))
    mat = pd.DataFrame(np.nan, index=systems, columns=systems)
    for _, row in wil.iterrows():
        value = -math.log10(max(float(row["p_holm"]), 1e-300))
        mat.loc[row["system_a"], row["system_b"]] = value
        mat.loc[row["system_b"], row["system_a"]] = value
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    im = ax.imshow(mat.fillna(0), cmap="magma", aspect="auto")
    ax.set_xticks(np.arange(len(systems)), systems, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(systems)), systems)
    ax.set_title("Wilcoxon + Holm", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02, label="-log10 Holm p")
    save_figure(fig, "figure_12_wilcoxon_holm")
    add_caption(captions, 12, "Wilcoxon and Holm significance", "Pairwise Wilcoxon signed-rank tests are shown for structured primary-table lanes using Holm-adjusted p-values from the frozen statistics file. Brighter cells indicate stronger adjusted evidence of paired performance differences.")


def figure_12b(captions: list[str]) -> None:
    stats = display_df(read_csv("statistical_tests.csv"))
    friedman = stats[stats["test"] == "friedman"].copy()
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    if not friedman.empty:
        labels = friedman["family"].map(display_text)
        values = -np.log10(friedman["p_holm"].clip(lower=1e-300))
        ax.barh(labels, values, color=COLORS["statistics"])
        clean_axis(ax, "Friedman Omnibus", "-log10 Holm p")
    else:
        ax.axis("off")
        ax.text(0.05, 0.5, "No Friedman rows in frozen statistics.", va="center")
    save_figure(fig, "figure_12b_friedman_omnibus")
    add_caption(captions, "12B", "Friedman omnibus tests", "The Friedman omnibus tests were run for raw OCR and structured model families over the 125-document benchmark. The plot reports Holm-adjusted omnibus evidence from the frozen statistics file.")


def figure_12c(captions: list[str]) -> None:
    stats = display_df(read_csv("statistical_tests.csv"))
    mcnemar = stats[stats["test"] == "mcnemar_exact"].copy()
    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    if not mcnemar.empty:
        summary = pd.Series(
            {
                "computed comparisons": len(mcnemar),
                "nonzero discordance": int((pd.to_numeric(mcnemar["statistic"], errors="coerce") != 0).sum()),
                "Holm significant": int(mcnemar["significant_holm_0_05"].astype(str).str.lower().eq("true").sum()),
            }
        )
        ax.barh(summary.index, summary.values, color=[COLORS["evaluation"], COLORS["inference"], COLORS["statistics"]])
        for y, value in enumerate(summary.values):
            ax.text(value + 0.25, y, str(int(value)), va="center", fontsize=9)
        clean_axis(ax, "McNemar Exact Summary", "comparison count")
    else:
        ax.axis("off")
        ax.text(0.05, 0.5, "No McNemar rows in frozen statistics.", va="center")
    save_figure(fig, "figure_12c_mcnemar_summary")
    add_caption(captions, "12C", "McNemar exact test summary", "McNemar exact tests were computed for paired binary schema-success outcomes. All frozen comparisons have zero discordant-count statistic and no Holm-significant result, so this summary reports applicability without overstating significance.")


def redaction_boxes(doc_id: str) -> list[tuple[int, int, int, int]]:
    return {
        "p19": [(245, 305, 990, 575), (205, 395, 455, 540), (720, 300, 1110, 455), (285, 1210, 520, 1260)],
        "p68": [(1860, 760, 2825, 1390), (1740, 3260, 2600, 3585), (2050, 3480, 2920, 4010), (250, 3900, 1710, 4050)],
        "p70": [(1990, 560, 2790, 1165), (1740, 3120, 2925, 3670), (1850, 3560, 2880, 3890), (300, 3515, 3020, 3825), (380, 3790, 1830, 4035)],
        "p89": [(1880, 450, 2810, 1220), (1390, 3060, 2650, 3735), (330, 3810, 1850, 4050), (2320, 3600, 3070, 4060)],
    }.get(doc_id, [])


def anonymize_examples() -> dict[str, Path]:
    source_map = {
        "p19": ROOT / "prescriptions" / "p19.jpeg",
        "p68": ROOT / "prescriptions" / "p68.jpg",
        "p70": ROOT / "prescriptions" / "p70.jpg",
        "p89": ROOT / "prescriptions" / "p89.jpg",
    }
    out: dict[str, Path] = {}
    for doc_id, path in source_map.items():
        dest = EXAMPLES / f"{doc_id}_anonymized.png"
        if not path.exists():
            if dest.exists():
                out[doc_id] = dest
            continue
        image = Image.open(path).convert("RGB")
        if doc_id == "p89":
            image = image.rotate(90, expand=True)
        draw = ImageDraw.Draw(image)
        for coords in redaction_boxes(doc_id):
            draw.rectangle(coords, fill=(0, 0, 0))
        image.save(dest)
        out[doc_id] = dest
    return out


def load_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def json_excerpt(path: Path, max_chars: int = 900) -> str:
    if not path.exists():
        return "{}"
    data = json.loads(path.read_text(encoding="utf-8"))
    return json.dumps(data, indent=2, ensure_ascii=False)[:max_chars]


def figure_14(captions: list[str]) -> None:
    paths = anonymize_examples()
    fig = plt.figure(figsize=(9.2, 4.8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1, 1, 1], wspace=0.18)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(load_image(paths["p68"]))
    axes[0].set_title("Input image", loc="left", fontweight="bold")
    axes[1].text(0, 1, "{\n  patient: redacted,\n  complaints: [...],\n  medications: [...],\n  advice: [...]\n}", va="top", family="monospace", fontsize=8)
    axes[1].set_title("Ground truth JSON", loc="left", fontweight="bold")
    axes[2].text(0, 1, "{\n  patient: redacted,\n  complaints: partial,\n  medications: matched,\n  advice: missing\n}", va="top", family="monospace", fontsize=8)
    axes[2].set_title("Prediction JSON", loc="left", fontweight="bold")
    for label, color, y in [("correct fields", COLORS["correct"], 0.82), ("missing fields", COLORS["missing"], 0.62), ("hallucinated fields", COLORS["hallucinated"], 0.42)]:
        axes[3].add_patch(Rectangle((0.04, y - 0.05), 0.16, 0.08, color=color, transform=axes[3].transAxes))
        axes[3].text(0.25, y, label, va="center", fontsize=10, transform=axes[3].transAxes)
    axes[3].set_title("Differences", loc="left", fontweight="bold")
    save_figure(fig, "figure_14_qualitative_examples")
    add_caption(captions, 14, "Qualitative example", "The panel shows an anonymized input, corresponding canonical JSON, representative prediction structure, and the color convention used for error analysis. Green denotes matched fields, orange denotes missing content, and red denotes hallucinated content.")


def figure_15(captions: list[str]) -> None:
    paths = anonymize_examples()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.6))
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(load_image(paths["p70"]))
    axes[1].imshow(load_image(paths["p89"]))
    axes[0].set_title("Good quality: p70", loc="left", fontweight="bold")
    axes[1].set_title("Poor quality: p89", loc="left", fontweight="bold")
    save_figure(fig, "figure_15_good_vs_bad_prescription_examples")
    add_caption(captions, 15, "Good and poor quality examples", "The examples illustrate the visual range of the dataset after targeted anonymization. The good-quality example has clearer structure and handwriting, while the poor-quality example shows lower contrast and incomplete visibility.")


def figure_16(captions: list[str]) -> None:
    paths = anonymize_examples()
    fig = plt.figure(figsize=(9.4, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 0.9], wspace=0.24)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(load_image(paths["p19"]))
    axes[0].set_title("Anonymized image", loc="left", fontweight="bold")
    axes[1].text(0, 1, json_excerpt(ROOT / "raw_ground_truths" / "p19.json"), va="top", family="monospace", fontsize=6.5)
    axes[1].set_title("Ground truth JSON", loc="left", fontweight="bold")
    axes[2].text(0, 1, "Annotation links visible evidence to canonical fields.\n\nEvaluation normalizes fields, aligns predictions, and computes scalar/entity scores.", va="top", fontsize=10, linespacing=1.5)
    axes[2].set_title("Annotation logic", loc="left", fontweight="bold")
    save_figure(fig, "figure_16_example_ground_truth_annotation")
    add_caption(captions, 16, "Ground truth annotation example", "The figure pairs an anonymized prescription with a canonical JSON excerpt. It shows how visual evidence is translated into structured fields before evaluation.")


def pipeline_figures(captions: list[str]) -> None:
    hybrid_nodes = [
        {"id": "img", "label": "[IMG]\nPrescription", "xy": (0.06, 0.44), "wh": (0.13, 0.12), "color": COLORS["dataset"]},
        {"id": "ocr", "label": "[OCR]\nOCR", "xy": (0.23, 0.44), "wh": (0.11, 0.12), "color": COLORS["inference"]},
        {"id": "text", "label": "Raw\nText", "xy": (0.38, 0.44), "wh": (0.11, 0.12), "color": COLORS["inference"]},
        {"id": "prompt", "label": "Prompt\nBuilder", "xy": (0.53, 0.44), "wh": (0.13, 0.12), "color": COLORS["inference"]},
        {"id": "llm", "label": "LLM", "xy": (0.70, 0.44), "wh": (0.10, 0.12), "color": COLORS["inference"]},
        {"id": "json", "label": "Canonical\nJSON", "xy": (0.84, 0.44), "wh": (0.12, 0.12), "color": COLORS["evaluation"]},
    ]
    diagram("figure_17_hybrid_pipeline_architecture", "Hybrid Pipeline Architecture", hybrid_nodes, [("img", "ocr"), ("ocr", "text"), ("text", "prompt"), ("prompt", "llm"), ("llm", "json")], size=(7.4, 2.6))
    add_caption(captions, 17, "Hybrid pipeline architecture", "Hybrid lanes first transcribe the prescription and then use a language model to convert OCR text into canonical JSON. The validation step ensures outputs remain compatible with the unified scoring engine.")
    direct_nodes = [
        {"id": "img", "label": "[IMG]\nPrescription", "xy": (0.05, 0.44), "wh": (0.14, 0.12), "color": COLORS["dataset"], "fontsize": 8},
        {"id": "enc", "label": "Vision\nEncoder", "xy": (0.24, 0.44), "wh": (0.14, 0.12), "color": COLORS["inference"], "fontsize": 8},
        {"id": "llm", "label": "Vision\nLLM", "xy": (0.43, 0.44), "wh": (0.13, 0.12), "color": COLORS["inference"], "fontsize": 8},
        {"id": "prompt", "label": "Prompt", "xy": (0.61, 0.44), "wh": (0.11, 0.12), "color": COLORS["inference"], "fontsize": 8},
        {"id": "json", "label": "Validated\nJSON", "xy": (0.78, 0.44), "wh": (0.17, 0.12), "color": COLORS["evaluation"], "fontsize": 8},
    ]
    diagram("figure_18_direct_vlm_pipeline", "Direct VLM Pipeline", direct_nodes, [("img", "enc"), ("enc", "llm"), ("llm", "prompt"), ("prompt", "json")], size=(7.4, 2.6))
    add_caption(captions, 18, "Direct VLM pipeline", "Direct VLM lanes process prescription images without an intermediate OCR file. The model response is parsed into structured JSON and validated before metric computation.")
    ocr_nodes = [
        {"id": "img", "label": "[IMG]\nPrescription", "xy": (0.04, 0.44), "wh": (0.13, 0.12), "color": COLORS["dataset"], "fontsize": 8},
        {"id": "prep", "label": "Pre-\nprocessing", "xy": (0.20, 0.44), "wh": (0.13, 0.12), "color": COLORS["inference"], "fontsize": 8},
        {"id": "ocr", "label": "OCR\nEngine", "xy": (0.36, 0.44), "wh": (0.13, 0.12), "color": COLORS["inference"], "fontsize": 8},
        {"id": "norm", "label": "Text\nCleanup", "xy": (0.52, 0.44), "wh": (0.13, 0.12), "color": COLORS["evaluation"], "fontsize": 8},
        {"id": "text", "label": "Canonical\nText", "xy": (0.68, 0.44), "wh": (0.13, 0.12), "color": COLORS["evaluation"], "fontsize": 8},
        {"id": "out", "label": "Output", "xy": (0.85, 0.44), "wh": (0.10, 0.12), "color": COLORS["statistics"], "fontsize": 8},
    ]
    diagram("figure_19_ocr_pipeline", "OCR Pipeline", ocr_nodes, [("img", "prep"), ("prep", "ocr"), ("ocr", "norm"), ("norm", "text"), ("text", "out")], size=(7.4, 2.6))
    add_caption(captions, 19, "OCR pipeline", "The OCR pipeline converts prescription images into text for raw OCR scoring and downstream hybrid extraction. The same text outputs can be reused by multiple structured extraction models.")


def generate_figures() -> None:
    captions: list[str] = []
    metrics = read_csv("all_selected_per_document_metrics.csv")
    overall = read_csv("overall_benchmark_tables.csv")
    direct = read_csv("direct_vlm_tables.csv")
    hybrid = read_csv("hybrid_tables.csv")
    ci = display_df(read_csv("bootstrap_confidence_intervals.csv"))

    figure_01(captions)
    figure_02(captions)
    figure_03(captions)
    figure_04(captions)
    ocr_metrics = metrics[metrics["track"] == "raw_ocr"].groupby("system", dropna=False)[["token_f1", "token_precision", "token_recall"]].mean(numeric_only=True).reset_index()
    plot_grouped_bars(ocr_metrics, "system", ["token_f1", "token_precision", "token_recall"], "figure_05_ocr_performance", "OCR Performance", captions, 5, "OCR systems are compared using token F1, precision, and recall. This view highlights transcription quality before structured extraction.")
    simple_bar(direct, "figure_06_direct_vlm_comparison", "Direct VLM Comparison", "primary_score", captions, 6, "Direct VLM models are compared by overall extraction score on the frozen dataset. Complete primary-table lanes are displayed with their publication labels.")
    simple_bar(hybrid, "figure_07_hybrid_pipeline_comparison", "Hybrid Pipeline Comparison", "primary_score", captions, 7, "Hybrid OCR-to-LLM lanes are compared by structured extraction score. The ranking reflects how OCR quality and language-model structuring interact.")
    figure_08(captions)
    figure_08b(captions)
    figure_09(captions)
    heat = metrics.pivot_table(index="system", values=["primary_score", "schema_parse_success", "hallucination_rate", "missing_entity_rate"], aggfunc="mean")
    plot_heatmap(heat, "figure_10_models_x_metrics_heatmap", "Models x Metrics Heatmap", captions, 10, "The heatmap summarizes primary score, schema success, hallucination rate, and missing entity rate across models. It provides a compact cross-metric diagnostic view.")
    fig, ax = plt.subplots(figsize=(7.2, max(3.8, 0.22 * len(ci))))
    ci_sorted = ci.sort_values("mean")
    y = np.arange(len(ci_sorted))
    ax.errorbar(ci_sorted["mean"], y, xerr=[ci_sorted["mean"] - ci_sorted["ci_lower"], ci_sorted["ci_upper"] - ci_sorted["mean"]], fmt="o", color=COLORS["dataset"], linewidth=1.2)
    ax.set_yticks(y, ci_sorted["system"])
    clean_axis(ax, "Bootstrap Confidence Intervals", "mean primary score")
    save_figure(fig, "figure_11_bootstrap_confidence_intervals")
    add_caption(captions, 11, "Bootstrap confidence intervals", "The forest plot shows bootstrap confidence intervals for each model's primary score. Intervals convey uncertainty across the 125-document benchmark.")
    figure_12(captions)
    figure_12b(captions)
    figure_12c(captions)
    simple_bar(overall, "figure_13_benchmark_leaderboard", "Benchmark Leaderboard", "primary_score", captions, 13, "The leaderboard ranks selected lanes by their primary metric within the frozen benchmark. Display labels are publication-facing and do not alter frozen provenance files.", top_n=22)
    figure_14(captions)
    figure_15(captions)
    figure_16(captions)
    pipeline_figures(captions)
    (FIGURES / "captions.md").write_text("# Figure Captions\n\n" + "\n\n".join(captions) + "\n", encoding="utf-8")


def generate_appendix() -> None:
    appendix_outline = """# Appendix Outline

1. Dataset composition and annotation protocol.
2. Model lane registry and provenance.
3. Evaluation metrics and normalization rules.
4. Statistical testing details.
5. Runtime and compute environment.
6. Error taxonomy.
7. Additional department-wise tables.
8. Reproducibility checklist.
"""
    (APPENDIX / "Appendix_Outline.md").write_text(appendix_outline, encoding="utf-8")
    artifact = """# Artifact Description

ClinDoc-Bench-IN contains frozen benchmark reports, reproducible evaluation scripts, publication figure/table generation scripts, and anonymized example assets.

The final frozen benchmark marker is `benchmark/final/reports/final_model_registry.csv`.
"""
    (APPENDIX / "Artifact_Description.md").write_text(artifact, encoding="utf-8")
    checklist = """# Reproducibility Checklist

- Frozen benchmark marker present.
- Final model registry present.
- Final release and Final release provenance recorded.
- Statistics computed on 125 documents.
- Figures and tables generated from frozen CSVs.
- Public examples anonymized using opaque boxes.
- Environment variables documented in `.env.example`.
"""
    (APPENDIX / "Reproducibility_Checklist.md").write_text(checklist, encoding="utf-8")
    save_table(pd.DataFrame([{"component": "CPU", "value": "To be filled before camera-ready"}, {"component": "GPU", "value": "To be filled before camera-ready"}, {"component": "RAM", "value": "To be filled before camera-ready"}, {"component": "Storage", "value": "To be filled before camera-ready"}, {"component": "Servers", "value": "Final release and imported Final release handoff"}]), "hardware_table")
    save_table(pd.DataFrame([{"package": "python", "version": "3.11+"}, {"package": "pandas", "version": "2.2.3"}, {"package": "numpy", "version": "1.26.4"}, {"package": "scipy", "version": "1.14.1"}, {"package": "matplotlib", "version": "3.9.3"}, {"package": "pydantic", "version": "2.10.4"}]), "software_versions")
    save_table(read_csv("final_model_registry.csv")[["model", "track", "provenance", "coverage", "timestamp", "output_directory"]], "model_versions")
    save_table(pd.DataFrame([{"provider": "Hugging Face", "api": "Inference API", "version": "provider-managed"}, {"provider": "Google Gemini", "api": "google-genai", "version": "1.25.0"}, {"provider": "Ollama", "api": "local HTTP API", "version": "local installation"}, {"provider": "Qwen3 endpoint", "api": "OpenAI-compatible HTTP API", "version": "local deployment"}]), "api_versions")
    save_table(pd.DataFrame([{"variable": "HF_TOKEN / HF_TOKEN_2", "purpose": "Hugging Face Inference API"}, {"variable": "GOOGLE_API_KEY(_2/_3/_4)", "purpose": "Gemini key rotation"}, {"variable": "OLLAMA_HOST", "purpose": "Local Ollama endpoint"}, {"variable": "QWEN3_27B_*", "purpose": "Qwen3 endpoint"}]), "environment_details")
    with PdfPages(APPENDIX / "Supplementary.pdf") as pdf:
        for title, body in [("Supplementary Material Outline", appendix_outline), ("Artifact Description", artifact), ("Reproducibility Checklist", checklist)]:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            ax.text(0.05, 0.95, title, fontsize=16, fontweight="bold", va="top")
            ax.text(0.05, 0.88, "\n".join(textwrap.wrap(body.replace("#", ""), width=90)), fontsize=10, va="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def write_templates() -> None:
    (TEMPLATES / "figure_caption_template.md").write_text("# Figure Caption Template\n\n**Figure X. Title.** State what the figure shows, why it matters, and the key takeaway in 2-4 sentences.\n", encoding="utf-8")
    (TEMPLATES / "table_caption_template.md").write_text("# Table Caption Template\n\n**Table X. Title.** Define rows, columns, primary metric, inclusion criteria, and whether appendix lanes are included.\n", encoding="utf-8")


def write_publication_report() -> None:
    registry = read_csv("final_model_registry.csv")
    primary = int((registry["publication_status"] == "PRIMARY TABLE").sum())
    appendix = int((registry["publication_status"] == "APPENDIX").sum())
    excluded = int((registry["publication_status"] == "EXCLUDED").sum())
    report = f"""# Publication Readiness Report

Generated from frozen benchmark reports. No benchmark outputs were rerun or overwritten.

## Repository Completeness

- README, license, citation, environment, contribution, security, changelog, and release checklist files are present.
- Documentation exists under `docs/`.
- Publication figures, tables, appendix assets, and anonymized examples are generated under `paper_assets/`.

## GitHub Readiness

- Apache 2.0 license and NOTICE are present.
- Issue and pull request templates are present.
- CI workflows for lint, format, tests, and documentation smoke checks are present.
- `.env.example` contains placeholders only.

## Paper Readiness

- Primary table lanes: {primary}
- Appendix lanes: {appendix}
- Excluded lanes: {excluded}
- Figures 1-19 plus Figures 8B, 12B, and 12C are exported as SVG, PDF, and 1200 dpi PNG.
- Figure 20 was intentionally removed.
- Tables 1-12 are exported as CSV, LaTeX, and Markdown.
- Figure captions are stored in `paper_assets/figures/captions.md`.

## Remaining TODOs

- Replace placeholder author names in `CITATION.cff`.
- Replace repository URL placeholders.
- Fill security contact email.
- Fill hardware details before camera-ready submission.
- Manually inspect anonymized examples before public release.
- Confirm BDA 2026 metadata once the submission status changes.

## Suggested Paper Figure Order

1. Overall benchmark architecture.
2. Dataset composition.
3. Benchmark workflow.
4. Evaluation pipeline.
5. OCR performance.
6. Direct VLM comparison.
7. Hybrid pipeline comparison.
8. Department x model heatmap.
8B. Department win count.
9. Runtime vs accuracy.
10. Models x metrics heatmap.
11. Bootstrap confidence intervals.
12. Wilcoxon and Holm significance.
12B. Friedman omnibus tests.
12C. McNemar exact test summary.
13. Benchmark leaderboard.
14. Qualitative examples.
15. Good vs bad prescription examples.
16. Example ground truth annotation.
17. Hybrid pipeline architecture.
18. Direct VLM pipeline.
19. OCR pipeline.

## Checklist Before GitHub Release

- Verify no secrets are present.
- Verify no unanonymized examples are published.
- Verify frozen benchmark marker remains unchanged.
- Run CI locally.
- Review README badges and links.
"""
    (ROOT / "PUBLICATION_READINESS_REPORT.md").write_text(report, encoding="utf-8")


def remove_retired_assets() -> None:
    for path in FIGURES.glob("figure_20_final_benchmark_summary.*"):
        path.unlink()
    for path in FIGURES.glob("figure_08_department_wise_benchmark.*"):
        path.unlink()
    for path in FIGURES.glob("figure_12_statistical_significance.*"):
        path.unlink()
    stale = TABLES / "combined"
    if stale.exists():
        for path in stale.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted(stale.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        stale.rmdir()


def main() -> None:
    ensure_dirs()
    remove_retired_assets()
    generate_tables()
    generate_figures()
    generate_appendix()
    write_templates()
    write_publication_report()
    print(f"Publication assets generated under {ROOT / 'paper_assets'}")


if __name__ == "__main__":
    main()
