"""Generate Figure 1: participant flow and ID-isolated validation split.

This script only reads existing summary tables and writes publication figure
assets, caption text, and a generation log under output/submission_assets_v2/.
It does not modify analysis data or results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

ROOT = Path(__file__).resolve().parents[2]

SAMPLE_FLOW = ROOT / "output/tables/07_sample_flow_summary.csv"
OVERLAP_AUDIT = ROOT / "output/tables/21_wave_overlap_audit_summary.csv"
ID_ISOLATED_FLOW = ROOT / "output/tables/21_id_isolated_sample_flow.csv"

ASSET_ROOT = ROOT / "output/submission_assets_v2"
FIGURE_PNG = ASSET_ROOT / "main_figures/Figure1_participant_flow_id_isolated.png"
FIGURE_PDF = ASSET_ROOT / "main_figures/Figure1_participant_flow_id_isolated.pdf"
CAPTION_OUT = ASSET_ROOT / "Figure1_caption_publication_legend.md"
LOG_OUT = ASSET_ROOT / "Figure1_generation_summary.txt"


def fmt_n(value: int | float) -> str:
    return f"{int(value):,}"


def get_sample_metric(df: pd.DataFrame, wave: str, metric: str) -> int:
    row = df[(df["wave"] == wave) & (df["metric"] == metric)]
    if row.empty:
        raise ValueError(f"Missing sample flow metric: {wave}/{metric}")
    return int(row["value"].iloc[0])


def get_id_metric(df: pd.DataFrame, dataset: str, metric: str) -> int:
    row = df[(df["dataset"] == dataset) & (df["metric"] == metric)]
    if row.empty:
        raise ValueError(f"Missing id-isolated sample flow metric: {dataset}/{metric}")
    return int(row["value"].iloc[0])


def draw_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str = "#333333",
    fontsize: int = 10,
    weight: str = "normal",
    linewidth: float = 0.95,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111111",
        fontweight=weight,
        linespacing=1.25,
    )


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = "#333333") -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.05, color=color, shrinkA=4, shrinkB=4),
    )


def main() -> None:
    for path in [SAMPLE_FLOW, OVERLAP_AUDIT, ID_ISOLATED_FLOW]:
        if not path.exists():
            raise FileNotFoundError(path)

    sample = pd.read_csv(SAMPLE_FLOW)
    overlap = pd.read_csv(OVERLAP_AUDIT)
    isolated = pd.read_csv(ID_ISOLATED_FLOW)

    w1_total = get_sample_metric(sample, "wave1", "age_ge_60_total")
    w1_eligible = get_sample_metric(sample, "wave1", "label_eligible_n")
    w1_ineligible = get_sample_metric(sample, "wave1", "label_ineligible_n")
    w1_pos = get_sample_metric(sample, "wave1", "possible_sarcopenia_positive_n")
    w1_neg = get_sample_metric(sample, "wave1", "possible_sarcopenia_negative_n")

    w3_total = get_sample_metric(sample, "wave3", "age_ge_60_total")
    w3_eligible = get_sample_metric(sample, "wave3", "label_eligible_n")
    w3_ineligible = get_sample_metric(sample, "wave3", "label_ineligible_n")
    w3_pos = get_sample_metric(sample, "wave3", "possible_sarcopenia_positive_n")
    w3_neg = get_sample_metric(sample, "wave3", "possible_sarcopenia_negative_n")

    main_overlap = overlap[overlap["scope"] == "05_full_analysis_base_current_main_data"]
    if main_overlap.empty:
        raise ValueError("Missing overlap scope: 05_full_analysis_base_current_main_data")
    overlap_id_n = int(main_overlap["overlap_id_n"].iloc[0])
    overlap_w1 = float(main_overlap["overlap_fraction_of_wave1"].iloc[0])
    overlap_w3 = float(main_overlap["overlap_fraction_of_wave3"].iloc[0])

    iso_w3_total = get_id_metric(isolated, "wave3_id_isolated", "age_ge_60_total")
    iso_w3_eligible = get_id_metric(isolated, "wave3_id_isolated", "label_eligible_n")
    iso_w3_ineligible = get_id_metric(isolated, "wave3_id_isolated", "label_ineligible_n")
    iso_w3_pos = get_id_metric(isolated, "wave3_id_isolated", "possible_sarcopenia_positive_n")
    iso_w3_neg = get_id_metric(isolated, "wave3_id_isolated", "possible_sarcopenia_negative_n")
    remaining_overlap = get_id_metric(isolated, "id_isolation", "remaining_overlap_id_n")

    FIGURE_PNG.parent.mkdir(parents=True, exist_ok=True)
    CAPTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    LOG_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.7, 4.65), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0.35, 7.7)
    ax.axis("off")

    # Grayscale publication palette.
    white = "#ffffff"
    light_grey = "#f2f2f2"
    line = "#2f2f2f"

    blue = white
    green = white
    amber = light_grey
    grey = light_grey

    ax.text(2.25, 7.38, "Wave 1 development source", ha="center", fontsize=8.8, fontweight="bold")
    ax.text(9.25, 7.38, "Wave 3 validation source", ha="center", fontsize=8.8, fontweight="bold")

    draw_box(ax, (0.7, 6.35), 3.1, 0.72, f"Wave 1 age ≥60 years\nn = {fmt_n(w1_total)}", blue, fontsize=7.8)
    draw_box(
        ax,
        (0.7, 5.15),
        3.1,
        0.9,
        f"Possible sarcopenia\nstatus classifiable\nn = {fmt_n(w1_eligible)}\npositive {fmt_n(w1_pos)}; negative {fmt_n(w1_neg)}",
        green,
        fontsize=6.9,
    )
    draw_box(ax, (0.7, 4.05), 3.1, 0.72, f"Possible sarcopenia status\nnot classifiable\nn = {fmt_n(w1_ineligible)}", grey, fontsize=6.9)

    draw_box(ax, (7.7, 6.35), 3.1, 0.72, f"Wave 3 age ≥60 years\nn = {fmt_n(w3_total)}", blue, fontsize=7.8)
    draw_box(
        ax,
        (7.7, 5.15),
        3.1,
        0.9,
        f"Possible sarcopenia\nstatus classifiable\nn = {fmt_n(w3_eligible)}\npositive {fmt_n(w3_pos)}; negative {fmt_n(w3_neg)}",
        green,
        fontsize=6.9,
    )
    draw_box(ax, (7.7, 4.05), 3.1, 0.72, f"Possible sarcopenia status\nnot classifiable\nn = {fmt_n(w3_ineligible)}", grey, fontsize=6.9)

    draw_arrow(ax, (2.25, 6.35), (2.25, 6.05), line)
    draw_arrow(ax, (2.25, 5.15), (2.25, 4.77), line)
    draw_arrow(ax, (9.25, 6.35), (9.25, 6.05), line)
    draw_arrow(ax, (9.25, 5.15), (9.25, 4.77), line)

    draw_box(
        ax,
        (4.15, 5.55),
        3.2,
        1.18,
        (
            f"Participant ID overlap\n"
            f"n = {fmt_n(overlap_id_n)}\n"
            f"{overlap_w1:.1%} of wave 1\n"
            f"{overlap_w3:.1%} of wave 3"
        ),
        amber,
        fontsize=6.8,
        weight="bold",
    )
    draw_arrow(ax, (3.8, 6.7), (4.15, 6.30), line)
    draw_arrow(ax, (7.7, 6.7), (7.35, 6.30), line)

    draw_box(
        ax,
        (4.15, 3.95),
        3.2,
        0.95,
        (
            f"Remove wave 3 records\n"
            f"with wave 1 IDs\n"
            f"removed n = {fmt_n(overlap_id_n)}"
        ),
        grey,
        fontsize=6.9,
    )
    draw_arrow(ax, (5.75, 5.55), (5.75, 4.90), line)

    draw_box(
        ax,
        (7.7, 2.75),
        3.1,
        0.82,
        f"ID-isolated wave 3\nage ≥60 years, n = {fmt_n(iso_w3_total)}",
        blue,
        fontsize=7.6,
        weight="bold",
        linewidth=1.4,
    )
    draw_box(
        ax,
        (7.7, 1.45),
        3.1,
        0.95,
        (
            f"Possible sarcopenia\nstatus classifiable\nn = {fmt_n(iso_w3_eligible)}\n"
            f"positive {fmt_n(iso_w3_pos)}; negative {fmt_n(iso_w3_neg)}"
        ),
        green,
        fontsize=6.8,
        weight="bold",
        linewidth=1.4,
    )
    draw_box(ax, (7.7, 0.45), 3.1, 0.62, f"Possible sarcopenia status\nnot classifiable\nn = {fmt_n(iso_w3_ineligible)}", grey, fontsize=6.8)

    draw_arrow(ax, (7.35, 4.42), (8.0, 3.57), line)
    draw_arrow(ax, (9.25, 2.75), (9.25, 2.4), line)
    draw_arrow(ax, (9.25, 1.45), (9.25, 1.07), line)

    fig.tight_layout(pad=0.18)
    fig.savefig(FIGURE_PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption = f"""# Figure 1 图题与图注建议

**图题：** 研究对象筛选流程与 ID 隔离后的严格跨波次验证样本构建

**图注：** Wave 1 中年龄 ≥60 岁样本共 {fmt_n(w1_total)} 例，其中 {fmt_n(w1_eligible)} 例可构建 possible sarcopenia 主标签。Wave 3 原始年龄 ≥60 岁样本共 {fmt_n(w3_total)} 例，其中 {fmt_n(w3_eligible)} 例可构建主标签；wave 1 与 wave 3 在当前主分析底板中存在 {fmt_n(overlap_id_n)} 个重叠 ID。为形成更严格的跨波次验证版本，本研究从 wave 3 中剔除所有已在 wave 1 出现的 ID，得到 id-isolated wave 3 样本 {fmt_n(iso_w3_total)} 例，其中 {fmt_n(iso_w3_eligible)} 例可用于主标签验证，最终与 wave 1 的 ID 重叠为 {fmt_n(remaining_overlap)}。Possible sarcopenia status classifiable required sufficient handgrip or chair-stand information to construct the screening outcome.
"""
    CAPTION_OUT.write_text(caption, encoding="utf-8")

    log = {
        "stage": "Figure 1 participant flow id-isolated generation",
        "inputs": {
            "sample_flow": str(SAMPLE_FLOW.relative_to(ROOT)),
            "overlap_audit": str(OVERLAP_AUDIT.relative_to(ROOT)),
            "id_isolated_flow": str(ID_ISOLATED_FLOW.relative_to(ROOT)),
        },
        "outputs": {
            "figure_png": str(FIGURE_PNG.relative_to(ROOT)),
            "figure_pdf": str(FIGURE_PDF.relative_to(ROOT)),
            "caption": str(CAPTION_OUT.relative_to(ROOT)),
        },
        "values": {
            "wave1_age_ge_60_total": w1_total,
            "wave1_label_eligible": w1_eligible,
            "wave1_label_ineligible": w1_ineligible,
            "wave1_positive": w1_pos,
            "wave1_negative": w1_neg,
            "wave3_age_ge_60_total_original": w3_total,
            "wave3_label_eligible_original": w3_eligible,
            "wave3_label_ineligible_original": w3_ineligible,
            "wave3_positive_original": w3_pos,
            "wave3_negative_original": w3_neg,
            "overlap_id_n": overlap_id_n,
            "overlap_fraction_wave1": overlap_w1,
            "overlap_fraction_wave3": overlap_w3,
            "wave3_age_ge_60_total_id_isolated": iso_w3_total,
            "wave3_label_eligible_id_isolated": iso_w3_eligible,
            "wave3_label_ineligible_id_isolated": iso_w3_ineligible,
            "wave3_positive_id_isolated": iso_w3_pos,
            "wave3_negative_id_isolated": iso_w3_neg,
            "remaining_overlap_id_n": remaining_overlap,
        },
        "notes": [
            "No existing results or manuscripts were modified.",
            "Figure title is intentionally excluded from the image and provided in the caption file.",
        ],
    }
    LOG_OUT.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {FIGURE_PDF.relative_to(ROOT)}")
    print(f"Wrote {FIGURE_PNG.relative_to(ROOT)}")
    print(f"Wrote {CAPTION_OUT.relative_to(ROOT)}")
    print(f"Wrote {LOG_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
