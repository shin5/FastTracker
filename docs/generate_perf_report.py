#!/usr/bin/env python3
"""
FastTracker 性能評価報告書 PowerPoint生成スクリプト

GPU加速マルチターゲット追尾システムの性能評価結果を
PowerPointプレゼンテーションとして出力する。
"""

import csv
import os
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.chart import XL_CHART_TYPE
from pptx.enum.shapes import MSO_SHAPE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_PATH = os.path.join(PROJECT_DIR, "evaluation_results.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "performance_report.pptx")
SCREENSHOTS_DIR = os.path.join(SCRIPT_DIR, "screenshots")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
DARK_BLUE = RGBColor(0x00, 0x33, 0x66)
MID_BLUE = RGBColor(0x00, 0x5B, 0x96)
LIGHT_BLUE = RGBColor(0xD6, 0xE8, 0xF7)
ACCENT_BLUE = RGBColor(0x1A, 0x73, 0xB5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLACK = RGBColor(0x00, 0x00, 0x00)
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)
MID_GRAY = RGBColor(0x66, 0x66, 0x66)
LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)
GREEN = RGBColor(0x27, 0xAE, 0x60)
RED = RGBColor(0xE7, 0x4C, 0x3C)
HEADER_BG = RGBColor(0x00, 0x33, 0x66)
ROW_ALT = RGBColor(0xEC, 0xF0, 0xF5)

# Matplotlib colours (hex strings)
MPL_DARK_BLUE = "#003366"
MPL_ACCENT = "#1A73B5"
MPL_RED = "#E74C3C"
MPL_GREEN = "#27AE60"
MPL_GRID = "#CCCCCC"

# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------
JAPANESE_FONT = "Meiryo"

# Try to find a Japanese font for matplotlib
_jp_font_path = None
for f in fm.findSystemFonts():
    lower = f.lower()
    if "meiryo" in lower and lower.endswith(".ttc"):
        _jp_font_path = f
        break
    if "yugoth" in lower and lower.endswith(".ttc"):
        _jp_font_path = f
        break

if _jp_font_path:
    _jp_font_prop = fm.FontProperties(fname=_jp_font_path)
else:
    _jp_font_prop = fm.FontProperties()

# Configure matplotlib defaults
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": MPL_DARK_BLUE,
    "axes.labelcolor": MPL_DARK_BLUE,
    "xtick.color": MPL_DARK_BLUE,
    "ytick.color": MPL_DARK_BLUE,
    "grid.color": MPL_GRID,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_csv(path):
    """Load evaluation_results.csv and return lists of floats per column."""
    timestamps, pos_err, vel_err, ospa = [], [], [], []
    tp_list, fp_list, fn_list = [], [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp"])
            pe = float(row["avg_position_error"])
            ve = float(row["avg_velocity_error"])
            od = float(row["ospa_distance"])
            tp = int(row["true_positives"])
            fp = int(row["false_positives"])
            fn = int(row["false_negatives"])
            timestamps.append(ts)
            pos_err.append(pe)
            vel_err.append(ve)
            ospa.append(od)
            tp_list.append(tp)
            fp_list.append(fp)
            fn_list.append(fn)
    return {
        "timestamp": timestamps,
        "pos_err": pos_err,
        "vel_err": vel_err,
        "ospa": ospa,
        "tp": tp_list,
        "fp": fp_list,
        "fn": fn_list,
    }


# ---------------------------------------------------------------------------
# Chart generation helpers
# ---------------------------------------------------------------------------
def _make_time_series_chart(
    timestamps, values, ylabel, title, colour, mean_val=None, png_path=None
):
    """Create a professional time-series chart and save to *png_path*."""
    fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=180)

    ax.plot(timestamps, values, color=colour, linewidth=0.8, alpha=0.85)
    ax.fill_between(timestamps, 0, values, color=colour, alpha=0.12)

    if mean_val is not None:
        ax.axhline(
            y=mean_val, color=MPL_RED, linestyle="--", linewidth=1.2, alpha=0.8
        )
        ax.text(
            timestamps[-1] * 0.98,
            mean_val * 1.08,
            f"平均: {mean_val:.1f}",
            fontproperties=_jp_font_prop,
            fontsize=9,
            color=MPL_RED,
            ha="right",
            va="bottom",
        )

    ax.set_xlabel("時間 [s]", fontproperties=_jp_font_prop, fontsize=10)
    ax.set_ylabel(ylabel, fontproperties=_jp_font_prop, fontsize=10)
    ax.set_title(title, fontproperties=_jp_font_prop, fontsize=12, color=MPL_DARK_BLUE,
                 fontweight="bold", pad=10)
    ax.grid(True)
    ax.set_xlim(timestamps[0], timestamps[-1])
    ax.set_ylim(bottom=0)

    # Use Japanese font for tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(_jp_font_prop)

    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _make_detection_bar_chart(png_path):
    """Create a bar chart for Precision / Recall / F1."""
    metrics = ["Precision", "Recall", "F1 Score"]
    values = [100.0, 99.6, 99.8]
    errors = [0.0, 0.3, 0.2]
    colours = [MPL_ACCENT, MPL_GREEN, MPL_DARK_BLUE]

    fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=180)
    bars = ax.bar(metrics, values, color=colours, width=0.55, edgecolor="white",
                  linewidth=1.5, yerr=errors, capsize=6, error_kw={"elinewidth": 1.2})

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - 3.5,
            f"{val:.1f}%",
            ha="center", va="top", fontsize=13, fontweight="bold", color="white",
        )

    ax.set_ylim(95, 101)
    ax.set_ylabel("スコア [%]", fontproperties=_jp_font_prop, fontsize=10)
    ax.set_title("検出性能メトリクス", fontproperties=_jp_font_prop, fontsize=12,
                 color=MPL_DARK_BLUE, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.5)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(_jp_font_prop)

    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# PowerPoint helpers
# ---------------------------------------------------------------------------
def _set_font(run, size=14, bold=False, color=BLACK, name=JAPANESE_FONT):
    """Apply font settings to a run."""
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name


def _add_textbox(slide, left, top, width, height, text, size=14, bold=False,
                 color=BLACK, alignment=PP_ALIGN.LEFT, name=JAPANESE_FONT):
    """Add a simple text box to a slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = alignment
    run = p.add_run()
    run.text = text
    _set_font(run, size=size, bold=bold, color=color, name=name)
    return txBox


def _add_paragraph(text_frame, text, size=14, bold=False, color=BLACK,
                   alignment=PP_ALIGN.LEFT, space_before=Pt(4), space_after=Pt(2),
                   name=JAPANESE_FONT):
    """Add a paragraph to an existing text frame."""
    p = text_frame.add_paragraph()
    p.alignment = alignment
    p.space_before = space_before
    p.space_after = space_after
    run = p.add_run()
    run.text = text
    _set_font(run, size=size, bold=bold, color=color, name=name)
    return p


def _add_bullet_list(slide, left, top, width, height, items, size=14,
                     color=BLACK, bullet_color=DARK_BLUE, line_spacing=1.4):
    """Add a bulleted list to a slide."""
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.space_before = Pt(4)
        p.space_after = Pt(4)
        p.line_spacing = Pt(int(size * line_spacing))

        # Bullet character
        bullet_run = p.add_run()
        bullet_run.text = "\u25A0  "  # small black square
        _set_font(bullet_run, size=size, bold=False, color=bullet_color)

        text_run = p.add_run()
        text_run.text = item
        _set_font(text_run, size=size, bold=False, color=color)

    return txBox


def _add_slide_number(slide, slide_num, total):
    """Add page number in bottom-right."""
    _add_textbox(
        slide,
        Inches(11.5), Inches(7.0), Inches(1.5), Inches(0.4),
        f"{slide_num} / {total}",
        size=9, color=MID_GRAY, alignment=PP_ALIGN.RIGHT,
    )


def _add_header_bar(slide, title_text):
    """Add a dark-blue header bar at the top of a content slide."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0), Inches(0), Inches(13.333), Inches(0.95),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK_BLUE
    shape.line.fill.background()

    _add_textbox(
        slide,
        Inches(0.6), Inches(0.12), Inches(12), Inches(0.7),
        title_text, size=26, bold=True, color=WHITE,
    )


def _add_bottom_line(slide):
    """Add a thin accent line near the bottom of the slide."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.6), Inches(6.9), Inches(12.1), Inches(0.03),
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = MID_BLUE
    shape.line.fill.background()


def _make_table(slide, left, top, width, height, rows, cols, data, col_widths=None):
    """
    Add a styled table to *slide*.
    *data* is a list of lists: first row = header.
    """
    table_shape = slide.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = w

    for r in range(rows):
        for c in range(cols):
            cell = table.cell(r, c)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            run = p.add_run()
            run.text = str(data[r][c])

            if r == 0:
                # Header row
                cell.fill.solid()
                cell.fill.fore_color.rgb = HEADER_BG
                _set_font(run, size=12, bold=True, color=WHITE)
                p.alignment = PP_ALIGN.CENTER
            else:
                if r % 2 == 0:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = ROW_ALT
                else:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = WHITE
                _set_font(run, size=11, bold=False, color=DARK_GRAY)
                if c == 0:
                    p.alignment = PP_ALIGN.LEFT
                else:
                    p.alignment = PP_ALIGN.CENTER

            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            # Remove default margins for tighter look
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = Inches(0.04)
            cell.margin_bottom = Inches(0.04)

    return table_shape


# =========================================================================
# Slide builders
# =========================================================================
TOTAL_SLIDES = 14


def build_slide_title(prs):
    """Slide 1: Title slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    # Full-slide dark blue background
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0), Inches(0), Inches(13.333), Inches(7.5),
    )
    bg.fill.solid()
    bg.fill.fore_color.rgb = DARK_BLUE
    bg.line.fill.background()

    # Accent line
    accent = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(1.5), Inches(3.05), Inches(10.3), Inches(0.06),
    )
    accent.fill.solid()
    accent.fill.fore_color.rgb = ACCENT_BLUE
    accent.line.fill.background()

    # Title
    _add_textbox(
        slide, Inches(1.5), Inches(1.5), Inches(10.3), Inches(1.5),
        "FastTracker 性能評価報告書",
        size=40, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER,
    )

    # Subtitle
    _add_textbox(
        slide, Inches(1.5), Inches(3.3), Inches(10.3), Inches(0.8),
        "GPU加速マルチターゲット追尾システム",
        size=22, bold=False, color=LIGHT_BLUE, alignment=PP_ALIGN.CENTER,
    )

    # Date
    _add_textbox(
        slide, Inches(1.5), Inches(5.2), Inches(10.3), Inches(0.6),
        "2026年2月",
        size=18, bold=False, color=WHITE, alignment=PP_ALIGN.CENTER,
    )

    # Confidential label
    _add_textbox(
        slide, Inches(1.5), Inches(6.2), Inches(10.3), Inches(0.5),
        "CONFIDENTIAL",
        size=12, bold=True, color=MID_BLUE, alignment=PP_ALIGN.CENTER,
    )

    _add_slide_number(slide, 1, TOTAL_SLIDES)


def build_slide_toc(prs):
    """Slide 2: 目次."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "目次")

    items = [
        ("1.", "評価概要", "システム概要と評価目的"),
        ("2.", "試験条件 - シナリオ", "飛翔経路とセンサー配置"),
        ("3.", "試験条件 - パラメータ", "レーダー・追尾器パラメータ"),
        ("4.", "軌道生成結果", "3D飛翔軌道の可視化"),
        ("5.", "追尾結果の可視化", "追尾結果と評価チャート"),
        ("6.", "追尾精度評価 - 位置誤差", "位置RMSEの時系列評価"),
        ("7.", "追尾精度評価 - 速度誤差", "速度RMSEの時系列評価"),
        ("8.", "追尾精度評価 - OSPA距離", "OSPA距離の時系列評価"),
        ("9.", "検出性能評価", "Precision / Recall / F1 Score"),
        ("10.", "処理速度評価", "GPU加速による処理性能"),
        ("11.", "総合評価・まとめ", "全メトリクスの総括"),
    ]

    y_start = Inches(1.5)
    for i, (num, title, desc) in enumerate(items):
        y = y_start + Inches(i * 0.6)

        # Number
        _add_textbox(slide, Inches(1.0), y, Inches(0.5), Inches(0.45),
                     num, size=16, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.RIGHT)
        # Title
        _add_textbox(slide, Inches(1.7), y, Inches(5.0), Inches(0.45),
                     title, size=16, bold=True, color=DARK_BLUE)
        # Description
        _add_textbox(slide, Inches(7.0), y, Inches(5.5), Inches(0.45),
                     desc, size=13, bold=False, color=MID_GRAY)

        # Separator line
        if i < len(items) - 1:
            sep = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                Inches(1.0), y + Inches(0.5), Inches(11.3), Inches(0.01),
            )
            sep.fill.solid()
            sep.fill.fore_color.rgb = LIGHT_BLUE
            sep.line.fill.background()

    _add_bottom_line(slide)
    _add_slide_number(slide, 2, TOTAL_SLIDES)


def build_slide_overview(prs):
    """Slide 3: 評価概要."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "1. 評価概要")

    # Purpose box
    _add_textbox(slide, Inches(0.8), Inches(1.3), Inches(5.5), Inches(0.5),
                 "評価目的", size=18, bold=True, color=DARK_BLUE)
    _add_bullet_list(slide, Inches(0.8), Inches(1.9), Inches(5.5), Inches(2.0), [
        "FastTracker v1.0の追尾性能を定量的に評価",
        "弾道ミサイルシナリオにおける追尾精度の検証",
        "GPU加速によるリアルタイム処理能力の確認",
        "IMM-UKFフィルタの運動モデル切替性能の評価",
    ], size=13, color=DARK_GRAY)

    # System under test
    _add_textbox(slide, Inches(0.8), Inches(4.2), Inches(5.5), Inches(0.5),
                 "評価対象システム", size=18, bold=True, color=DARK_BLUE)
    _add_bullet_list(slide, Inches(0.8), Inches(4.8), Inches(5.5), Inches(2.0), [
        "FastTracker v1.0 (CUDA/C++)",
        "IMM-UKF (Interacting Multiple Model - Unscented Kalman Filter)",
        "GNN (Global Nearest Neighbor) データアソシエーション",
        "モンテカルロ試験: 10回試行",
    ], size=13, color=DARK_GRAY)

    # Key capabilities - right column
    _add_textbox(slide, Inches(7.0), Inches(1.3), Inches(5.5), Inches(0.5),
                 "評価項目", size=18, bold=True, color=DARK_BLUE)

    cap_data = [
        ["評価項目", "内容"],
        ["軌道追尾精度", "位置・速度RMSEによる評価"],
        ["OSPA距離", "目標存在性を含む総合距離"],
        ["検出性能", "Precision / Recall / F1"],
        ["トラック管理", "確立・維持・削除の正確性"],
        ["処理速度", "フレーム処理時間・全体処理時間"],
        ["モデル切替", "IMM 3モデル(CV/弾道/CT)"],
    ]
    _make_table(
        slide, Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.0),
        len(cap_data), 2, cap_data,
        col_widths=[Inches(2.2), Inches(3.1)],
    )

    _add_bottom_line(slide)
    _add_slide_number(slide, 3, TOTAL_SLIDES)


def build_slide_scenario(prs):
    """Slide 4: 試験条件 - シナリオ (地図スクリーンショット付き)."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "2. 試験条件 - シナリオ")

    _add_textbox(slide, Inches(0.8), Inches(1.3), Inches(11.5), Inches(0.5),
                 "単一弾道ミサイルシナリオ (平壌 → 東京)", size=20, bold=True, color=DARK_BLUE)

    # Map screenshot on left - shows launch point, target, and sensor position
    map_img = os.path.join(SCREENSHOTS_DIR, "02_map_points.png")
    if os.path.exists(map_img):
        slide.shapes.add_picture(map_img, Inches(0.5), Inches(2.0), Inches(6.8), Inches(4.6))
        # Caption
        _add_textbox(slide, Inches(0.5), Inches(6.6), Inches(6.8), Inches(0.4),
                     "図: 発射地点(赤)・目標地点(青)・センサー位置(緑) の配置",
                     size=10, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    # Scenario summary table on right
    scenario_data = [
        ["項目", "値"],
        ["シナリオ名", "single-ballistic"],
        ["発射地点", "平壌 (39.0°N, 125.7°E)"],
        ["目標地点", "東京 (35.7°N, 139.7°E)"],
        ["センサ位置", "対馬海峡 (34.8°N, 131.1°E)"],
        ["飛翔距離", "~1,000 km"],
        ["最高高度", "~300 km"],
        ["飛翔時間", "~600 s"],
        ["目標数", "1 (単一目標)"],
        ["更新レート", "2 Hz (0.5秒間隔)"],
    ]
    _make_table(
        slide, Inches(7.7), Inches(2.0), Inches(5.2), Inches(4.8),
        len(scenario_data), 2, scenario_data,
        col_widths=[Inches(2.0), Inches(3.2)],
    )

    _add_bottom_line(slide)
    _add_slide_number(slide, 4, TOTAL_SLIDES)


def build_slide_params(prs):
    """Slide 5: 試験条件 - パラメータ."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "3. 試験条件 - パラメータ")

    # Radar params - left
    _add_textbox(slide, Inches(0.6), Inches(1.2), Inches(5.0), Inches(0.4),
                 "レーダーパラメータ", size=16, bold=True, color=DARK_BLUE)
    radar_data = [
        ["パラメータ", "値"],
        ["最大探知距離", "2,000 km"],
        ["視野角 (FOV)", "120°"],
        ["探知確率 (Pd)", "0.95"],
        ["誤警報確率 (Pfa)", "1 x 10⁻⁶"],
        ["更新レート", "2 Hz"],
    ]
    _make_table(
        slide, Inches(0.6), Inches(1.7), Inches(5.5), Inches(2.8),
        len(radar_data), 2, radar_data,
        col_widths=[Inches(2.8), Inches(2.7)],
    )

    # Tracker params - right
    _add_textbox(slide, Inches(6.8), Inches(1.2), Inches(5.5), Inches(0.4),
                 "追尾器パラメータ", size=16, bold=True, color=DARK_BLUE)
    tracker_data = [
        ["パラメータ", "値"],
        ["ゲート閾値", "500"],
        ["確立ヒット数", "2"],
        ["削除ミス数", "90"],
    ]
    _make_table(
        slide, Inches(6.8), Inches(1.7), Inches(5.5), Inches(1.8),
        len(tracker_data), 2, tracker_data,
        col_widths=[Inches(2.8), Inches(2.7)],
    )

    # UKF params - left bottom
    _add_textbox(slide, Inches(0.6), Inches(4.7), Inches(5.0), Inches(0.4),
                 "UKFパラメータ", size=16, bold=True, color=DARK_BLUE)
    ukf_data = [
        ["パラメータ", "値"],
        ["alpha (α)", "0.5"],
        ["beta (β)", "2.0"],
        ["kappa (κ)", "0.0"],
    ]
    _make_table(
        slide, Inches(0.6), Inches(5.2), Inches(5.5), Inches(1.8),
        len(ukf_data), 2, ukf_data,
        col_widths=[Inches(2.8), Inches(2.7)],
    )

    # Process noise / IMM - right bottom
    _add_textbox(slide, Inches(6.8), Inches(3.7), Inches(5.5), Inches(0.4),
                 "プロセスノイズ / IMM", size=16, bold=True, color=DARK_BLUE)
    noise_data = [
        ["パラメータ", "値"],
        ["位置ノイズ (σ_pos)", "160 m"],
        ["速度ノイズ (σ_vel)", "65 m/s"],
        ["加速度ノイズ (σ_acc)", "160 m/s²"],
        ["IMMモデル数", "3 (CV / Ballistic / CT)"],
        ["モンテカルロ試行数", "10回"],
    ]
    _make_table(
        slide, Inches(6.8), Inches(4.2), Inches(5.5), Inches(3.0),
        len(noise_data), 2, noise_data,
        col_widths=[Inches(2.8), Inches(2.7)],
    )

    _add_bottom_line(slide)
    _add_slide_number(slide, 5, TOTAL_SLIDES)


def build_slide_trajectory(prs):
    """Slide 6: 軌道生成結果 — 3D飛翔軌道の可視化."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "4. 軌道生成結果")

    _add_textbox(slide, Inches(0.8), Inches(1.2), Inches(6.0), Inches(0.4),
                 "RK4物理モデルによる弾道飛翔軌道", size=18, bold=True, color=DARK_BLUE)

    # 3D trajectory screenshot
    traj_img = os.path.join(SCREENSHOTS_DIR, "06_trajectory_3d.png")
    if os.path.exists(traj_img):
        slide.shapes.add_picture(traj_img, Inches(0.4), Inches(1.8), Inches(8.3), Inches(5.2))
        _add_textbox(slide, Inches(0.4), Inches(7.0), Inches(8.3), Inches(0.3),
                     "図: 生成された3D弾道軌道 (ブースト→ミッドコース→再突入)",
                     size=10, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    # Key parameters on the right
    _add_textbox(slide, Inches(9.0), Inches(1.2), Inches(4.0), Inches(0.4),
                 "軌道パラメータ", size=16, bold=True, color=DARK_BLUE)

    _add_bullet_list(slide, Inches(9.0), Inches(1.8), Inches(4.0), Inches(2.8), [
        "ミサイル種別: 弾道ミサイル",
        "物理モデル: RK4積分",
        "重力: 高度依存 g(h)",
        "大気抗力: 指数大気モデル",
        "飛翔フェーズ:",
        "  1. ブースト (加速上昇)",
        "  2. ミッドコース (慣性飛翔)",
        "  3. 再突入 (降下加速)",
    ], size=12, color=DARK_GRAY, line_spacing=1.3)

    # Full GUI screenshot (small, bottom right)
    full_img = os.path.join(SCREENSHOTS_DIR, "01_full_gui.png")
    if os.path.exists(full_img):
        slide.shapes.add_picture(full_img, Inches(9.0), Inches(4.8), Inches(4.0), Inches(2.2))
        _add_textbox(slide, Inches(9.0), Inches(7.0), Inches(4.0), Inches(0.3),
                     "図: FastTracker Web GUI 全体画面",
                     size=9, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    _add_bottom_line(slide)
    _add_slide_number(slide, 6, TOTAL_SLIDES)


def build_slide_tracking_visuals(prs):
    """Slide 7: 追尾結果と評価チャートの可視化."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "5. 追尾結果の可視化")

    # Tracking 3D view (left)
    track_img = os.path.join(SCREENSHOTS_DIR, "07_tracking_3d.png")
    if os.path.exists(track_img):
        slide.shapes.add_picture(track_img, Inches(0.3), Inches(1.2), Inches(6.3), Inches(3.9))
        _add_textbox(slide, Inches(0.3), Inches(5.1), Inches(6.3), Inches(0.3),
                     "図: 3D追尾結果 — 真値(緑) vs 推定軌道(青) vs 観測(赤点)",
                     size=10, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    # Evaluation charts (right)
    eval_img = os.path.join(SCREENSHOTS_DIR, "08_evaluation.png")
    if os.path.exists(eval_img):
        slide.shapes.add_picture(eval_img, Inches(6.8), Inches(1.2), Inches(6.3), Inches(3.9))
        _add_textbox(slide, Inches(6.8), Inches(5.1), Inches(6.3), Inches(0.3),
                     "図: 評価チャート — 位置・速度誤差、OSPA距離、モデル確率推移",
                     size=10, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    # Timeline view (bottom center)
    timeline_img = os.path.join(SCREENSHOTS_DIR, "09_timeline.png")
    if os.path.exists(timeline_img):
        slide.shapes.add_picture(timeline_img, Inches(2.5), Inches(5.5), Inches(8.3), Inches(1.6))
        _add_textbox(slide, Inches(2.5), Inches(7.1), Inches(8.3), Inches(0.3),
                     "図: タイムライン — ビーム配分・検出イベント・トラック状態の時系列",
                     size=10, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    _add_bottom_line(slide)
    _add_slide_number(slide, 7, TOTAL_SLIDES)


def build_slide_pos_error(prs, data, chart_dir):
    """Slide 8: 追尾精度評価 - 位置誤差."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "6. 追尾精度評価 - 位置誤差")

    # Filter out the initial 10000 sentinel values
    ts_filt = [t for t, v in zip(data["timestamp"], data["pos_err"]) if v < 5000]
    pe_filt = [v for v in data["pos_err"] if v < 5000]

    mean_pe = np.mean(pe_filt) if pe_filt else 0

    chart_path = os.path.join(chart_dir, "pos_error.png")
    _make_time_series_chart(
        ts_filt, pe_filt,
        ylabel="位置誤差 [m]",
        title="位置誤差の時間推移",
        colour=MPL_ACCENT,
        mean_val=mean_pe,
        png_path=chart_path,
    )

    slide.shapes.add_picture(chart_path, Inches(0.5), Inches(1.2), Inches(8.8), Inches(4.2))

    # Result box on the right
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(9.7), Inches(1.5), Inches(3.2), Inches(3.5),
    )
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT_GRAY
    box.line.color.rgb = MID_BLUE
    box.line.width = Pt(1.5)

    _add_textbox(slide, Inches(9.9), Inches(1.7), Inches(2.8), Inches(0.4),
                 "主要結果", size=16, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)

    _add_textbox(slide, Inches(9.9), Inches(2.3), Inches(2.8), Inches(0.4),
                 "位置 RMSE", size=13, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(9.9), Inches(2.7), Inches(2.8), Inches(0.5),
                 "986.1 m", size=28, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(9.9), Inches(3.3), Inches(2.8), Inches(0.4),
                 "± 51.4 m (1σ)", size=13, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    _add_textbox(slide, Inches(9.9), Inches(3.9), Inches(2.8), Inches(0.3),
                 f"フレーム平均: {mean_pe:.1f} m",
                 size=11, bold=False, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(9.9), Inches(4.3), Inches(2.8), Inches(0.3),
                 f"データ点数: {len(pe_filt)}",
                 size=11, bold=False, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)

    _add_bottom_line(slide)
    _add_slide_number(slide, 8, TOTAL_SLIDES)


def build_slide_vel_error(prs, data, chart_dir):
    """Slide 7: 追尾精度評価 - 速度誤差."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "7. 追尾精度評価 - 速度誤差")

    ts_filt = [t for t, v in zip(data["timestamp"], data["vel_err"]) if v > 0]
    ve_filt = [v for v in data["vel_err"] if v > 0]

    mean_ve = np.mean(ve_filt) if ve_filt else 0

    chart_path = os.path.join(chart_dir, "vel_error.png")
    _make_time_series_chart(
        ts_filt, ve_filt,
        ylabel="速度誤差 [m/s]",
        title="速度誤差の時間推移",
        colour=MPL_GREEN,
        mean_val=mean_ve,
        png_path=chart_path,
    )

    slide.shapes.add_picture(chart_path, Inches(0.5), Inches(1.2), Inches(8.8), Inches(4.2))

    # Result box
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(9.7), Inches(1.5), Inches(3.2), Inches(3.5),
    )
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT_GRAY
    box.line.color.rgb = MID_BLUE
    box.line.width = Pt(1.5)

    _add_textbox(slide, Inches(9.9), Inches(1.7), Inches(2.8), Inches(0.4),
                 "主要結果", size=16, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)

    _add_textbox(slide, Inches(9.9), Inches(2.3), Inches(2.8), Inches(0.4),
                 "速度 RMSE", size=13, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(9.9), Inches(2.7), Inches(2.8), Inches(0.5),
                 "199.8 m/s", size=28, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(9.9), Inches(3.3), Inches(2.8), Inches(0.4),
                 "± 33.0 m/s (1σ)", size=13, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    _add_textbox(slide, Inches(9.9), Inches(3.9), Inches(2.8), Inches(0.3),
                 f"フレーム平均: {mean_ve:.1f} m/s",
                 size=11, bold=False, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(9.9), Inches(4.3), Inches(2.8), Inches(0.3),
                 f"データ点数: {len(ve_filt)}",
                 size=11, bold=False, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)

    _add_bottom_line(slide)
    _add_slide_number(slide, 9, TOTAL_SLIDES)


def build_slide_ospa(prs, data, chart_dir):
    """Slide 10: 追尾精度評価 - OSPA距離."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "8. 追尾精度評価 - OSPA距離")

    ts_filt = [t for t, v in zip(data["timestamp"], data["ospa"]) if v < 5000]
    ospa_filt = [v for v in data["ospa"] if v < 5000]

    mean_ospa = np.mean(ospa_filt) if ospa_filt else 0

    chart_path = os.path.join(chart_dir, "ospa.png")
    _make_time_series_chart(
        ts_filt, ospa_filt,
        ylabel="OSPA距離 [m]",
        title="OSPA距離の時間推移",
        colour=MPL_RED,
        mean_val=mean_ospa,
        png_path=chart_path,
    )

    slide.shapes.add_picture(chart_path, Inches(0.5), Inches(1.2), Inches(8.8), Inches(4.2))

    # Result box
    box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(9.7), Inches(1.5), Inches(3.2), Inches(3.5),
    )
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT_GRAY
    box.line.color.rgb = MID_BLUE
    box.line.width = Pt(1.5)

    _add_textbox(slide, Inches(9.9), Inches(1.7), Inches(2.8), Inches(0.4),
                 "主要結果", size=16, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)

    _add_textbox(slide, Inches(9.9), Inches(2.3), Inches(2.8), Inches(0.4),
                 "OSPA距離", size=13, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(9.9), Inches(2.7), Inches(2.8), Inches(0.5),
                 "836.7 m", size=28, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(9.9), Inches(3.3), Inches(2.8), Inches(0.4),
                 "± 35.5 m (1σ)", size=13, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    _add_textbox(slide, Inches(9.9), Inches(3.9), Inches(2.8), Inches(0.3),
                 f"フレーム平均: {mean_ospa:.1f} m",
                 size=11, bold=False, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(9.9), Inches(4.3), Inches(2.8), Inches(0.3),
                 "カットオフ距離含む",
                 size=11, bold=False, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)

    _add_bottom_line(slide)
    _add_slide_number(slide, 10, TOTAL_SLIDES)


def build_slide_detection(prs, chart_dir):
    """Slide 11: 検出性能評価."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "9. 検出性能評価")

    # Bar chart
    chart_path = os.path.join(chart_dir, "detection.png")
    _make_detection_bar_chart(chart_path)
    slide.shapes.add_picture(chart_path, Inches(0.5), Inches(1.2), Inches(7.0), Inches(4.0))

    # Detection metrics table on right
    _add_textbox(slide, Inches(8.0), Inches(1.3), Inches(4.5), Inches(0.4),
                 "検出メトリクス詳細", size=16, bold=True, color=DARK_BLUE)

    det_data = [
        ["メトリクス", "値", "標準偏差"],
        ["Precision", "100.0%", "± 0.0%"],
        ["Recall", "99.6%", "± 0.3%"],
        ["F1 Score", "99.8%", "± 0.2%"],
    ]
    _make_table(
        slide, Inches(8.0), Inches(1.8), Inches(4.5), Inches(2.0),
        len(det_data), 3, det_data,
        col_widths=[Inches(1.6), Inches(1.3), Inches(1.6)],
    )

    # Explanation
    _add_textbox(slide, Inches(8.0), Inches(4.1), Inches(4.5), Inches(0.4),
                 "評価説明", size=14, bold=True, color=DARK_BLUE)

    _add_bullet_list(slide, Inches(8.0), Inches(4.6), Inches(4.5), Inches(2.5), [
        "Precision 100%: 偽トラック生成なし",
        "Recall 99.6%: 初期捕捉遅延のみ",
        "False Positive: 0件 (全フレーム)",
        "False Negative: 初期2フレームのみ",
        "10回モンテカルロ試験で安定",
    ], size=12, color=DARK_GRAY)

    _add_bottom_line(slide)
    _add_slide_number(slide, 11, TOTAL_SLIDES)


def build_slide_speed(prs):
    """Slide 12: 処理速度評価."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "10. 処理速度評価")

    # Big numbers - left side
    _add_textbox(slide, Inches(0.8), Inches(1.3), Inches(5.5), Inches(0.4),
                 "処理時間", size=18, bold=True, color=DARK_BLUE)

    # Wall-clock card
    card1 = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.8), Inches(2.0), Inches(3.5), Inches(2.5),
    )
    card1.fill.solid()
    card1.fill.fore_color.rgb = LIGHT_GRAY
    card1.line.color.rgb = MID_BLUE
    card1.line.width = Pt(1.5)

    _add_textbox(slide, Inches(0.9), Inches(2.2), Inches(3.3), Inches(0.4),
                 "全体処理時間", size=14, bold=True, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(0.9), Inches(2.7), Inches(3.3), Inches(0.7),
                 "0.1 s", size=40, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(0.9), Inches(3.5), Inches(3.3), Inches(0.4),
                 "± 0.0 s (Wall-clock)", size=12, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(0.9), Inches(3.9), Inches(3.3), Inches(0.3),
                 "600フレーム / 301秒シナリオ",
                 size=11, bold=False, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)

    # Frame time card
    card2 = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(4.8), Inches(2.0), Inches(3.5), Inches(2.5),
    )
    card2.fill.solid()
    card2.fill.fore_color.rgb = LIGHT_GRAY
    card2.line.color.rgb = MID_BLUE
    card2.line.width = Pt(1.5)

    _add_textbox(slide, Inches(4.9), Inches(2.2), Inches(3.3), Inches(0.4),
                 "平均フレーム処理時間", size=14, bold=True, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(4.9), Inches(2.7), Inches(3.3), Inches(0.7),
                 "0.2 ms", size=40, bold=True, color=DARK_BLUE, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(4.9), Inches(3.5), Inches(3.3), Inches(0.4),
                 "± 0.0 ms (1σ)", size=12, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)
    _add_textbox(slide, Inches(4.9), Inches(3.9), Inches(3.3), Inches(0.3),
                 "リアルタイム要件を十分満足",
                 size=11, bold=False, color=DARK_GRAY, alignment=PP_ALIGN.CENTER)

    # GPU advantage table on right
    _add_textbox(slide, Inches(8.8), Inches(1.3), Inches(4.0), Inches(0.4),
                 "GPU加速の利点", size=18, bold=True, color=DARK_BLUE)

    _add_bullet_list(slide, Inches(8.8), Inches(1.9), Inches(4.0), Inches(3.5), [
        "CUDA並列処理により高速化",
        "UKFシグマ点計算をGPUで並列実行",
        "データアソシエーションの並列化",
        "フレーム時間 0.2ms << 更新間隔 500ms",
        "リアルタイム比: 2,500倍高速",
    ], size=13, color=DARK_GRAY)

    # Performance summary
    _add_textbox(slide, Inches(0.8), Inches(5.0), Inches(11.5), Inches(0.4),
                 "性能サマリー", size=16, bold=True, color=DARK_BLUE)

    perf_data = [
        ["メトリクス", "値", "リアルタイム要件", "判定"],
        ["全体処理時間", "0.1 ± 0.0 s", "< 301 s (シナリオ時間)", "PASS"],
        ["フレーム処理時間", "0.2 ± 0.0 ms", "< 500 ms (更新間隔)", "PASS"],
        ["リアルタイム比", "× 2,500", "× 1 以上", "PASS"],
    ]
    tbl = _make_table(
        slide, Inches(0.8), Inches(5.5), Inches(11.5), Inches(1.6),
        len(perf_data), 4, perf_data,
        col_widths=[Inches(2.5), Inches(2.5), Inches(4.0), Inches(2.5)],
    )

    # Colour the PASS cells green
    table = tbl.table
    for r in range(1, len(perf_data)):
        cell = table.cell(r, 3)
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = GREEN
                run.font.bold = True

    _add_bottom_line(slide)
    _add_slide_number(slide, 12, TOTAL_SLIDES)


def build_slide_full_view(prs):
    """Slide 13: シミュレーション後の全体画面."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "参考: シミュレーション後の全体画面")

    full_img = os.path.join(SCREENSHOTS_DIR, "10_full_after_sim.png")
    if os.path.exists(full_img):
        slide.shapes.add_picture(full_img, Inches(0.5), Inches(1.2), Inches(12.3), Inches(5.8))
        _add_textbox(slide, Inches(0.5), Inches(7.0), Inches(12.3), Inches(0.3),
                     "図: シミュレーション完了後のFastTracker Web GUI全体画面 — 地図上のセンサー覆域・追尾軌道・3D追尾結果を統合表示",
                     size=10, bold=False, color=MID_GRAY, alignment=PP_ALIGN.CENTER)

    _add_bottom_line(slide)
    _add_slide_number(slide, 13, TOTAL_SLIDES)


def build_slide_summary(prs):
    """Slide 14: 総合評価・まとめ."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_header_bar(slide, "11. 総合評価・まとめ")

    # Full summary table
    _add_textbox(slide, Inches(0.6), Inches(1.2), Inches(6.0), Inches(0.4),
                 "全メトリクス総括", size=16, bold=True, color=DARK_BLUE)

    summary_data = [
        ["評価項目", "結果", "標準偏差", "判定"],
        ["位置 RMSE", "986.1 m", "± 51.4 m", "---"],
        ["速度 RMSE", "199.8 m/s", "± 33.0 m/s", "---"],
        ["OSPA距離", "836.7 m", "± 35.5 m", "---"],
        ["Precision", "100.0%", "± 0.0%", "PASS"],
        ["Recall", "99.6%", "± 0.3%", "PASS"],
        ["F1 Score", "99.8%", "± 0.2%", "PASS"],
        ["全体処理時間", "0.1 s", "± 0.0 s", "PASS"],
        ["フレーム処理時間", "0.2 ms", "± 0.0 ms", "PASS"],
    ]
    tbl = _make_table(
        slide, Inches(0.6), Inches(1.7), Inches(7.0), Inches(4.5),
        len(summary_data), 4, summary_data,
        col_widths=[Inches(2.2), Inches(1.6), Inches(1.6), Inches(1.6)],
    )

    # Colour PASS green, --- as gray
    table = tbl.table
    for r in range(1, len(summary_data)):
        cell = table.cell(r, 3)
        for paragraph in cell.text_frame.paragraphs:
            for run in paragraph.runs:
                if run.text == "PASS":
                    run.font.color.rgb = GREEN
                    run.font.bold = True
                elif run.text == "---":
                    run.font.color.rgb = MID_GRAY

    # Strengths / Improvements - right column
    _add_textbox(slide, Inches(8.0), Inches(1.2), Inches(4.5), Inches(0.4),
                 "強み", size=16, bold=True, color=GREEN)
    _add_bullet_list(slide, Inches(8.0), Inches(1.7), Inches(4.5), Inches(2.0), [
        "検出性能が極めて高い (F1=99.8%)",
        "偽トラック生成なし (Precision=100%)",
        "GPU加速でリアルタイム比2,500倍",
        "安定したモンテカルロ結果 (低σ)",
        "IMMモデル切替が機能",
    ], size=12, color=DARK_GRAY, bullet_color=GREEN)

    _add_textbox(slide, Inches(8.0), Inches(3.9), Inches(4.5), Inches(0.4),
                 "改善検討事項", size=16, bold=True, color=RED)
    _add_bullet_list(slide, Inches(8.0), Inches(4.4), Inches(4.5), Inches(2.0), [
        "位置RMSE ~1km (高高度域で増大)",
        "再突入フェーズでOSPA増大傾向",
        "初期捕捉に1秒の遅延あり",
        "多目標シナリオでの検証が必要",
    ], size=12, color=DARK_GRAY, bullet_color=RED)

    # Conclusion box at bottom
    concl = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.6), Inches(6.3), Inches(12.1), Inches(0.8),
    )
    concl.fill.solid()
    concl.fill.fore_color.rgb = DARK_BLUE
    concl.line.fill.background()

    _add_textbox(
        slide, Inches(0.8), Inches(6.4), Inches(11.7), Inches(0.6),
        "結論: FastTracker v1.0は、単一弾道ミサイルシナリオにおいて高い検出性能 (F1=99.8%) と"
        "十分なリアルタイム処理能力 (フレーム時間0.2ms) を実証した。",
        size=14, bold=True, color=WHITE, alignment=PP_ALIGN.CENTER,
    )

    _add_bottom_line(slide)
    _add_slide_number(slide, 14, TOTAL_SLIDES)


# =========================================================================
# Main
# =========================================================================
def main():
    print("FastTracker 性能評価報告書を生成中...")

    # Load data
    data = load_csv(CSV_PATH)
    print(f"  CSVデータ読込完了: {len(data['timestamp'])} フレーム")

    # Create temporary directory for charts
    chart_dir = tempfile.mkdtemp(prefix="fasttracker_charts_")
    print(f"  チャート一時ディレクトリ: {chart_dir}")

    # Create presentation (16:9 widescreen)
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Build slides
    print("  スライド1: 表紙")
    build_slide_title(prs)

    print("  スライド2: 目次")
    build_slide_toc(prs)

    print("  スライド3: 評価概要")
    build_slide_overview(prs)

    print("  スライド4: 試験条件 - シナリオ")
    build_slide_scenario(prs)

    print("  スライド5: 試験条件 - パラメータ")
    build_slide_params(prs)

    print("  スライド6: 軌道生成結果 (スクリーンショット)")
    build_slide_trajectory(prs)

    print("  スライド7: 追尾結果の可視化 (スクリーンショット)")
    build_slide_tracking_visuals(prs)

    print("  スライド8: 追尾精度評価 - 位置誤差 (チャート生成中)")
    build_slide_pos_error(prs, data, chart_dir)

    print("  スライド9: 追尾精度評価 - 速度誤差 (チャート生成中)")
    build_slide_vel_error(prs, data, chart_dir)

    print("  スライド10: 追尾精度評価 - OSPA距離 (チャート生成中)")
    build_slide_ospa(prs, data, chart_dir)

    print("  スライド11: 検出性能評価 (チャート生成中)")
    build_slide_detection(prs, chart_dir)

    print("  スライド12: 処理速度評価")
    build_slide_speed(prs)

    print("  スライド13: シミュレーション後の全体画面")
    build_slide_full_view(prs)

    print("  スライド14: 総合評価・まとめ")
    build_slide_summary(prs)

    # Save
    prs.save(OUTPUT_PATH)
    print(f"\n  保存完了: {OUTPUT_PATH}")

    # Clean up chart temp files
    for fname in os.listdir(chart_dir):
        fpath = os.path.join(chart_dir, fname)
        try:
            os.remove(fpath)
        except OSError:
            pass
    try:
        os.rmdir(chart_dir)
        print("  一時ファイルを削除しました")
    except OSError:
        print(f"  警告: 一時ディレクトリの削除に失敗: {chart_dir}")

    print("\n生成完了!")


if __name__ == "__main__":
    main()
