#!/usr/bin/env python3
"""
FastTracker Algorithm Description Document Generator

Generates a comprehensive Japanese algorithm description document
(algorithm_description.docx) for the FastTracker system.

Usage:
    python generate_algorithm_doc.py

Requirements:
    - python-docx
    - matplotlib
"""

import os
import sys
import tempfile
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.mathtext

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DOCX = os.path.join(SCRIPT_DIR, "algorithm_description.docx")
TEMP_DIR = tempfile.mkdtemp(prefix="fasttracker_doc_")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FONT_NAME_JP = "游ゴシック"
FONT_NAME_EN = "Yu Gothic"
TITLE_COLOR = RGBColor(0x1A, 0x3C, 0x6E)  # Dark navy
HEADING_COLOR = RGBColor(0x1F, 0x4E, 0x79)
ACCENT_COLOR = RGBColor(0x2E, 0x75, 0xB6)
TABLE_HEADER_BG = "1F4E79"
TABLE_ALT_BG = "D6E4F0"

# Track generated images for cleanup
_image_counter = 0


def render_math(latex_str, filename=None, fontsize=14, dpi=200):
    """Render a LaTeX math string to a PNG image using matplotlib mathtext.

    Args:
        latex_str: LaTeX math string (without surrounding $).
        filename: Output filename (auto-generated if None).
        fontsize: Font size for the rendered equation.
        dpi: Resolution of the output image.

    Returns:
        Absolute path to the generated PNG file.
    """
    global _image_counter
    if filename is None:
        _image_counter += 1
        filename = f"eq_{_image_counter:03d}.png"

    filepath = os.path.join(TEMP_DIR, filename)

    fig = plt.figure(figsize=(0.01, 0.01))
    fig.patch.set_alpha(0.0)

    text = fig.text(
        0, 0,
        f"${latex_str}$",
        fontsize=fontsize,
        color="black",
        ha="left",
        va="baseline",
    )

    fig.savefig(
        filepath,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    return filepath


def render_math_block(latex_str, filename=None, fontsize=16, dpi=200):
    """Render a display-style math block (larger, centered)."""
    return render_math(latex_str, filename=filename, fontsize=fontsize, dpi=dpi)


# ---------------------------------------------------------------------------
# Document helpers
# ---------------------------------------------------------------------------

def set_cell_shading(cell, color_hex):
    """Set background color of a table cell."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)


def set_run_font(run, size=10, bold=False, italic=False, color=None, name=None):
    """Apply font formatting to a run."""
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = color
    fname = name or FONT_NAME_JP
    run.font.name = fname
    r = run._element
    rPr = r.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = parse_xml(f'<w:rFonts {nsdecls("w")}/>')
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:eastAsia"), fname)


def add_paragraph_with_font(doc, text, style=None, size=10, bold=False,
                            alignment=None, space_after=Pt(6), color=None):
    """Add a paragraph with proper Japanese font settings."""
    p = doc.add_paragraph(style=style)
    if alignment is not None:
        p.alignment = alignment
    p.paragraph_format.space_after = space_after

    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, color=color)
    return p


def add_math_image(doc, latex_str, fontsize=14, width_inches=None):
    """Render a math formula and insert it as an inline image."""
    img_path = render_math(latex_str, fontsize=fontsize)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run()
    if width_inches:
        run.add_picture(img_path, width=Inches(width_inches))
    else:
        # Auto-size but cap max width
        from PIL import Image as PILImage
        try:
            with PILImage.open(img_path) as im:
                w_px, h_px = im.size
            # Limit to 6 inches wide
            w_in = min(w_px / 200.0, 6.0)
            run.add_picture(img_path, width=Inches(w_in))
        except Exception:
            run.add_picture(img_path, width=Inches(4.0))
    return p


def add_math_inline(paragraph, latex_str, fontsize=12):
    """Add an inline math image to an existing paragraph."""
    img_path = render_math(latex_str, fontsize=fontsize)
    run = paragraph.add_run()
    try:
        from PIL import Image as PILImage
        with PILImage.open(img_path) as im:
            w_px, h_px = im.size
        w_in = min(w_px / 200.0, 3.0)
        run.add_picture(img_path, width=Inches(w_in))
    except Exception:
        run.add_picture(img_path, width=Inches(2.0))


def add_bullet(doc, text, level=0, size=10):
    """Add a bullet point."""
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(1.5 + level * 1.0)
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    set_run_font(run, size=size)
    return p


def add_table_with_header(doc, headers, rows, col_widths=None):
    """Add a formatted table with a colored header row."""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    # Header row
    hdr_row = table.rows[0]
    for i, header_text in enumerate(headers):
        cell = hdr_row.cells[i]
        cell.text = ""
        set_cell_shading(cell, TABLE_HEADER_BG)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(header_text)
        set_run_font(run, size=9, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))

    # Data rows
    for row_idx, row_data in enumerate(rows):
        row = table.rows[1 + row_idx]
        for col_idx, cell_text in enumerate(row_data):
            cell = row.cells[col_idx]
            cell.text = ""
            if row_idx % 2 == 1:
                set_cell_shading(cell, TABLE_ALT_BG)
            p = cell.paragraphs[0]
            run = p.add_run(str(cell_text))
            set_run_font(run, size=9)

    # Set column widths if provided
    if col_widths:
        for row in table.rows:
            for i, width in enumerate(col_widths):
                row.cells[i].width = Cm(width)

    return table


def setup_styles(doc):
    """Configure document styles for Japanese text."""
    style = doc.styles["Normal"]
    font = style.font
    font.name = FONT_NAME_JP
    font.size = Pt(10)
    rPr = style.element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = parse_xml(f'<w:rFonts {nsdecls("w")}/>')
        rPr.insert(0, rFonts)
    rFonts.set(qn("w:eastAsia"), FONT_NAME_JP)

    # Heading styles
    for level in range(1, 4):
        style_name = f"Heading {level}"
        if style_name in doc.styles:
            h_style = doc.styles[style_name]
            h_font = h_style.font
            h_font.name = FONT_NAME_JP
            h_font.color.rgb = HEADING_COLOR
            h_rPr = h_style.element.get_or_add_rPr()
            h_rFonts = h_rPr.find(qn("w:rFonts"))
            if h_rFonts is None:
                h_rFonts = parse_xml(f'<w:rFonts {nsdecls("w")}/>')
                h_rPr.insert(0, h_rFonts)
            h_rFonts.set(qn("w:eastAsia"), FONT_NAME_JP)

            if level == 1:
                h_font.size = Pt(18)
            elif level == 2:
                h_font.size = Pt(14)
            else:
                h_font.size = Pt(12)


def setup_page(doc):
    """Set up A4 page with reasonable margins and page numbers."""
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.0)

    # Add page numbers in footer
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Page number field
    run = p.add_run()
    fldChar1 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
    run._element.append(fldChar1)

    run2 = p.add_run()
    instrText = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> PAGE </w:instrText>')
    run2._element.append(instrText)

    run3 = p.add_run()
    fldChar2 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="separate"/>')
    run3._element.append(fldChar2)

    run4 = p.add_run("1")
    set_run_font(run4, size=9, color=RGBColor(0x80, 0x80, 0x80))

    run5 = p.add_run()
    fldChar3 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
    run5._element.append(fldChar3)


# ---------------------------------------------------------------------------
# Document sections
# ---------------------------------------------------------------------------

def add_title_page(doc):
    """Create a professional title page."""
    # Spacer
    for _ in range(6):
        doc.add_paragraph()

    # Title
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("FastTracker")
    set_run_font(run, size=36, bold=True, color=TITLE_COLOR, name="Consolas")

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run2 = p2.add_run("アルゴリズム解説書")
    set_run_font(run2, size=28, bold=True, color=TITLE_COLOR)

    # Subtitle
    doc.add_paragraph()
    p3 = doc.add_paragraph()
    p3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run3 = p3.add_run("GPU加速型マルチターゲット追尾システム")
    set_run_font(run3, size=14, color=ACCENT_COLOR)

    p4 = doc.add_paragraph()
    p4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run4 = p4.add_run("IMM-UKF / CUDA / 弾道ミサイル・HGV対応")
    set_run_font(run4, size=12, color=ACCENT_COLOR)

    # Spacer
    for _ in range(6):
        doc.add_paragraph()

    # Document info table
    info_table = doc.add_table(rows=4, cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    info_data = [
        ("文書番号", "FT-ALG-001"),
        ("版数", "1.0"),
        ("作成日", "2026年2月17日"),
        ("分類", "技術文書"),
    ]
    for i, (label, value) in enumerate(info_data):
        cell_l = info_table.rows[i].cells[0]
        cell_l.text = ""
        set_cell_shading(cell_l, TABLE_HEADER_BG)
        p = cell_l.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(label)
        set_run_font(run, size=10, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF))
        cell_l.width = Cm(4)

        cell_r = info_table.rows[i].cells[1]
        cell_r.text = ""
        p = cell_r.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(value)
        set_run_font(run, size=10)
        cell_r.width = Cm(8)

    doc.add_page_break()


def add_toc(doc):
    """Add a table of contents page."""
    doc.add_heading("目次", level=1)

    # TOC field code
    p = doc.add_paragraph()
    run = p.add_run()
    fldChar1 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
    run._element.append(fldChar1)

    run2 = p.add_run()
    instrText = parse_xml(
        f'<w:instrText {nsdecls("w")} xml:space="preserve">'
        ' TOC \\o "1-3" \\h \\z \\u </w:instrText>'
    )
    run2._element.append(instrText)

    run3 = p.add_run()
    fldChar2 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="separate"/>')
    run3._element.append(fldChar2)

    # Placeholder text
    run4 = p.add_run("（目次はWordで「フィールドの更新」を実行すると自動生成されます）")
    set_run_font(run4, size=10, italic=True, color=RGBColor(0x80, 0x80, 0x80))

    run5 = p.add_run()
    fldChar3 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
    run5._element.append(fldChar3)

    # Manual TOC listing for reference
    doc.add_paragraph()
    toc_entries = [
        ("1.", "システム概要"),
        ("2.", "状態空間モデル"),
        ("3.", "Unscented Kalman Filter (UKF)"),
        ("4.", "IMM (Interacting Multiple Model) フィルタ"),
        ("5.", "運動モデル"),
        ("  5.1", "CV（等速直線）モデル"),
        ("  5.2", "弾道（RK4）モデル"),
        ("  5.3", "CT（旋回）モデル"),
        ("6.", "データアソシエーション"),
        ("7.", "航跡管理"),
        ("8.", "評価指標"),
    ]
    for num, title in toc_entries:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        run = p.add_run(f"{num}  {title}")
        indent = "  " in num
        set_run_font(run, size=10 if not indent else 9.5)

    doc.add_page_break()


def add_section_1(doc):
    """Section 1: System Overview."""
    doc.add_heading("1. システム概要", level=1)

    add_paragraph_with_font(
        doc,
        "FastTrackerは、弾道ミサイルおよび極超音速滑空体（HGV）を対象とした"
        "GPU加速型マルチターゲット追尾システムである。CUDAによる並列計算を活用し、"
        "UKFシグマポイント演算およびコスト行列計算を高速化する。",
        size=10,
    )

    doc.add_heading("1.1 システムアーキテクチャ", level=2)

    add_paragraph_with_font(
        doc,
        "本システムは以下の6つの主要コンポーネントで構成される。"
        "各コンポーネントはパイプライン構造で逐次処理を行う。",
        size=10,
    )

    # Architecture pipeline
    pipeline = [
        ("Target Generator", "目標生成器", "弾道ミサイル・HGV等の目標軌道を物理モデルに基づき生成"),
        ("Radar Simulator", "レーダシミュレータ", "観測ノイズ・検出確率を模擬し、観測値を生成"),
        ("Data Association", "データアソシエーション", "観測値と既存航跡の対応付け（ハンガリアン法）"),
        ("IMM-UKF", "IMM-UKFフィルタ", "複数運動モデルによる状態推定（GPU加速）"),
        ("Track Manager", "航跡管理器", "航跡の生成・確認・削除を管理"),
        ("Evaluator", "評価器", "RMSE, OSPA等の追尾性能評価"),
    ]

    add_table_with_header(
        doc,
        ["コンポーネント", "日本語名", "機能概要"],
        [(c, j, d) for c, j, d in pipeline],
        col_widths=[4.0, 4.0, 8.0],
    )

    doc.add_paragraph()
    add_paragraph_with_font(
        doc,
        "処理フロー:",
        size=10, bold=True,
    )
    add_paragraph_with_font(
        doc,
        "Target Generator → Radar Simulator → Data Association → IMM-UKF → Track Manager → Evaluator",
        size=10,
        alignment=WD_ALIGN_PARAGRAPH.CENTER,
    )

    doc.add_heading("1.2 GPU加速", level=2)
    add_paragraph_with_font(
        doc,
        "CUDAを用いた以下の処理のGPU並列化により、リアルタイム処理を実現する。",
        size=10,
    )
    add_bullet(doc, "UKFシグマポイントの生成・伝播（各航跡を並列処理）")
    add_bullet(doc, "コスト行列の計算（観測-航跡ペアを並列計算）")
    add_bullet(doc, "行列演算（コレスキー分解、行列乗算等）")

    doc.add_page_break()


def add_section_2(doc):
    """Section 2: State-Space Model."""
    doc.add_heading("2. 状態空間モデル", level=1)

    # --- State vector ---
    doc.add_heading("2.1 状態ベクトル", level=2)
    add_paragraph_with_font(
        doc,
        "状態ベクトルは9次元（STATE_DIM=9）で、位置・速度・加速度を含む。",
        size=10,
    )
    add_math_image(doc, r"\mathbf{x} = [x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]^T", fontsize=16)

    add_table_with_header(
        doc,
        ["状態変数", "物理量", "単位", "説明"],
        [
            ("x, y, z", "位置", "m", "目標原点座標系（x=東, y=北, z=高度）"),
            ("vx, vy, vz", "速度", "m/s", "各軸方向の速度成分"),
            ("ax, ay, az", "加速度", "m/s²", "各軸方向の加速度成分"),
        ],
        col_widths=[3.0, 2.5, 2.0, 8.5],
    )

    doc.add_paragraph()

    # --- Measurement vector ---
    doc.add_heading("2.2 観測ベクトル", level=2)
    add_paragraph_with_font(
        doc,
        "観測ベクトルは4次元（MEAS_DIM=4）で、レーダ観測量を表す。",
        size=10,
    )
    add_math_image(doc, r"\mathbf{z} = [r, \theta, \phi, \dot{d}]^T", fontsize=16)

    add_table_with_header(
        doc,
        ["観測変数", "物理量", "単位", "定義"],
        [
            ("r", "距離", "m", "レーダから目標までのスラントレンジ"),
            ("θ", "方位角", "rad", "atan2(dy, dx)"),
            ("φ", "仰角", "rad", "atan2(dz, r_horiz)"),
            ("ḋ", "ドップラー速度", "m/s", "(dx·vx + dy·vy + dz·vz) / r"),
        ],
        col_widths=[2.5, 3.0, 2.0, 8.5],
    )

    doc.add_paragraph()
    add_paragraph_with_font(
        doc,
        "ここで、dx = x - sensor_x, dy = y - sensor_y, dz = z - sensor_z "
        "はセンサ位置からの相対座標である。",
        size=10,
    )

    # --- Measurement noise ---
    doc.add_heading("2.3 観測ノイズ", level=2)
    add_paragraph_with_font(
        doc,
        "観測ノイズ共分散行列 R はデフォルトで以下の対角行列で定義される。",
        size=10,
    )

    add_table_with_header(
        doc,
        ["観測量", "パラメータ", "デフォルト値", "備考"],
        [
            ("距離", "σ_r", "10.0 m", ""),
            ("方位角", "σ_θ", "0.01 rad", "≈ 0.57°"),
            ("仰角", "σ_φ", "0.01 rad", "≈ 0.57°"),
            ("ドップラー", "σ_ḋ", "2.0 m/s", ""),
        ],
        col_widths=[3.0, 3.0, 3.5, 3.5],
    )

    doc.add_paragraph()
    add_math_image(
        doc,
        r"R = \mathrm{diag}(\sigma_r^2,\ \sigma_\theta^2,\ \sigma_\phi^2,\ \sigma_{\dot{d}}^2)",
        fontsize=15,
    )

    doc.add_page_break()


def add_section_3(doc):
    """Section 3: UKF."""
    doc.add_heading("3. Unscented Kalman Filter (UKF)", level=1)

    add_paragraph_with_font(
        doc,
        "Unscented Kalman Filter（UKF）は、非線形系に対するカルマンフィルタの拡張であり、"
        "シグマポイントによる統計的線形化を用いる。"
        "本システムではCUDAによりシグマポイント演算をGPU上で並列処理する。",
        size=10,
    )

    # --- Parameters ---
    doc.add_heading("3.1 UKFパラメータ", level=2)
    add_table_with_header(
        doc,
        ["パラメータ", "記号", "値", "説明"],
        [
            ("拡散パラメータ", "α", "0.5", "シグマポイントの拡散度を制御"),
            ("分布パラメータ", "β", "2.0", "ガウス分布では β=2 が最適"),
            ("二次スケーリング", "κ", "0.0", "通常 0 に設定"),
            ("状態次元", "n", "9", "STATE_DIM"),
            ("スケーリング", "λ", "α²(n+κ)−n", "= 0.5²×9 − 9 = −6.75"),
        ],
        col_widths=[3.5, 2.0, 3.5, 7.0],
    )

    doc.add_paragraph()
    add_paragraph_with_font(doc, "λの計算式:", size=10, bold=True)
    add_math_image(doc, r"\lambda = \alpha^2 (n + \kappa) - n", fontsize=16)

    # --- Sigma points ---
    doc.add_heading("3.2 シグマポイント", level=2)
    add_paragraph_with_font(
        doc,
        "2n+1 = 19個のシグマポイントを以下のように生成する（√(·) はコレスキー分解を表す）。",
        size=10,
    )

    add_math_image(doc, r"\chi_0 = \bar{\mathbf{x}}", fontsize=15)
    add_math_image(
        doc,
        r"\chi_i = \bar{\mathbf{x}} + \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i "
        r"\quad (i=1,\ldots,n)",
        fontsize=15,
    )
    add_math_image(
        doc,
        r"\chi_{i+n} = \bar{\mathbf{x}} - \left(\sqrt{(n+\lambda)\mathbf{P}}\right)_i "
        r"\quad (i=1,\ldots,n)",
        fontsize=15,
    )

    # --- Weights ---
    doc.add_heading("3.3 重み係数", level=2)
    add_paragraph_with_font(doc, "平均用重み:", size=10, bold=True)
    add_math_image(doc, r"W_0^m = \frac{\lambda}{n + \lambda}", fontsize=15)
    add_math_image(
        doc,
        r"W_i^m = \frac{1}{2(n + \lambda)} \quad (i=1,\ldots,2n)",
        fontsize=15,
    )

    add_paragraph_with_font(doc, "共分散用重み:", size=10, bold=True)
    add_math_image(
        doc,
        r"W_0^c = \frac{\lambda}{n + \lambda} + (1 - \alpha^2 + \beta)",
        fontsize=15,
    )
    add_math_image(
        doc,
        r"W_i^c = \frac{1}{2(n + \lambda)} \quad (i=1,\ldots,2n)",
        fontsize=15,
    )

    # --- Prediction ---
    doc.add_heading("3.4 予測ステップ", level=2)
    add_paragraph_with_font(doc, "予測ステップは以下の手順で実行する。", size=10)

    add_paragraph_with_font(doc, "手順1: シグマポイント生成", size=10, bold=True)
    add_paragraph_with_font(
        doc,
        "現在の状態推定値 x̂ と共分散行列 P から、上記のシグマポイント χᵢ を生成する。",
        size=10,
    )

    add_paragraph_with_font(doc, "手順2: 運動モデルによる伝播", size=10, bold=True)
    add_math_image(doc, r"\chi_i^* = f(\chi_i, \Delta t)", fontsize=15)

    add_paragraph_with_font(doc, "手順3: 予測平均", size=10, bold=True)
    add_math_image(
        doc,
        r"\hat{\mathbf{x}}^- = \sum_{i=0}^{2n} W_i^m \chi_i^*",
        fontsize=15,
    )

    add_paragraph_with_font(doc, "手順4: 予測共分散", size=10, bold=True)
    add_math_image(
        doc,
        r"\mathbf{P}^- = \sum_{i=0}^{2n} W_i^c "
        r"(\chi_i^* - \hat{\mathbf{x}}^-)(\chi_i^* - \hat{\mathbf{x}}^-)^T + \mathbf{Q} \cdot \Delta t",
        fontsize=15,
    )

    # --- Update ---
    doc.add_heading("3.5 更新ステップ", level=2)

    add_paragraph_with_font(doc, "手順1: 観測空間への変換", size=10, bold=True)
    add_math_image(doc, r"\mathbf{Z}_i = h(\chi_i^*)", fontsize=15)

    add_paragraph_with_font(doc, "手順2: 予測観測値", size=10, bold=True)
    add_math_image(
        doc,
        r"\hat{\mathbf{z}} = \sum_{i=0}^{2n} W_i^m \mathbf{Z}_i",
        fontsize=15,
    )

    add_paragraph_with_font(doc, "手順3: イノベーション共分散", size=10, bold=True)
    add_math_image(
        doc,
        r"\mathbf{S} = \sum_{i=0}^{2n} W_i^c "
        r"(\mathbf{Z}_i - \hat{\mathbf{z}})(\mathbf{Z}_i - \hat{\mathbf{z}})^T + \mathbf{R}",
        fontsize=15,
    )

    add_paragraph_with_font(doc, "手順4: 交差共分散", size=10, bold=True)
    add_math_image(
        doc,
        r"\mathbf{P}_{xz} = \sum_{i=0}^{2n} W_i^c "
        r"(\chi_i^* - \hat{\mathbf{x}}^-)(\mathbf{Z}_i - \hat{\mathbf{z}})^T",
        fontsize=15,
    )

    add_paragraph_with_font(doc, "手順5: カルマンゲイン", size=10, bold=True)
    add_math_image(doc, r"\mathbf{K} = \mathbf{P}_{xz} \cdot \mathbf{S}^{-1}", fontsize=15)

    add_paragraph_with_font(doc, "手順6: 状態更新", size=10, bold=True)
    add_math_image(
        doc,
        r"\hat{\mathbf{x}} = \hat{\mathbf{x}}^- + \mathbf{K}(\mathbf{z} - \hat{\mathbf{z}})",
        fontsize=15,
    )

    add_paragraph_with_font(doc, "手順7: 共分散更新", size=10, bold=True)
    add_math_image(
        doc,
        r"\mathbf{P} = \mathbf{P}^- - \mathbf{K} \cdot \mathbf{S} \cdot \mathbf{K}^T",
        fontsize=15,
    )

    # --- Process noise ---
    doc.add_heading("3.6 プロセスノイズ", level=2)
    add_paragraph_with_font(
        doc,
        "プロセスノイズ共分散行列 Q はデフォルトで以下の対角行列である。",
        size=10,
    )

    add_table_with_header(
        doc,
        ["成分", "パラメータ", "デフォルト値"],
        [
            ("位置 (x, y, z)", "σ_pos", "160 m"),
            ("速度 (vx, vy, vz)", "σ_vel", "65 m/s"),
            ("加速度 (ax, ay, az)", "σ_acc", "160 m/s²"),
        ],
        col_widths=[5.0, 4.0, 4.0],
    )

    doc.add_paragraph()
    add_math_image(
        doc,
        r"\mathbf{Q} = \mathrm{diag}("
        r"\sigma_{pos}^2, \sigma_{pos}^2, \sigma_{pos}^2, "
        r"\sigma_{vel}^2, \sigma_{vel}^2, \sigma_{vel}^2, "
        r"\sigma_{acc}^2, \sigma_{acc}^2, \sigma_{acc}^2)",
        fontsize=13,
    )

    doc.add_page_break()


def add_section_4(doc):
    """Section 4: IMM Filter."""
    doc.add_heading("4. IMM (Interacting Multiple Model) フィルタ", level=1)

    add_paragraph_with_font(
        doc,
        "IMM（Interacting Multiple Model）フィルタは、"
        "複数の運動モデルを同時に適用し、モデル確率に基づいて重み付け統合を行う。"
        "これにより、機動変化を伴う目標の追尾精度を向上させる。",
        size=10,
    )

    # --- Models ---
    doc.add_heading("4.1 運動モデル構成", level=2)
    add_paragraph_with_font(
        doc,
        "本システムでは3種類の運動モデルを使用する。",
        size=10,
    )

    add_table_with_header(
        doc,
        ["モデルID", "名称", "略称", "適用場面", "ノイズスケール"],
        [
            ("0", "等速直線", "CV", "巡航・中間飛翔段階", "10%（安定飛行）"),
            ("1", "弾道", "Ballistic", "弾道飛翔（重力＋大気抵抗）", "30%（物理モデルが正確）"),
            ("2", "旋回", "CT", "HGV機動・旋回飛翔", "100%（機動）"),
        ],
        col_widths=[2.0, 3.0, 2.5, 4.5, 4.0],
    )

    # --- Transition matrix ---
    doc.add_heading("4.2 モデル遷移確率行列", level=2)
    add_paragraph_with_font(
        doc,
        "モデル間の遷移はマルコフ連鎖でモデル化され、"
        "遷移確率行列 Π は以下のように定義される。",
        size=10,
    )

    add_table_with_header(
        doc,
        ["", "→ CV", "→ Ballistic", "→ CT"],
        [
            ("CV", "0.80", "0.15", "0.05"),
            ("Ballistic", "0.10", "0.85", "0.05"),
            ("CT", "0.05", "0.10", "0.85"),
        ],
        col_widths=[3.0, 3.5, 3.5, 3.5],
    )

    doc.add_paragraph()
    add_paragraph_with_font(
        doc,
        "初期モデル確率は一様分布: μ₀ = [1/3, 1/3, 1/3]",
        size=10,
    )

    # --- IMM Cycle ---
    doc.add_heading("4.3 IMMサイクル", level=2)

    add_paragraph_with_font(doc, "手順1: 混合（Mixing）", size=10, bold=True)
    add_paragraph_with_font(
        doc,
        "各モデルの予測確率を遷移確率と現在のモデル確率から計算する。",
        size=10,
    )
    add_math_image(
        doc,
        r"c_j = \sum_k \pi(k \to j) \cdot \mu_k",
        fontsize=15,
    )

    add_paragraph_with_font(doc, "手順2: モデル別予測", size=10, bold=True)
    add_paragraph_with_font(
        doc,
        "各運動モデルがそれぞれ独立に状態予測を行う（CV, Ballistic, CTモデル）。",
        size=10,
    )

    add_paragraph_with_font(doc, "手順3: 重み付け統合", size=10, bold=True)
    add_math_image(
        doc,
        r"\hat{\mathbf{x}} = \sum_j c_j \cdot \hat{\mathbf{x}}_j",
        fontsize=15,
    )
    add_math_image(
        doc,
        r"\mathbf{P} = \sum_j c_j \left[ \mathbf{P}_j + "
        r"(\hat{\mathbf{x}}_j - \hat{\mathbf{x}})(\hat{\mathbf{x}}_j - \hat{\mathbf{x}})^T \right]",
        fontsize=15,
    )

    add_paragraph_with_font(doc, "手順4: 尤度ベース更新（観測後）", size=10, bold=True)
    add_paragraph_with_font(
        doc,
        "観測値を取得した後、各モデルの尤度に基づきモデル確率を更新する。",
        size=10,
    )
    add_math_image(
        doc,
        r"L_j = \mathcal{N}(\mathrm{innovation}_j;\ 0,\ \mathbf{S}_j)",
        fontsize=15,
    )
    add_math_image(
        doc,
        r"\mu_j = \frac{c_j \cdot L_j}{\sum_k c_k \cdot L_k}",
        fontsize=15,
    )

    add_paragraph_with_font(
        doc,
        "数値安定性のため、尤度計算はlog空間で実行する。"
        "また、モデル確率の下限値（floor）を0.01に設定し、"
        "特定モデルの確率が0に収束することを防止する。",
        size=10,
    )

    doc.add_page_break()


def add_section_5(doc):
    """Section 5: Motion Models."""
    doc.add_heading("5. 運動モデル", level=1)

    # --- 5.1 CV ---
    doc.add_heading("5.1 CV（等速直線）モデル", level=2)
    add_paragraph_with_font(
        doc,
        "巡航・中間飛翔段階の安定した飛行を表現するモデルである。"
        "加速度は時定数 τ=5s で指数減衰する。",
        size=10,
    )

    add_paragraph_with_font(doc, "位置更新:", size=10, bold=True)
    add_math_image(doc, r"\mathbf{p}(t+\Delta t) = \mathbf{p}(t) + \mathbf{v} \cdot \Delta t", fontsize=15)

    add_paragraph_with_font(doc, "速度更新:", size=10, bold=True)
    add_math_image(doc, r"\mathbf{v}(t+\Delta t) = \mathbf{v}(t)", fontsize=15)

    add_paragraph_with_font(doc, "加速度更新（指数減衰）:", size=10, bold=True)
    add_math_image(
        doc,
        r"\mathbf{a}(t+\Delta t) = \mathbf{a}(t) \cdot \exp\left(-\frac{\Delta t}{\tau}\right)"
        r"\quad (\tau = 5\ \mathrm{s})",
        fontsize=15,
    )

    # --- 5.2 Ballistic ---
    doc.add_heading("5.2 弾道（RK4）モデル", level=2)
    add_paragraph_with_font(
        doc,
        "重力および大気抵抗を考慮した物理ベースの弾道運動モデルである。"
        "RK4（4次ルンゲ・クッタ法）による数値積分を用いる。",
        size=10,
    )

    add_paragraph_with_font(doc, "高度依存重力:", size=10, bold=True)
    add_math_image(
        doc,
        r"g(h) = g_0 \cdot \left(\frac{R_E}{R_E + h}\right)^2",
        fontsize=15,
    )

    add_table_with_header(
        doc,
        ["定数", "記号", "値"],
        [
            ("標準重力加速度", "g₀", "9.80665 m/s²"),
            ("地球半径", "R_E", "6,371,000 m"),
        ],
        col_widths=[5.0, 3.0, 5.0],
    )

    doc.add_paragraph()
    add_paragraph_with_font(doc, "大気密度モデル（指数大気）:", size=10, bold=True)
    add_math_image(
        doc,
        r"\rho(h) = \rho_0 \cdot \exp\left(-\frac{h}{H}\right)",
        fontsize=15,
    )

    add_table_with_header(
        doc,
        ["定数", "記号", "値"],
        [
            ("海面大気密度", "ρ₀", "1.225 kg/m³"),
            ("スケールハイト", "H", "7,400 m"),
        ],
        col_widths=[5.0, 3.0, 5.0],
    )

    doc.add_paragraph()
    add_paragraph_with_font(doc, "抗力:", size=10, bold=True)
    add_math_image(
        doc,
        r"F_{drag} = \beta \cdot \rho(h) \cdot |\mathbf{v}| \cdot \mathbf{v}",
        fontsize=15,
    )
    add_paragraph_with_font(
        doc,
        "ここで β = Cd·A/(2m) = 0.001 は代表的な弾道係数である。",
        size=10,
    )

    add_paragraph_with_font(doc, "運動方程式:", size=10, bold=True)
    add_math_image(doc, r"\ddot{x} = -\beta \cdot \rho \cdot |\mathbf{v}| \cdot v_x", fontsize=15)
    add_math_image(doc, r"\ddot{y} = -\beta \cdot \rho \cdot |\mathbf{v}| \cdot v_y", fontsize=15)
    add_math_image(
        doc,
        r"\ddot{z} = -g(h) - \beta \cdot \rho \cdot |\mathbf{v}| \cdot v_z",
        fontsize=15,
    )

    add_paragraph_with_font(
        doc,
        "上記の微分方程式をRK4（4段ルンゲ・クッタ法）で各タイムステップごとに数値積分する。",
        size=10,
    )

    # --- 5.3 CT ---
    doc.add_heading("5.3 CT（旋回）モデル", level=2)
    add_paragraph_with_font(
        doc,
        "HGVの機動飛翔（旋回）を表現するCoordinated Turnモデルである。"
        "旋回角速度 ω を状態から推定し、旋回運動を適用する。",
        size=10,
    )

    add_paragraph_with_font(doc, "旋回角速度の推定:", size=10, bold=True)
    add_math_image(
        doc,
        r"\omega = \frac{v_x \cdot a_y - v_y \cdot a_x}{v_x^2 + v_y^2}",
        fontsize=15,
    )
    add_paragraph_with_font(
        doc,
        "ω は ±0.785 rad/s（±45°/s）にクランプされる。",
        size=10,
    )

    add_paragraph_with_font(doc, "|ω| > 10⁻⁴ の場合（旋回）:", size=10, bold=True)
    add_paragraph_with_font(doc, "水平面内の位置更新:", size=10)
    add_math_image(
        doc,
        r"x(t+\Delta t) = x + \frac{v_x \sin(\omega \Delta t) + v_y (\cos(\omega \Delta t) - 1)}{\omega}",
        fontsize=14,
    )
    add_math_image(
        doc,
        r"y(t+\Delta t) = y + \frac{-v_x (\cos(\omega \Delta t) - 1) + v_y \sin(\omega \Delta t)}{\omega}",
        fontsize=14,
    )

    add_paragraph_with_font(doc, "水平面内の速度更新:", size=10)
    add_math_image(
        doc,
        r"v_x' = v_x \cos(\omega \Delta t) - v_y \sin(\omega \Delta t)",
        fontsize=14,
    )
    add_math_image(
        doc,
        r"v_y' = v_x \sin(\omega \Delta t) + v_y \cos(\omega \Delta t)",
        fontsize=14,
    )

    add_paragraph_with_font(doc, "向心加速度:", size=10)
    add_math_image(
        doc,
        r"a_x = -\omega \cdot v_y', \quad a_y = \omega \cdot v_x'",
        fontsize=14,
    )

    add_paragraph_with_font(doc, "|ω| ≤ 10⁻⁴ の場合（直線近似）:", size=10, bold=True)
    add_paragraph_with_font(
        doc,
        "旋回角速度が十分小さい場合、等加速度直線運動モデルにフォールバックする。",
        size=10,
    )

    add_paragraph_with_font(doc, "垂直方向（共通）:", size=10, bold=True)
    add_math_image(
        doc,
        r"z(t+\Delta t) = z + v_z \cdot \Delta t + \frac{1}{2} a_z \cdot \Delta t^2",
        fontsize=15,
    )

    doc.add_page_break()


def add_section_6(doc):
    """Section 6: Data Association."""
    doc.add_heading("6. データアソシエーション", level=1)

    add_paragraph_with_font(
        doc,
        "データアソシエーションは、レーダ観測値を既存航跡に対応付ける処理である。"
        "コスト行列の計算はGPU上で並列実行され、最適割当はハンガリアン法（Munkres法）で解く。",
        size=10,
    )

    # --- Cost matrix ---
    doc.add_heading("6.1 コスト行列", level=2)
    add_paragraph_with_font(
        doc,
        "コスト行列の各要素は、正規化イノベーション距離（NID）で計算される。"
        "GPU上で各観測-航跡ペアを並列に計算する。",
        size=10,
    )
    add_math_image(
        doc,
        r"d^2(i,j) = \sum_{k \in \{r, az, el, dop\}} "
        r"\left(\frac{z_k^{meas} - z_k^{pred}}{\sigma_k}\right)^2",
        fontsize=15,
    )

    # --- Gating ---
    doc.add_heading("6.2 ゲーティング", level=2)
    add_paragraph_with_font(
        doc,
        "計算された距離がゲート閾値を超える場合、当該ペアを棄却する。",
        size=10,
    )

    add_table_with_header(
        doc,
        ["パラメータ", "値", "説明"],
        [
            ("ゲート閾値", "500.0", "NIDがこの値を超えるペアを棄却"),
            ("棄却コスト", "10¹⁰", "棄却されたペアに割り当てるコスト値"),
        ],
        col_widths=[4.0, 3.0, 9.0],
    )

    # --- Hungarian ---
    doc.add_heading("6.3 ハンガリアン法（Munkres法）", level=2)
    add_paragraph_with_font(
        doc,
        "ハンガリアン法は O(n³) の計算量で最適割当問題を解くアルゴリズムである。"
        "以下のステップで構成される。",
        size=10,
    )

    steps = [
        ("行縮約", "各行の最小値を行全体から減算する"),
        ("列縮約", "各列の最小値を列全体から減算する"),
        ("初期スターリング", "各行・列で唯一のゼロ要素にスターを付与する"),
        ("列カバー", "スター付きゼロを含む列をカバーする"),
        ("未カバーゼロの探索", "未カバーのゼロ要素にプライムを付与する"),
        ("拡張パス", "スターとプライムの交互パスにより割当を改善する"),
        ("行列調整", "未カバー要素の最小値で行列を調整し、手順4に戻る"),
    ]
    for i, (step_name, step_desc) in enumerate(steps, 1):
        add_paragraph_with_font(
            doc,
            f"ステップ{i}: {step_name} — {step_desc}",
            size=10,
        )

    doc.add_page_break()


def add_section_7(doc):
    """Section 7: Track Management."""
    doc.add_heading("7. 航跡管理", level=1)

    add_paragraph_with_font(
        doc,
        "航跡管理は、航跡のライフサイクル（生成・確認・消失）を制御する。",
        size=10,
    )

    # --- States ---
    doc.add_heading("7.1 航跡状態", level=2)
    add_paragraph_with_font(
        doc,
        "航跡は以下の3状態で管理される。",
        size=10,
    )

    add_table_with_header(
        doc,
        ["状態", "英語名", "説明", "遷移条件"],
        [
            ("仮航跡", "TENTATIVE", "新規生成された未確認航跡", "未対応観測から生成"),
            ("確認航跡", "CONFIRMED", "信頼性が確認された航跡", "ヒット数 ≥ confirm_hits (=2)"),
            ("消失航跡", "LOST", "観測が途絶えた航跡", "ミス数 ≥ delete_misses (=90 ≈ 3s @30Hz)"),
        ],
        col_widths=[2.5, 3.0, 5.0, 5.5],
    )

    doc.add_paragraph()
    add_paragraph_with_font(
        doc,
        "状態遷移: TENTATIVE → CONFIRMED → LOST",
        size=10,
        alignment=WD_ALIGN_PARAGRAPH.CENTER,
        bold=True,
    )

    # --- Initialization ---
    doc.add_heading("7.2 航跡初期化", level=2)
    add_paragraph_with_font(
        doc,
        "未対応の観測値から新規航跡を初期化する。極座標観測を直交座標に変換する。",
        size=10,
    )

    add_paragraph_with_font(doc, "位置の初期化（極座標→直交座標）:", size=10, bold=True)
    add_math_image(
        doc,
        r"x = r \cos(\phi) \cos(\theta) + x_{sensor}",
        fontsize=14,
    )
    add_math_image(
        doc,
        r"y = r \cos(\phi) \sin(\theta) + y_{sensor}",
        fontsize=14,
    )
    add_math_image(
        doc,
        r"z = r \sin(\phi) + z_{sensor}",
        fontsize=14,
    )

    add_paragraph_with_font(
        doc,
        "速度はドップラー観測値に基づき視線方向（LOS）に沿って初期化される。"
        "初期共分散は観測距離に応じてスケーリングされる。",
        size=10,
    )

    # --- Parameters ---
    doc.add_heading("7.3 航跡管理パラメータ", level=2)
    add_table_with_header(
        doc,
        ["パラメータ", "デフォルト値", "説明"],
        [
            ("confirm_hits", "2", "確認に必要なヒット数"),
            ("delete_misses", "90", "削除までのミス数（30Hzで約3秒）"),
            ("初期化最小SNR", "22.0 dB", "航跡初期化に必要な最小SNR"),
        ],
        col_widths=[4.0, 3.5, 8.5],
    )

    doc.add_page_break()


def add_section_8(doc):
    """Section 8: Evaluation Metrics."""
    doc.add_heading("8. 評価指標", level=1)

    add_paragraph_with_font(
        doc,
        "追尾性能を定量的に評価するための各種指標を以下に定義する。",
        size=10,
    )

    # --- RMSE ---
    doc.add_heading("8.1 RMSE（二乗平均平方根誤差）", level=2)

    add_paragraph_with_font(doc, "位置RMSE:", size=10, bold=True)
    add_math_image(
        doc,
        r"\mathrm{RMSE}_{pos} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} "
        r"\|\mathbf{p}_{track,i} - \mathbf{p}_{truth,i}\|^2}",
        fontsize=15,
    )

    add_paragraph_with_font(doc, "速度RMSE:", size=10, bold=True)
    add_math_image(
        doc,
        r"\mathrm{RMSE}_{vel} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} "
        r"\|\mathbf{v}_{track,i} - \mathbf{v}_{truth,i}\|^2}",
        fontsize=15,
    )

    # --- OSPA ---
    doc.add_heading("8.2 OSPA（Optimal SubPattern Assignment）", level=2)
    add_paragraph_with_font(
        doc,
        "OSPAはパターン間の距離指標であり、位置誤差と個数誤差を統合的に評価する。",
        size=10,
    )
    add_math_image(
        doc,
        r"d_{OSPA}^{(p)}(X,Y) = \left[ \frac{1}{\max(m,n)} "
        r"\left( \sum_{i} \min(d(x_i, y_{\pi(i)}), c)^p + |m-n| \cdot c^p \right) "
        r"\right]^{1/p}",
        fontsize=14,
    )

    add_table_with_header(
        doc,
        ["パラメータ", "記号", "デフォルト値", "説明"],
        [
            ("カットオフ距離", "c", "10,000 m", "個数ペナルティの上限距離"),
            ("次数", "p", "1", "距離のLpノルム次数"),
        ],
        col_widths=[3.5, 2.0, 3.5, 7.0],
    )

    # --- Detection metrics ---
    doc.add_heading("8.3 検出指標", level=2)

    add_math_image(
        doc,
        r"\mathrm{Precision} = \frac{TP}{TP + FP}",
        fontsize=15,
    )
    add_math_image(
        doc,
        r"\mathrm{Recall} = \frac{TP}{TP + FN}",
        fontsize=15,
    )
    add_math_image(
        doc,
        r"F_1 = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}"
        r"{\mathrm{Precision} + \mathrm{Recall}}",
        fontsize=15,
    )

    add_table_with_header(
        doc,
        ["指標", "定義", "説明"],
        [
            ("TP (True Positive)", "正しく追尾された目標", "航跡と真値が対応"),
            ("FP (False Positive)", "誤追尾", "真値に対応しない航跡"),
            ("FN (False Negative)", "追尾漏れ", "航跡が対応しない真値"),
        ],
        col_widths=[4.0, 4.5, 7.5],
    )

    # --- Track continuity ---
    doc.add_heading("8.4 航跡連続性指標", level=2)
    add_paragraph_with_font(
        doc,
        "各真値目標に対する航跡の連続性を以下の3カテゴリで評価する。",
        size=10,
    )

    add_table_with_header(
        doc,
        ["カテゴリ", "英語名", "条件", "説明"],
        [
            ("主追尾", "Mostly Tracked (MT)", "追尾率 ≥ 80%", "ライフタイムの80%以上で追尾"),
            ("部分追尾", "Partially Tracked (PT)", "20% ≤ 追尾率 < 80%", "ライフタイムの20～80%で追尾"),
            ("主消失", "Mostly Lost (ML)", "追尾率 < 20%", "ライフタイムの20%未満で追尾"),
        ],
        col_widths=[2.5, 4.0, 3.5, 6.0],
    )

    doc.add_page_break()


def add_revision_history(doc):
    """Add revision history table."""
    doc.add_heading("変更履歴", level=1)

    add_table_with_header(
        doc,
        ["版数", "日付", "変更内容", "担当"],
        [
            ("1.0", "2026/02/17", "初版作成", "—"),
            ("", "", "", ""),
            ("", "", "", ""),
        ],
        col_widths=[2.0, 3.0, 8.0, 3.0],
    )


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

def generate_document():
    """Generate the complete algorithm description document."""
    print("FastTracker Algorithm Document Generator")
    print("=" * 50)

    doc = Document()

    print("[1/11] Setting up page layout and styles...")
    setup_page(doc)
    setup_styles(doc)

    print("[2/11] Creating title page...")
    add_title_page(doc)

    print("[3/11] Adding table of contents...")
    add_toc(doc)

    print("[4/11] Section 1: System Overview...")
    add_section_1(doc)

    print("[5/11] Section 2: State-Space Model...")
    add_section_2(doc)

    print("[6/11] Section 3: UKF...")
    add_section_3(doc)

    print("[7/11] Section 4: IMM Filter...")
    add_section_4(doc)

    print("[8/11] Section 5: Motion Models...")
    add_section_5(doc)

    print("[9/11] Section 6-7: Data Association & Track Management...")
    add_section_6(doc)
    add_section_7(doc)

    print("[10/11] Section 8: Evaluation Metrics...")
    add_section_8(doc)

    print("[11/11] Adding revision history...")
    add_revision_history(doc)

    # Save
    print(f"\nSaving document to: {OUTPUT_DOCX}")
    doc.save(OUTPUT_DOCX)
    print("Document saved successfully.")

    # Cleanup temp images
    try:
        shutil.rmtree(TEMP_DIR)
        print(f"Cleaned up temporary files: {TEMP_DIR}")
    except Exception as e:
        print(f"Warning: Could not clean up temp dir: {e}")

    print(f"\nGenerated: {OUTPUT_DOCX}")
    print(f"Math images rendered: {_image_counter}")
    return OUTPUT_DOCX


if __name__ == "__main__":
    generate_document()
