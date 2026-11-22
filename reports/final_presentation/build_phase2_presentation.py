"""Generate an executive-grade, student-wise Phase 1 & Phase 2 presentation deck.

Student-by-Student Structure:
  Slide 1:  Executive Title & Team Identification
  Slide 2:  Executive Overview & Two-Phase Scientific Vision (Big KPI Cards)
  Slide 3:  Team Architecture & Interface Contracts (Students 1-5 Matrix)
  Slide 4:  Student 1 — Chemical Preprocessing & Assay Missingness (with Figure)
  Slide 5:  Student 1 — Bemis-Murcko Scaffold Partitioning & Leakage Prevention
  Slide 6:  Student 1 — Molecular Graph Featurization (19 Node + 18 Edge Features)
  Slide 7:  Student 2 — MolecularGNN Architecture Overview (with Pipeline Diagram)
  Slide 8:  Student 2 — Edge-Aware Message Passing & Residual LayerNorm Mechanics
  Slide 9:  Student 2 — Graph-Level Readout Pooling & 12-Task Prediction Head
  Slide 10: Student 3 — Multi-Task Masked BCE Loss Formulation
  Slide 11: Student 3 — Optimization Protocol, Learning Rate Schedules & Early Stopping
  Slide 12: Student 3 — Atomic Checkpointing, Verification & Reproducibility
  Slide 13: Student 4 — Quarantined Test Evaluation Protocol (N=783)
  Slide 14: Student 4 — Final Test ROC-AUC Across 12 Endpoints (with ROC Plot)
  Slide 15: Student 4 — Precision-Recall AUC vs Positive Prevalence (with PR Plot)
  Slide 16: Student 4 — Continuous Prediction Probability Distributions (with Distribution Plot)
  Slide 17: Student 5 — Centralized YAML Experiment Runner & Tracking Registry
  Slide 18: Student 5 — Systematic Architectural Ablations (with Ablation Plot)
  Slide 19: Student 5 — Cross-Paradigm Benchmark: Phase 1 vs Phase 2 (with Benchmark Plot)
  Slide 20: Conclusion, Technical Contributions & Viva Defense Q&A
"""

from __future__ import annotations

import os
from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = PROJECT_ROOT / "reports/final_ieee_report/figures"
OUT_PATH = PROJECT_ROOT / "reports/final_presentation/phase2_final_presentation.pptx"

# Modern Executive Color Palette
BG_DARK = RGBColor(0x0F, 0x17, 0x2A)       # Deep slate 900
BG_LIGHT = RGBColor(0xF8, 0xFA, 0xFC)      # Slate 50
TEXT_MAIN = RGBColor(0x0F, 0x17, 0x2A)     # Dark slate
TEXT_MUTED = RGBColor(0x64, 0x74, 0x8B)    # Slate 500
TEXT_LIGHT = RGBColor(0xE2, 0xE8, 0xF0)    # Slate 200
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

CARD_BG = RGBColor(0xFF, 0xFF, 0xFF)
CARD_BORDER = RGBColor(0xCB, 0xD5, 0xE1)   # Slate 300
HIGHLIGHT_BG = RGBColor(0xF1, 0xF5, 0xF9)

# Student-Specific Accent Themes
COLOR_S1 = RGBColor(0x0D, 0x94, 0x88)  # Teal (Student 1)
COLOR_S2 = RGBColor(0x25, 0x63, 0xEB)  # Royal Blue (Student 2)
COLOR_S3 = RGBColor(0x7C, 0x3A, 0xED)  # Purple (Student 3)
COLOR_S4 = RGBColor(0xDC, 0x26, 0x26)  # Red / Crimson (Student 4)
COLOR_S5 = RGBColor(0xEA, 0x58, 0x0C)  # Amber / Orange (Student 5)
COLOR_NEUTRAL = RGBColor(0x33, 0x41, 0x55)

FONT_HEAD = "Arial"
FONT_BODY = "Calibri"


def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def set_slide_bg(slide, color):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, category_badge, badge_color):
        # Header banner shape
        banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
        banner.fill.solid()
        banner.fill.fore_color.rgb = BG_DARK
        banner.line.fill.background()

        # Accent bar on top
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
        bar.fill.solid()
        bar.fill.fore_color.rgb = badge_color
        bar.line.fill.background()

        # Category pill badge
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(0.2), Inches(3.2), Inches(0.28))
        pill.fill.solid()
        pill.fill.fore_color.rgb = badge_color
        pill.line.fill.background()
        p = pill.text_frame.paragraphs[0]
        p.text = category_badge.upper()
        p.font.name = FONT_HEAD
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        pill.text_frame.margin_top = Inches(0.02)

        # Title text
        tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.52), Inches(11.5), Inches(0.55))
        p = tb.text_frame.paragraphs[0]
        p.text = title_text
        p.font.name = FONT_HEAD
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = WHITE
        tb.text_frame.margin_left = Inches(0)
        tb.text_frame.margin_top = Inches(0)

    def add_card(slide, left, top, width, height, title=None, border_color=CARD_BORDER, bg_color=CARD_BG):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = border_color
        card.line.width = Pt(1.5)

        if title:
            tb = slide.shapes.add_textbox(Inches(left + 0.25), Inches(top + 0.18), Inches(width - 0.5), Inches(0.4))
            p = tb.text_frame.paragraphs[0]
            p.text = title
            p.font.name = FONT_HEAD
            p.font.size = Pt(15)
            p.font.bold = True
            p.font.color.rgb = TEXT_MAIN
        return card

    def add_kpi_card(slide, left, top, width, height, value, label, subtext="", accent_color=COLOR_S2):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.5)

        # Accent strip
        strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.1))
        strip.fill.solid()
        strip.fill.fore_color.rgb = accent_color
        strip.line.fill.background()

        tb = slide.shapes.add_textbox(Inches(left + 0.15), Inches(top + 0.18), Inches(width - 0.3), Inches(height - 0.25))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = value
        p1.font.name = FONT_HEAD
        p1.font.size = Pt(28)
        p1.font.bold = True
        p1.font.color.rgb = accent_color
        p1.alignment = PP_ALIGN.CENTER

        p2 = tf.add_paragraph()
        p2.text = label
        p2.font.name = FONT_HEAD
        p2.font.size = Pt(12)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_MAIN
        p2.alignment = PP_ALIGN.CENTER

        if subtext:
            p3 = tf.add_paragraph()
            p3.text = subtext
            p3.font.name = FONT_BODY
            p3.font.size = Pt(10)
            p3.font.color.rgb = TEXT_MUTED
            p3.alignment = PP_ALIGN.CENTER

    def add_notes(slide, text):
        slide.notes_slide.notes_text_frame.text = text

    # =========================================================================
    # SLIDE 1: Title Slide (Executive Widescreen)
    # =========================================================================
    s1 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s1, BG_DARK)

    # Accent decorative top bar
    bar1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    bar1.fill.solid()
    bar1.fill.fore_color.rgb = COLOR_S2
    bar1.line.fill.background()

    # Category Pill
    pill1 = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.0), Inches(4.5), Inches(0.4))
    pill1.fill.solid()
    pill1.fill.fore_color.rgb = COLOR_S2
    pill1.line.fill.background()
    p = pill1.text_frame.paragraphs[0]
    p.text = "B.TECH RESEARCH FINAL PROJECT DEFENSE"
    p.font.name = FONT_HEAD
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    # Main Title
    tb_title = s1.shapes.add_textbox(Inches(1.0), Inches(1.6), Inches(11.3), Inches(1.8))
    tf = tb_title.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Molecular Toxicity Prediction via Chemical\nFingerprints & Molecular Graph Networks"
    p.font.name = FONT_HEAD
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = "A Cross-Paradigm Benchmark of Classical ML vs. Edge-Aware GNNs on Tox21"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    p2.space_before = Pt(8)

    # Student Cards Container (5 cards for 5 students)
    students = [
        ("Student 1", "Shourya Pratap", "Data Engineering & Graphs", COLOR_S1),
        ("Student 2", "Bulbul Agarwalla", "GNN Architecture & MPNN", COLOR_S2),
        ("Student 3", "Amogha Raj Sindur", "Training Pipeline & Loss", COLOR_S3),
        ("Student 4", "Kanishka Dubey", "Validation & Test Metrics", COLOR_S4),
        ("Student 5", "Aditya Yadav", "Integration & Final Paper", COLOR_S5),
    ]

    card_w, card_h = 2.15, 2.2
    for idx, (role, name, resp, color) in enumerate(students):
        c_left = 1.0 + idx * 2.3
        c_top = 4.2
        c = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(c_left), Inches(c_top), Inches(card_w), Inches(card_h))
        c.fill.solid()
        c.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        c.line.color.rgb = color
        c.line.width = Pt(2.0)

        tb_card = s1.shapes.add_textbox(Inches(c_left + 0.1), Inches(c_top + 0.2), Inches(card_w - 0.2), Inches(card_h - 0.3))
        tf_c = tb_card.text_frame
        tf_c.word_wrap = True

        p = tf_c.paragraphs[0]
        p.text = role.upper()
        p.font.name = FONT_HEAD
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = color

        p = tf_c.add_paragraph()
        p.text = name
        p.font.name = FONT_HEAD
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.space_before = Pt(4)

        p = tf_c.add_paragraph()
        p.text = resp
        p.font.name = FONT_BODY
        p.font.size = Pt(11)
        p.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
        p.space_before = Pt(6)

    tb_inst = s1.shapes.add_textbox(Inches(1.0), Inches(6.7), Inches(11.3), Inches(0.4))
    p = tb_inst.text_frame.paragraphs[0]
    p.text = "Newton School of Technology • Department of Computer Science & AI • Academic Year 2026"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)

    add_notes(s1, "Welcome the viva committee. Introduce the team and state the core research objective: benchmarking classical descriptor ML vs deep molecular graph neural networks under a leak-free Bemis-Murcko scaffold split.")

    # =========================================================================
    # SLIDE 2: Executive Overview (KPIs & Paradigm Overview)
    # =========================================================================
    s2 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s2, BG_LIGHT)
    add_header(s2, "Executive Summary: Chemical Representation & Benchmark Scope", "PROJECT VISION", COLOR_NEUTRAL)

    # 4 Key Metrics Cards
    add_kpi_card(s2, 0.8, 1.45, 2.7, 1.4, "7,823", "Cleaned Compounds", "RDKit Sanitized from 7,831", COLOR_S1)
    add_kpi_card(s2, 3.8, 1.45, 2.7, 1.4, "12", "Tox21 Endpoints", "7 Nuclear Receptors + 5 Stress Pathways", COLOR_S2)
    add_kpi_card(s2, 6.8, 1.45, 2.7, 1.4, "0.7488", "Phase 1 Champion", "Random Forest (FP + 9 Descriptors)", COLOR_S3)
    add_kpi_card(s2, 9.8, 1.45, 2.7, 1.4, "0.7395", "Phase 2 MolecularGNN", "Learned Edge-Aware MPNN (Parity)", COLOR_S5)

    # 2 Comparison Columns
    add_card(s2, 0.8, 3.1, 5.7, 3.9, "Phase 1: Classical Machine Learning", COLOR_S2)
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(3.7), Inches(5.3), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    bullet_items_p1 = [
        "Fixed-Length Structural Representations: 2,048-bit Morgan Circular Fingerprints (ECFP4 radius=2).",
        "Continuous Global Properties: 9 standardized RDKit physicochemical descriptors (MolWt, LogP, TPSA, etc.).",
        "Supervised Classifiers: L2-Logistic Regression, Random Forest (500 trees), XGBoost gradient boosting.",
        "Outcome: Random Forest + Combined features established the benchmark ceiling at 0.7488 ROC-AUC.",
        "Limitation: Manual feature engineering; bit hash collisions; no direct edge geometry."
    ]
    for i, itm in enumerate(bullet_items_p1):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    add_card(s2, 6.8, 3.1, 5.7, 3.9, "Phase 2: Molecular Graph Representation Learning", COLOR_S5)
    tb = s2.shapes.add_textbox(Inches(7.0), Inches(3.7), Inches(5.3), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    bullet_items_p2 = [
        "Topological Molecular Graphs: G = (V, E) operating directly on atoms and chemical bonds.",
        "Rich Spatial Features: 19 atom/node attributes + 18 bond/edge attributes.",
        "Edge-Aware Message Passing: 3-hop spatial neighborhood propagation with residual connections.",
        "Outcome: MolecularGNN achieves 0.7395 ROC-AUC, demonstrating statistical parity without handcrafted features.",
        "Core Advantage: End-to-end differentiable representation learning directly from 2D molecular structure."
    ]
    for i, itm in enumerate(bullet_items_p2):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    add_notes(s2, "Highlight the transition from Phase 1 (fingerprints + tabular models) to Phase 2 (graph networks). Notice that GNN achieves parity with RF while eliminating manual feature engineering.")

    # =========================================================================
    # SLIDE 3: Team Architecture & Ownership Matrix
    # =========================================================================
    s3 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s3, BG_LIGHT)
    add_header(s3, "Team Architecture: Modular Engineering & Integration Contracts", "TEAM MATRIX", COLOR_NEUTRAL)

    roles_data = [
        ("STUDENT 1", "Shourya Pratap", "Pre-processing & Graph Featurization",
         ["RDKit chemical sanitization (7,823 mols)", "Bemis-Murcko canonical scaffold splitting", "19 atom & 18 bond feature extraction", "PyTorch Graph Dataset collation"], COLOR_S1),
        ("STUDENT 2", "Bulbul Agarwalla", "GNN Architecture & Message Passing",
         ["MolecularGNN PyTorch implementation", "Atom/bond linear feature encoders", "Edge-conditioned message passing (MPNN)", "Mean/Sum pooling & 12-task prediction head"], COLOR_S2),
        ("STUDENT 3", "Amogha Raj Sindur", "Multi-Task Training & Optimization",
         ["Masked BCEWithLogitsLoss engine", "AdamW optimizer & ReduceLROnPlateau", "Early stopping & learning rate scheduling", "SHA-256 verified checkpoint serialization"], COLOR_S3),
        ("STUDENT 4", "Kanishka Dubey", "Test Evaluation & Error Diagnostics",
         ["Quarantined test evaluation (N=783)", "Mask-aware ROC-AUC & PR-AUC computation", "Zero test leakage enforcement", "Publication-ready figure & artifact generation"], COLOR_S4),
        ("STUDENT 5", "Aditya Yadav", "Integration, Ablations & Publications",
         ["Centralized YAML experiment orchestrator", "Systematic ablation matrix execution", "Cross-phase benchmark comparison", "Final IEEE manuscript & presentation deck"], COLOR_S5),
    ]

    for idx, (badge, name, title, tasks, col) in enumerate(roles_data):
        c_top = 1.45 + idx * 1.15
        card = add_card(s3, 0.8, c_top, 11.7, 1.05, border_color=col)

        # Student Badge & Name
        tb_n = s3.shapes.add_textbox(Inches(1.0), Inches(c_top + 0.1), Inches(2.8), Inches(0.85))
        tf = tb_n.text_frame
        p = tf.paragraphs[0]
        p.text = badge
        p.font.name = FONT_HEAD
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = col

        p2 = tf.add_paragraph()
        p2.text = name
        p2.font.name = FONT_HEAD
        p2.font.size = Pt(14)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_MAIN

        p3 = tf.add_paragraph()
        p3.text = title
        p3.font.name = FONT_BODY
        p3.font.size = Pt(10)
        p3.font.color.rgb = TEXT_MUTED

        # Deliverables Bullets
        tb_d = s3.shapes.add_textbox(Inches(4.0), Inches(c_top + 0.12), Inches(8.3), Inches(0.85))
        tf_d = tb_d.text_frame
        tf_d.word_wrap = True
        for i, tsk in enumerate(tasks):
            p = tf_d.paragraphs[0] if i == 0 else tf_d.add_paragraph()
            p.text = f"• {tsk}"
            p.font.name = FONT_BODY
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_MAIN
            p.space_after = Pt(1)

    add_notes(s3, "Emphasize that the project followed a strict collaborative contract. Each student owned an independent module with clean integration boundaries and automated test validation.")

    # =========================================================================
    # SLIDE 4: STUDENT 1 — Chemical Preprocessing & Missingness
    # =========================================================================
    s4 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s4, BG_LIGHT)
    add_header(s4, "Chemical Data Curation & Missing Label Handling", "STUDENT 1: DATA ENGINEERING", COLOR_S1)

    # Left Column: Curation Logic
    add_card(s4, 0.8, 1.45, 5.8, 5.5, "RDKit Sanitization & Label Masking", COLOR_S1)
    tb = s4.shapes.add_textbox(Inches(1.0), Inches(2.05), Inches(5.4), Inches(4.7))
    tf = tb.text_frame
    tf.word_wrap = True
    s1_curation_points = [
        "Raw Dataset Filtering: 7,831 raw compound records processed via RDKit. Exactly 8 invalid or unparseable structures were eliminated, yielding 7,823 validated chemical molecules.",
        "Chemical Sanitization: Salt stripping, counter-ion removal, valence normalization, and canonical tautomer standardization.",
        "Preserving Assay Missingness: Tox21 is an incomplete assay matrix where 23.6% of entries are unobserved (compounds not tested on certain pathways).",
        "Masked Boolean Policy: Missing values are represented as NaN paired with an explicit boolean observation mask m in {0, 1}.",
        "Strict Zero-Imputation Rejection: Imputing untested entries as inactive (0) introduces false negatives, skewing class boundaries and corrupting model calibration."
    ]
    for i, itm in enumerate(s1_curation_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    # Right Column: Visual of Label Distribution
    add_card(s4, 6.9, 1.45, 5.6, 5.5, "Tox21 Label Distribution & Missingness", COLOR_S1)
    fig_path = FIG_DIR / "fig_label_statistics.png"
    if fig_path.exists():
        s4.shapes.add_picture(str(fig_path), Inches(7.1), Inches(2.1), width=Inches(5.2))

    add_notes(s4, "Student 1's curation foundation: explain why RDKit sanitization is essential and why missing labels cannot be zero-imputed.")

    # =========================================================================
    # SLIDE 5: STUDENT 1 — Scaffold Partitioning
    # =========================================================================
    s5 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s5, BG_LIGHT)
    add_header(s5, "Bemis–Murcko Scaffold Splitting & Zero Leakage Protocol", "STUDENT 1: DATA ENGINEERING", COLOR_S1)

    # 3 Stat Cards for Split Counts
    add_kpi_card(s5, 0.8, 1.45, 3.6, 1.4, "6,258", "Training Graphs (80%)", "Model Optimization", COLOR_S1)
    add_kpi_card(s5, 4.8, 1.45, 3.6, 1.4, "782", "Validation Graphs (10%)", "Hyperparameter Tuning", COLOR_S2)
    add_kpi_card(s5, 8.8, 1.45, 3.7, 1.4, "783", "Quarantined Test (10%)", "Untouched Final Benchmark", COLOR_S4)

    add_card(s5, 0.8, 3.1, 11.7, 3.9, "Why Scaffold Splitting is Required in Cheminformatics", COLOR_S1)
    tb = s5.shapes.add_textbox(Inches(1.1), Inches(3.7), Inches(11.1), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    scaffold_bullets = [
        "The Scaffold Leakage Hazard: Under standard random splitting, molecules with identical core ring systems (scaffolds) appear in both train and test partitions. The model memorizes scaffold-activity pairs rather than true toxicophores.",
        "Bemis–Murcko Core Extraction: All side chains are stripped, isolating molecular rings and linkers. Molecules are clustered by core framework.",
        "Deterministic Cardinality Allocation: Scaffold clusters are sorted by descending size and partitioned strictly 80/10/10 into train, val, and test sets.",
        "Scaffold Disjointness Guarantee: Zero scaffolds overlap between partitions (Train ∩ Test = ∅).",
        "Out-of-Distribution Generalization: This protocol tests the model's ability to extrapolate toxicity to novel chemical families, mirroring genuine pharmaceutical discovery."
    ]
    for i, itm in enumerate(scaffold_bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    add_notes(s5, "Student 1: Emphasize that scaffold splitting is substantially harder than random splitting. MoleculeNet benchmarks Tox21 with random splits; our scaffold evaluation is much more rigorous.")

    # =========================================================================
    # SLIDE 6: STUDENT 1 — Molecular Graph Featurization
    # =========================================================================
    s6 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s6, BG_LIGHT)
    add_header(s6, "Molecular Graph Construction: 19 Atom & 18 Bond Features", "STUDENT 1: DATA ENGINEERING", COLOR_S1)

    add_card(s6, 0.8, 1.45, 5.7, 5.5, "19 Node Features (Atoms)", COLOR_S1)
    tb_node = s6.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb_node.text_frame
    tf.word_wrap = True
    node_feats = [
        "Atomic Identity (10 dims): One-hot encoding of C, N, O, S, F, P, Cl, Br, I, and other heteroatoms.",
        "Degree / Connectivity (6 dims): One-hot encoding of heavy atom valence degree (0 to 5+).",
        "Formal Charge (1 dim): Integer electrical charge of the atom.",
        "Aromaticity Indicator (1 dim): Boolean flag for membership in an aromatic ring system.",
        "Hydrogen Count (1 dim): Explicit number of bonded hydrogens.",
        "Hybridization State: Encoded orbital geometry (sp, sp2, sp3)."
    ]
    for i, itm in enumerate(node_feats):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    add_card(s6, 6.8, 1.45, 5.7, 5.5, "18 Edge Features (Bonds)", COLOR_S1)
    tb_edge = s6.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb_edge.text_frame
    tf.word_wrap = True
    edge_feats = [
        "Bond Order (4 dims): One-hot encoding of Single, Double, Triple, or Aromatic covalent bonds.",
        "Conjugation Status (1 dim): Boolean flag indicating pi-orbital conjugation across adjacent bonds.",
        "Ring System Membership (1 dim): Boolean flag indicating whether the bond is inside a ring.",
        "Stereochemical Configuration (6 dims): Stereo isomerism (None, Any, Z/E, Cis/Trans).",
        "Bidirectional Directed Graph: Every chemical bond is represented as two directed edges (u -> v and v -> u) to support message passing."
    ]
    for i, itm in enumerate(edge_feats):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    add_notes(s6, "Student 1: Detail the 19 atom and 18 bond features. Note that bonds are bidirectional directed edges so information can flow freely in both directions.")

    # =========================================================================
    # SLIDE 7: STUDENT 2 — MolecularGNN Architecture
    # =========================================================================
    s7 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s7, BG_LIGHT)
    add_header(s7, "MolecularGNN Architecture: Edge-Conditioned MPNN", "STUDENT 2: GNN ARCHITECTURE", COLOR_S2)

    add_card(s7, 0.8, 1.45, 5.8, 5.5, "End-to-End Pipeline & Architecture Flow", COLOR_S2)
    fig_p = FIG_DIR / "fig_pipeline.png"
    if fig_p.exists():
        s7.shapes.add_picture(str(fig_p), Inches(1.0), Inches(2.1), width=Inches(5.4))

    add_card(s7, 6.9, 1.45, 5.6, 5.5, "Modular Architectural Components", COLOR_S2)
    tb = s7.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    arch_points = [
        "Input Feature Encoders: Linear projection mapping 19-dim atom features and 18-dim bond features into a unified latent space (hidden_dim = 128).",
        "L=3 Message Passing Layers: 3 sequential rounds of edge-aware neighborhood propagation, covering a 3-bond topological radius.",
        "Residual Connections: Skip connections h^(l) = LayerNorm(h^(l-1) + Update) preventing signal degradation in deeper layers.",
        "Layer Normalization & Dropout: Stabilizes training across variable molecular sizes; 10% dropout prevents overfitting.",
        "Global Readout Pooling: Aggregates atom representations into a single size-invariant molecular embedding.",
        "Multi-Task Head: Direct linear projection to 12 unconstrained task logits."
    ]
    for i, itm in enumerate(arch_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    add_notes(s7, "Student 2: Walk through the architectural diagram. Point out the atom/bond encoders, the 3 message passing layers, and the multi-task prediction head.")

    # =========================================================================
    # SLIDE 8: STUDENT 2 — Message Passing Mechanics
    # =========================================================================
    s8 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s8, BG_LIGHT)
    add_header(s8, "Mathematical Formulation of Message Passing & Aggregation", "STUDENT 2: GNN ARCHITECTURE", COLOR_S2)

    add_card(s8, 0.8, 1.45, 11.7, 5.5, "The Message Passing & State Update Equations", COLOR_S2)
    tb = s8.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(11.1), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True

    math_sections = [
        "1. Linear Feature Encoders:",
        "   h_i^(0) = ReLU(W_v · x_i + b_v)      [Atom embedding: 19 -> 128]",
        "   h_ij^e  = ReLU(W_e · e_ij + b_e)     [Bond embedding: 18 -> 128]",
        "",
        "2. Edge-Conditioned Message Construction:",
        "   m_ij^(l) = MLP_msg( [ h_j^(l-1) || h_ij^e ] )",
        "   Each directed message concatenates the neighbor atom state with the bond connecting them.",
        "",
        "3. Permutation-Invariant Aggregation:",
        "   m̄_i^(l) = Σ_{j ∈ N(i)} m_ij^(l)",
        "   Incoming messages are summed across all bonded neighbors j in N(i).",
        "",
        "4. Node State Update with Residual LayerNorm:",
        "   h_i^(l) = LayerNorm( h_i^(l-1) + Dropout( ReLU( W_u · [ h_i^(l-1) || m̄_i^(l) ] ) ) )",
        "   Residual skip connection ensures atom identity is preserved while integrating neighbor context."
    ]
    for itm in math_sections:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = itm
        if itm.startswith(("1.", "2.", "3.", "4.")):
            p.font.name = FONT_HEAD
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = COLOR_S2
            p.space_before = Pt(4)
        elif "h_" in itm or "m_" in itm:
            p.font.name = "Consolas"
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = TEXT_MAIN
        else:
            p.font.name = FONT_BODY
            p.font.size = Pt(12)
            p.font.color.rgb = TEXT_MUTED

    add_notes(s8, "Student 2: Explain the math. Key points to highlight: edge features are included in messages; aggregation is sum; update has residual + LayerNorm.")

    # =========================================================================
    # SLIDE 9: STUDENT 2 — Readout Pooling & Prediction Head
    # =========================================================================
    s9 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s9, BG_LIGHT)
    add_header(s9, "Graph Readout Pooling & 12-Task Prediction Head", "STUDENT 2: GNN ARCHITECTURE", COLOR_S2)

    add_card(s9, 0.8, 1.45, 5.7, 5.5, "Graph-Level Readout Pooling", COLOR_S2)
    tb = s9.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    pool_points = [
        "Atom to Molecule Transition: Message passing yields hidden vectors for each individual atom. A readout function must aggregate them into one vector per molecule.",
        "Mean Pooling Formulation:",
        "   h_G = (1 / |V|) Σ_{i ∈ V} h_i^(L)",
        "Why Mean Pooling Outperforms Sum Pooling:",
        "   • Molecules in Tox21 range from 2 atoms to 50+ atoms.",
        "   • Sum pooling creates vector norms that scale with molecule size, introducing severe size bias.",
        "   • Mean pooling is size-invariant, focusing purely on chemical feature density.",
        "Ablation Proof: Mean pooling achieved 0.2771 val loss vs 0.3047 for Sum pooling (confirmed in Phase 2 ablations)."
    ]
    for i, itm in enumerate(pool_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}" if not itm.startswith("   ") else itm
        p.font.name = FONT_BODY
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(5)

    add_card(s9, 6.8, 1.45, 5.7, 5.5, "12-Task Multi-Task Prediction Head", COLOR_S2)
    tb = s9.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    head_points = [
        "Multi-Task Linear Projection:",
        "   ẑ = W_out · h_G + b_out",
        "   Output Shape: [batch_size, 12] continuous logits.",
        "Unconstrained Logits Output: The model outputs raw logits rather than sigmoid probabilities.",
        "Why No Sigmoid in Model: Numerically stable training requires BCEWithLogitsLoss (log-sum-exp trick). Probabilities are obtained via σ(ẑ) strictly during evaluation.",
        "Endpoint Order Preservation: The 12 outputs strictly match the canonical endpoint order established by Student 1 (NR-AR through SR-p53)."
    ]
    for i, itm in enumerate(head_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}" if not itm.startswith("   ") else itm
        p.font.name = FONT_BODY
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(5)

    add_notes(s9, "Student 2: Explain readout pooling and the prediction head. Be prepared to explain why mean pooling beats sum pooling (size variance).")

    # =========================================================================
    # SLIDE 10: STUDENT 3 — Masked Multi-Task Loss Engine
    # =========================================================================
    s10 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s10, BG_LIGHT)
    add_header(s10, "Multi-Task Training: Mask-Aware Binary Cross-Entropy Loss", "STUDENT 3: TRAINING PIPELINE", COLOR_S3)

    add_card(s10, 0.8, 1.45, 11.7, 5.5, "Masked Loss Formulation & Gradient Protection", COLOR_S3)
    tb = s10.shapes.add_textbox(Inches(1.1), Inches(2.1), Inches(11.1), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True

    loss_explanation = [
        "The Incomplete Label Challenge: Across 12 endpoints, 23.6% of entries are unobserved (compounds were not tested on every assay).",
        "",
        "Masked BCEWithLogitsLoss Equation:",
        "   L(Y, Ẑ, M) = Σ_{i=1}^B Σ_{k=1}^{12} m_ik · [ -y_ik · log σ(ẑ_ik) - (1 - y_ik) · log(1 - σ(ẑ_ik)) ] / Σ_{i,k} m_ik",
        "",
        "Key Engineering Properties:",
        "• Strict Masking: For missing entries (m_ik = 0), the loss contribution is exactly zero. Zero gradient is backpropagated.",
        "• Normalization by Observed Entries: Loss is divided by the count of actually observed labels, not batch_size × 12.",
        "• Numerical Stability: Implemented via PyTorch BCEWithLogitsLoss with binary masking, avoiding numerical underflow in log-sigmoid.",
        "• Class Imbalance Pos-Weight Support: Configurable positive multiplier w_k = N_neg / N_pos dynamically weighting rare toxic compounds."
    ]
    for itm in loss_explanation:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = itm
        if itm.startswith("   L"):
            p.font.name = "Consolas"
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = COLOR_S3
            p.space_after = Pt(6)
        elif itm.startswith("•"):
            p.font.name = FONT_BODY
            p.font.size = Pt(13)
            p.font.color.rgb = TEXT_MAIN
            p.space_after = Pt(4)
        else:
            p.font.name = FONT_HEAD
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = TEXT_MAIN

    add_notes(s10, "Student 3: Explain the masked loss function. This is a favorite faculty question: How did you train on multi-task data when a quarter of the labels are missing?")

    # =========================================================================
    # SLIDE 11: STUDENT 3 — Optimization & Schedulers
    # =========================================================================
    s11 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s11, BG_LIGHT)
    add_header(s11, "Optimization Strategy: AdamW, Schedulers & Early Stopping", "STUDENT 3: TRAINING PIPELINE", COLOR_S3)

    add_kpi_card(s11, 0.8, 1.45, 2.7, 1.4, "AdamW", "Optimizer", "lr=1e-3, weight_decay=1e-4", COLOR_S3)
    add_kpi_card(s11, 3.8, 1.45, 2.7, 1.4, "Plateau", "LR Scheduler", "factor=0.5, patience=4", COLOR_S3)
    add_kpi_card(s11, 6.8, 1.45, 2.7, 1.4, "1.0", "Gradient Clipping", "Max Norm L2", COLOR_S3)
    add_kpi_card(s11, 9.8, 1.45, 2.7, 1.4, "8", "Early Stopping", "Patience on Val Loss", COLOR_S3)

    add_card(s11, 0.8, 3.1, 11.7, 3.9, "Training Dynamics & Convergence Safeguards", COLOR_S3)
    tb = s11.shapes.add_textbox(Inches(1.1), Inches(3.7), Inches(11.1), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    train_points = [
        "Adaptive Learning Rate Schedule: ReduceLROnPlateau halves learning rate (1e-3 -> 5e-4 -> 2.5e-4) whenever validation loss plateaus for 4 epochs.",
        "Gradient Clipping: Gradients are clipped to a maximum L2-norm of 1.0 to prevent exploding gradients during message passing aggregation.",
        "Early Stopping Mechanism: Halts training if validation loss fails to improve for 8 consecutive epochs, saving compute and preventing overfitting on the 6,258 training graphs.",
        "Batch Size (32): Balances stochastic gradient noise with GPU/CPU memory throughput for variable-sized molecular graphs.",
        "Deterministic Epoch Logging: Structured CSV and JSON logs record train loss, validation loss, and learning rate at every epoch."
    ]
    for i, itm in enumerate(train_points):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(6)

    add_notes(s11, "Student 3: Explain optimization hyperparameters. Why AdamW? Why ReduceLROnPlateau? Why early stopping?")

    # =========================================================================
    # SLIDE 12: STUDENT 3 — Checkpointing & Reproducibility
    # =========================================================================
    s12 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s12, BG_LIGHT)
    add_header(s12, "Checkpoint Integrity & Deterministic Reproducibility Controls", "STUDENT 3: TRAINING PIPELINE", COLOR_S3)

    add_card(s12, 0.8, 1.45, 5.7, 5.5, "Atomic Checkpointing Protocol", COLOR_S3)
    tb = s12.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    ckpt_bullets = [
        "Best Model Checkpoint: Saved only when validation loss achieves a new minimum.",
        "Atomic File Writing: Model state is written to a temporary file before renaming, preventing corrupted files if interrupted.",
        "Metadata Bundling: Checkpoints bundle model state_dict, optimizer state, epoch index, validation loss, and full architecture hyperparameter dictionary.",
        "SHA-256 Checksum Verification: Checkpoints calculate and verify cryptographic SHA-256 hashes to guarantee bit-for-bit weight preservation.",
        "Isolated Last Checkpoint: 'last_gnn.pt' is saved alongside 'best_gnn.pt' for full training post-mortem analysis."
    ]
    for i, itm in enumerate(ckpt_bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    add_card(s12, 6.8, 1.45, 5.7, 5.5, "Multi-Seed Reproducibility Verification", COLOR_S3)
    tb = s12.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    repro_bullets = [
        "Global Seed Seeding: Seed 42 set across Python random, NumPy, PyTorch CPU, and PyTorch CUDA.",
        "Deterministic Data Collation: PyTorch DataLoader workers initialized with deterministic seed generators.",
        "Seed 42 vs Seed 43 Verification: Running an identical architecture with alternative seed 43 converged to val loss 0.2820 (vs 0.2771 for seed 42, Delta < 0.005).",
        "Deterministic Inference: Repeated test inference on quarantined graphs produces identical probabilities with maximum absolute difference < 1e-6."
    ]
    for i, itm in enumerate(repro_bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    add_notes(s12, "Student 3: Emphasize scientific reproducibility. SHA-256 verified checkpoints and seed audits guarantee that anyone cloning the repo gets the exact same numbers.")

    # =========================================================================
    # SLIDE 13: STUDENT 4 — Quarantined Test Evaluation
    # =========================================================================
    s13 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s13, BG_LIGHT)
    add_header(s13, "Independent Test Evaluation & Quarantined Protocol", "STUDENT 4: EVALUATION & VALIDATION", COLOR_S4)

    add_kpi_card(s13, 0.8, 1.45, 3.6, 1.4, "783", "Quarantined Test Graphs", "Zero Scaffold Overlap", COLOR_S4)
    add_kpi_card(s13, 4.8, 1.45, 3.6, 1.4, "7,175", "Valid Observed Labels", "76.4% Assay Observation Density", COLOR_S4)
    add_kpi_card(s13, 8.8, 1.45, 3.7, 1.4, "0.0", "Data Leakage", "Test Set Never Seen in Training", COLOR_S4)

    add_card(s13, 0.8, 3.1, 11.7, 3.9, "Student 4 Independent Evaluation Principles", COLOR_S4)
    tb = s13.shapes.add_textbox(Inches(1.1), Inches(3.7), Inches(11.1), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    eval_principles = [
        "Strict Quarantine Enforcement: Test molecules were never used for backpropagation, learning rate decisions, or early stopping.",
        "Model Invariance: Evaluated strictly in model.eval() with torch.no_grad() — frozen weights, no fine-tuning or threshold calibration.",
        "Mask-Aware Metric Computation: Metrics computed strictly over the 7,175 observed assay labels; unobserved assays excluded.",
        "Rank-Based Threshold-Free Metrics: Primary metrics are ROC-AUC (Mann-Whitney U rank statistic) and PR-AUC (Average Precision via trapezoidal integration).",
        "Macro Aggregation with Uncertainty: Computes macro sample mean across all 12 endpoints alongside sample Standard Error (SEM) and 95% Student-t confidence intervals."
    ]
    for i, itm in enumerate(eval_principles):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    add_notes(s13, "Student 4: Emphasize the separation of concerns. Student 4 operated as an independent auditor evaluating Student 3's checkpoint on the untouched test partition.")

    # =========================================================================
    # SLIDE 14: STUDENT 4 — Final Test ROC-AUC
    # =========================================================================
    s14 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s14, BG_LIGHT)
    add_header(s14, "Test Set Performance: ROC-AUC Across All 12 Endpoints", "STUDENT 4: EVALUATION & VALIDATION", COLOR_S4)

    add_card(s14, 0.8, 1.45, 6.2, 5.5, "Endpoint-Wise ROC-AUC vs Macro Mean", COLOR_S4)
    fig_roc = FIG_DIR / "endpoint_roc_auc.png"
    if fig_roc.exists():
        s14.shapes.add_picture(str(fig_roc), Inches(1.0), Inches(2.1), width=Inches(5.8))

    add_card(s14, 7.3, 1.45, 5.2, 5.5, "ROC-AUC Results & Analysis", COLOR_S4)
    tb = s14.shapes.add_textbox(Inches(7.5), Inches(2.1), Inches(4.8), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    roc_analysis = [
        "Macro Mean ROC-AUC: 0.7395 ± 0.0182 (SEM)",
        "95% Confidence Interval: [0.6995, 0.7795]",
        "All 12 Endpoints Valid: 0 excluded, 100% accounted for.",
        "",
        "Top Performing Assays:",
        "• SR-MMP (0.849): Mitochondrial membrane potential disruption — strong structural signals.",
        "• NR-AhR (0.832): Aryl hydrocarbon receptor — planar aromatic rings well captured by GNN.",
        "• NR-Aromatase (0.803): Steroid synthesis inhibition.",
        "",
        "Challenging Assays:",
        "• NR-AR (0.590): Androgen receptor — lowest positive prevalence (1.5%), severe class scarcity."
    ]
    for itm in roc_analysis:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = itm
        if itm.startswith("Macro Mean"):
            p.font.name = FONT_HEAD
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = COLOR_S4
            p.space_after = Pt(4)
        elif itm.startswith(("Top Performing", "Challenging")):
            p.font.name = FONT_HEAD
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = TEXT_MAIN
            p.space_before = Pt(6)
        else:
            p.font.name = FONT_BODY
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_MAIN

    add_notes(s14, "Student 4: Walk through the ROC-AUC bar chart. Explain why SR-MMP and NR-AhR perform best (strong topological motifs) and why NR-AR is challenging (1.5% prevalence).")

    # =========================================================================
    # SLIDE 15: STUDENT 4 — Final Test PR-AUC & Prevalence
    # =========================================================================
    s15 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s15, BG_LIGHT)
    add_header(s15, "Test Set Performance: Precision-Recall vs. Baseline Prevalence", "STUDENT 4: EVALUATION & VALIDATION", COLOR_S4)

    add_card(s15, 0.8, 1.45, 6.2, 5.5, "Endpoint PR-AUC vs Random Baseline Prevalence", COLOR_S4)
    fig_pr = FIG_DIR / "endpoint_pr_auc.png"
    if fig_pr.exists():
        s15.shapes.add_picture(str(fig_pr), Inches(1.0), Inches(2.1), width=Inches(5.8))

    add_card(s15, 7.3, 1.45, 5.2, 5.5, "PR-AUC Diagnostics & Class Imbalance", COLOR_S4)
    tb = s15.shapes.add_textbox(Inches(7.5), Inches(2.1), Inches(4.8), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    pr_analysis = [
        "Macro Mean PR-AUC: 0.3118 ± 0.0331 (SEM)",
        "95% Confidence Interval: [0.2389, 0.3847]",
        "",
        "PR-AUC vs Random Baseline:",
        "• A dummy random classifier achieves PR-AUC equal to positive class prevalence (1.5% - 15.2%).",
        "• MolecularGNN significantly exceeds random baseline on every single endpoint.",
        "• For SR-MMP, PR-AUC reaches 0.589 vs baseline prevalence of 0.152 (3.9x improvement).",
        "",
        "Prevalence Correlation:",
        "• Strong positive correlation between endpoint prevalence and PR-AUC.",
        "• Confirms that data scarcity on rare toxic endpoints is the primary bottleneck for precision."
    ]
    for itm in pr_analysis:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = itm
        if itm.startswith("Macro Mean"):
            p.font.name = FONT_HEAD
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = COLOR_S4
            p.space_after = Pt(4)
        elif itm.startswith(("PR-AUC vs", "Prevalence")):
            p.font.name = FONT_HEAD
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = TEXT_MAIN
            p.space_before = Pt(6)
        else:
            p.font.name = FONT_BODY
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_MAIN

    add_notes(s15, "Student 4: Explain the red diamonds on the PR-AUC chart: they represent the random baseline (positive prevalence). The green bars are GNN performance.")

    # =========================================================================
    # SLIDE 16: STUDENT 4 — Prediction Distributions & Confusion
    # =========================================================================
    s16 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s16, BG_LIGHT)
    add_header(s16, "Continuous Prediction Distributions & Error Diagnostics", "STUDENT 4: EVALUATION & VALIDATION", COLOR_S4)

    add_card(s16, 0.8, 1.45, 6.2, 5.5, "Prediction Probability Distributions", COLOR_S4)
    fig_dist = FIG_DIR / "prediction_probability_distributions.png"
    if fig_dist.exists():
        s16.shapes.add_picture(str(fig_dist), Inches(1.0), Inches(2.1), width=Inches(5.8))

    add_card(s16, 7.3, 1.45, 5.2, 5.5, "Distribution Insights & Threshold Collapse", COLOR_S4)
    tb = s16.shapes.add_textbox(Inches(7.5), Inches(2.1), Inches(4.8), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    dist_analysis = [
        "Continuous Probability Separation:",
        "• Blue histograms = True Negatives (cluster sharply near 0.0).",
        "• Red histograms = True Positives (shifted toward higher probabilities).",
        "",
        "The Threshold Collapse Phenomenon:",
        "• Fixed threshold t=0.5 achieves Mean Balanced Accuracy 0.5599 and F1 0.1777.",
        "• Why? Because positive prevalence is ~5%, calibrated model probabilities for positives naturally peak around 0.2 - 0.4.",
        "• An arbitrary threshold of 0.5 cuts off valid positives, demonstrating why rank-based metrics (ROC-AUC) are the gold standard."
    ]
    for itm in dist_analysis:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = itm
        if itm.startswith("Continuous") or itm.startswith("The Threshold"):
            p.font.name = FONT_HEAD
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = COLOR_S4
            p.space_before = Pt(6)
        else:
            p.font.name = FONT_BODY
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_MAIN
            p.space_after = Pt(2)

    add_notes(s16, "Student 4: Use the probability distribution plot to demonstrate model calibration and explain why standard 0.5 threshold fails under imbalanced data.")

    # =========================================================================
    # SLIDE 17: STUDENT 5 — Centralized Experiment Orchestrator
    # =========================================================================
    s17 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s17, BG_LIGHT)
    add_header(s17, "Centralized Experiment Orchestration & Configuration Engine", "STUDENT 5: INTEGRATION & BENCHMARKS", COLOR_S5)

    add_card(s17, 0.8, 1.45, 5.7, 5.5, "Orchestration Architecture", COLOR_S5)
    tb = s17.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    orch_bullets = [
        "src/experiments/run_experiment.py: Unified CLI and programmatic runner executing end-to-end experiment pipelines.",
        "src/experiments/experiment_config.py: Dataclass schema loading declarative YAML configs with full type validation.",
        "src/experiments/experiment_registry.py: Automated tracking registry recording git commit hashes, runtimes, device info, and metrics.",
        "configs/phase2/: 8 declarative YAML configs managing baseline and ablation settings.",
        "Integration Contract Verification: Directly interfaces Student 1 (data), Student 2 (model), Student 3 (training), and Student 4 (evaluation)."
    ]
    for i, itm in enumerate(orch_bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    add_card(s17, 6.8, 1.45, 5.7, 5.5, "Example YAML Experiment Schema", COLOR_S5)
    tb_yaml = s17.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf_y = tb_yaml.text_frame
    tf_y.word_wrap = True
    yaml_sample = [
        "experiment_id: gnn_baseline",
        "description: Default Phase 2 MolecularGNN",
        "",
        "# Architecture Parameters",
        "hidden_dim: 128",
        "num_layers: 3",
        "pooling: mean",
        "dropout: 0.1",
        "residual: true",
        "norm: layer",
        "",
        "# Optimization Parameters",
        "epochs: 10",
        "batch_size: 32",
        "learning_rate: 0.001",
        "weight_decay: 0.0001",
        "early_stopping_patience: 8",
        "",
        "# Reproducibility & Environment",
        "seed: 42",
        "device: auto"
    ]
    for itm in yaml_sample:
        p = tf_y.add_paragraph() if tf_y.paragraphs[0].text else tf_y.paragraphs[0]
        p.text = itm
        p.font.name = "Consolas"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_S5 if itm.startswith(("#", "experiment_id")) else TEXT_MAIN

    add_notes(s17, "Student 5: Present the experiment orchestration infrastructure. Explain that this enables any team member or researcher to reproduce runs with a single command.")

    # =========================================================================
    # SLIDE 18: STUDENT 5 — Systematic Ablation Study
    # =========================================================================
    s18 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s18, BG_LIGHT)
    add_header(s18, "Systematic Architectural & Optimization Ablation Matrix", "STUDENT 5: INTEGRATION & BENCHMARKS", COLOR_S5)

    add_card(s18, 0.8, 1.45, 6.2, 5.5, "Validation Loss Comparison Across Ablations", COLOR_S5)
    fig_abl = FIG_DIR / "ablation_validation_loss_comparison.png"
    if fig_abl.exists():
        s18.shapes.add_picture(str(fig_abl), Inches(1.0), Inches(2.1), width=Inches(5.8))

    add_card(s18, 7.3, 1.45, 5.2, 5.5, "Ablation Matrix Results & Findings", COLOR_S5)
    tb = s18.shapes.add_textbox(Inches(7.5), Inches(2.1), Inches(4.8), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    ablation_summary = [
        "1. Readout Pooling (Mean vs Sum):",
        "   • Mean: 0.2771 vs Sum: 0.3047",
        "   • Sum pooling introduces massive variance from variable molecular sizes.",
        "",
        "2. Message Passing Depth (L=3 vs L=2):",
        "   • L=3: 0.2771 vs L=2: 0.2783",
        "   • 3-hop receptive field captures entire aromatic ring systems and functional pharmacophores.",
        "",
        "3. Hidden Channels (d=128 vs d=64):",
        "   • d=128: 0.2771 vs d=64: 0.2809",
        "   • 128 channels provide needed capacity for 12 parallel multi-task endpoints.",
        "",
        "4. Seed Stability (Seed 42 vs Seed 43):",
        "   • Seed 42: 0.2771 vs Seed 43: 0.2820 (Delta < 0.005 confirms stable optimization)."
    ]
    for itm in ablation_summary:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = itm
        if itm.startswith(("1.", "2.", "3.", "4.")):
            p.font.name = FONT_HEAD
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = COLOR_S5
            p.space_before = Pt(4)
        else:
            p.font.name = FONT_BODY
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_MAIN

    add_notes(s18, "Student 5: Walk through the ablation results. Highlight that architecture choices were empirically validated, not guessed.")

    # =========================================================================
    # SLIDE 19: STUDENT 5 — Cross-Paradigm Benchmark
    # =========================================================================
    s19 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s19, BG_LIGHT)
    add_header(s19, "Cross-Paradigm Benchmark: Classical ML vs. Graph Neural Networks", "STUDENT 5: INTEGRATION & BENCHMARKS", COLOR_S5)

    add_card(s19, 0.8, 1.45, 5.8, 5.5, "Phase 1 Classical Benchmark Overview", COLOR_S5)
    fig_mc = FIG_DIR / "fig_model_comparison.png"
    if fig_mc.exists():
        s19.shapes.add_picture(str(fig_mc), Inches(1.0), Inches(2.1), width=Inches(5.4))

    add_card(s19, 6.9, 1.45, 5.6, 5.5, "Head-to-Head Quarantined Test Benchmark", COLOR_S5)
    tb = s19.shapes.add_textbox(Inches(7.1), Inches(2.1), Inches(5.2), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Benchmark Comparison (N=783 Test Graphs):"
    p.font.name = FONT_HEAD
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    bench_rows = [
        "• LR (Morgan): ROC 0.6865 ± 0.0210 | PR 0.2866",
        "• XGBoost (Comb): ROC 0.7456 ± 0.0175 | PR 0.3789",
        "• RF Champion (Comb): ROC 0.7488 ± 0.0185 | PR 0.3891",
        "• MolecularGNN: ROC 0.7395 ± 0.0182 | PR 0.3118",
        "",
        "Statistical Parity Conclusion:",
        "1. Overlapping Confidence Intervals:",
        "   RF CI [0.7125, 0.7851] vs GNN CI [0.6995, 0.7795].",
        "   Delta = 0.0093 is well within standard error.",
        "",
        "2. Representation Learning Win:",
        "   GNN matches the heavily engineered RF champion (2,057 dims) from raw atom/bond features alone.",
        "",
        "3. PR-AUC Dynamic:",
        "   RF retains higher PR-AUC due to axis-aligned splits on rare positive fingerprint bits."
    ]
    for itm in bench_rows:
        p = tf.add_paragraph()
        p.text = itm
        if itm.startswith("Statistical Parity") or itm.startswith("• RF") or itm.startswith("• MolecularGNN"):
            p.font.name = FONT_HEAD
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = COLOR_S5 if "MolecularGNN" in itm else TEXT_MAIN
        else:
            p.font.name = FONT_BODY
            p.font.size = Pt(11)
            p.font.color.rgb = TEXT_MUTED if itm.startswith("   ") else TEXT_MAIN

    add_notes(s19, "Student 5: Deliver the headline finding of the entire B.Tech thesis: GNN achieves statistical parity with the Random Forest baseline without requiring 2,057 engineered features.")

    # =========================================================================
    # SLIDE 20: Conclusion & Viva Defense Summary
    # =========================================================================
    s20 = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_bg(s20, BG_DARK)

    # Accent decorative top bar
    bar20 = s20.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    bar20.fill.solid()
    bar20.fill.fore_color.rgb = COLOR_S2
    bar20.line.fill.background()

    tb = s20.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.3), Inches(1.0))
    p = tb.text_frame.paragraphs[0]
    p.text = "Conclusion & Summary of Scientific Contributions"
    p.font.name = FONT_HEAD
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE

    # 3 Summary Cards
    c1 = s20.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.9), Inches(3.6), Inches(4.8))
    c1.fill.solid()
    c1.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    c1.line.color.rgb = COLOR_S1
    c1.line.width = Pt(1.5)
    tb = s20.shapes.add_textbox(Inches(1.2), Inches(2.1), Inches(3.2), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "1. RIGOROUS BENCHMARK"
    p.font.name = FONT_HEAD
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_S1
    bullets_c1 = [
        "7,823 validated molecules curated via RDKit.",
        "Zero-leakage Bemis-Murcko scaffold split (6,258/782/783).",
        "Masked multi-task training with no false-negative imputation.",
        "Full test suite: 212 tests passed."
    ]
    for b in bullets_c1:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(6)

    c2 = s20.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.85), Inches(1.9), Inches(3.6), Inches(4.8))
    c2.fill.solid()
    c2.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    c2.line.color.rgb = COLOR_S2
    c2.line.width = Pt(1.5)
    tb = s20.shapes.add_textbox(Inches(5.05), Inches(2.1), Inches(3.2), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "2. PARITY ACHIEVED"
    p.font.name = FONT_HEAD
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_S2
    bullets_c2 = [
        "Phase 1 RF: 0.7488 ± 0.0185 ROC-AUC.",
        "Phase 2 GNN: 0.7395 ± 0.0182 ROC-AUC.",
        "Statistical parity with overlapping 95% confidence intervals.",
        "Zero handcrafted features needed for GNN."
    ]
    for b in bullets_c2:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(6)

    c3 = s20.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.7), Inches(1.9), Inches(3.6), Inches(4.8))
    c3.fill.solid()
    c3.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    c3.line.color.rgb = COLOR_S5
    c3.line.width = Pt(1.5)
    tb = s20.shapes.add_textbox(Inches(8.9), Inches(2.1), Inches(3.2), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "3. FUTURE HORIZONS"
    p.font.name = FONT_HEAD
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_S5
    bullets_c3 = [
        "3D Equivariant GNNs (EGNN/SchNet) incorporating spatial bond lengths.",
        "Self-supervised pre-training on ChEMBL/ZINC (2M+ molecules).",
        "Ensemble of Random Forest + GNN for complementary feature synergy.",
        "Open for Faculty Questions."
    ]
    for b in bullets_c3:
        p = tf.add_paragraph()
        p.text = f"• {b}"
        p.font.name = FONT_BODY
        p.font.size = Pt(12)
        p.font.color.rgb = TEXT_LIGHT
        p.space_before = Pt(6)

    add_notes(s20, "Conclude the presentation. Thank the viva committee and faculty. Open the floor for questions, pointing to the viva defense guide for deep technical answers.")

    prs.save(str(OUT_PATH))
    print(f"Generated {OUT_PATH} successfully! ({len(prs.slides)} slides)")


if __name__ == "__main__":
    create_deck()
