"""Generate the exact Phase 2 final presentation matching the requested 20-slide deck.

Slide structure:
  Slide 1:  B.TECH RESEARCH | FINAL PROJECT DEFENSE
  Slide 2:  PROJECT VISION: Executive summary: two paradigms, one benchmark
  Slide 3:  CONTRIBUTORS: Team and module ownership
  Slide 4:  DATA ENGINEERING: Curating Tox21 and handling missing labels
  Slide 5:  DATA ENGINEERING: Scaffold splitting with a zero-leakage protocol
  Slide 6:  DATA ENGINEERING: Molecular graph: 19 atom and 18 bond features
  Slide 7:  GNN ARCHITECTURE: MolecularGNN: an edge-conditioned MPNN
  Slide 8:  GNN ARCHITECTURE: Message passing and aggregation, step by step
  Slide 9:  GNN ARCHITECTURE: Readout pooling and the 12-task prediction head
  Slide 10: TRAINING PIPELINE: Training on incomplete labels: the masked loss
  Slide 11: TRAINING PIPELINE: Optimization: AdamW, scheduler and early stopping
  Slide 12: TRAINING PIPELINE: Checkpoint integrity and reproducibility
  Slide 13: EVALUATION & VALIDATION: Independent test evaluation under quarantine
  Slide 14: EVALUATION & VALIDATION: ROC-AUC across all 12 endpoints
  Slide 15: EVALUATION & VALIDATION: Precision-recall against the random baseline
  Slide 16: EVALUATION & VALIDATION: Prediction distributions and threshold collapse
  Slide 17: INTEGRATION & BENCHMARKS: One-command, config-driven experiments
  Slide 18: INTEGRATION & BENCHMARKS: Ablations: every design choice was tested
  Slide 19: INTEGRATION & BENCHMARKS: Cross-paradigm benchmark: classical ML vs GNN
  Slide 20: Conclusion and scientific contributions
"""

from __future__ import annotations

from pathlib import Path
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = PROJECT_ROOT / "reports/final_ieee_report/figures"
OUT_PATH = PROJECT_ROOT / "reports/final_presentation/phase2_final_presentation.pptx"

# Color Palette: Modern, Academic & Crisp
BG_DARK = RGBColor(0x0F, 0x17, 0x2A)       # Deep slate 900
BG_LIGHT = RGBColor(0xF8, 0xFA, 0xFC)      # Slate 50
TEXT_MAIN = RGBColor(0x0F, 0x17, 0x2A)     # Dark slate
TEXT_MUTED = RGBColor(0x64, 0x74, 0x8B)    # Slate 500
TEXT_LIGHT = RGBColor(0xE2, 0xE8, 0xF0)    # Slate 200
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

CARD_BG = RGBColor(0xFF, 0xFF, 0xFF)
CARD_BORDER = RGBColor(0xCB, 0xD5, 0xE1)   # Slate 300

# Badges / Category Accents
COLOR_VISION = RGBColor(0x25, 0x63, 0xEB)       # Blue
COLOR_CONTRIB = RGBColor(0x47, 0x55, 0x69)      # Slate
COLOR_DATA = RGBColor(0x0D, 0x94, 0x88)         # Teal (Student 1)
COLOR_ARCH = RGBColor(0x25, 0x63, 0xEB)         # Royal Blue (Student 2)
COLOR_TRAIN = RGBColor(0x7C, 0x3A, 0xED)        # Purple (Student 3)
COLOR_EVAL = RGBColor(0xDC, 0x26, 0x26)         # Red (Student 4)
COLOR_BENCH = RGBColor(0xEA, 0x58, 0x0C)        # Orange (Student 5)

FONT_HEAD = "Arial"
FONT_BODY = "Calibri"


def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    def set_bg(slide, color):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = color

    def add_header(slide, title_text, category_badge, badge_color):
        # Header banner
        banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
        banner.fill.solid()
        banner.fill.fore_color.rgb = BG_DARK
        banner.line.fill.background()

        # Accent top bar
        bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
        bar.fill.solid()
        bar.fill.fore_color.rgb = badge_color
        bar.line.fill.background()

        # Category pill
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
        p.font.size = Pt(21)
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
            p.font.size = Pt(14)
            p.font.bold = True
            p.font.color.rgb = TEXT_MAIN
        return card

    def add_kpi(slide, left, top, width, height, val, lbl, sub="", color=COLOR_VISION):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = CARD_BORDER
        card.line.width = Pt(1.5)

        strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.08))
        strip.fill.solid()
        strip.fill.fore_color.rgb = color
        strip.line.fill.background()

        tb = slide.shapes.add_textbox(Inches(left + 0.1), Inches(top + 0.15), Inches(width - 0.2), Inches(height - 0.2))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = val
        p1.font.name = FONT_HEAD
        p1.font.size = Pt(26)
        p1.font.bold = True
        p1.font.color.rgb = color
        p1.alignment = PP_ALIGN.CENTER

        p2 = tf.add_paragraph()
        p2.text = lbl
        p2.font.name = FONT_HEAD
        p2.font.size = Pt(12)
        p2.font.bold = True
        p2.font.color.rgb = TEXT_MAIN
        p2.alignment = PP_ALIGN.CENTER

        if sub:
            p3 = tf.add_paragraph()
            p3.text = sub
            p3.font.name = FONT_BODY
            p3.font.size = Pt(10)
            p3.font.color.rgb = TEXT_MUTED
            p3.alignment = PP_ALIGN.CENTER

    def add_notes(slide, txt):
        slide.notes_slide.notes_text_frame.text = txt

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    s1 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s1, BG_DARK)

    bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_VISION
    bar.line.fill.background()

    pill = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.1), Inches(4.8), Inches(0.4))
    pill.fill.solid()
    pill.fill.fore_color.rgb = COLOR_VISION
    pill.line.fill.background()
    p = pill.text_frame.paragraphs[0]
    p.text = "B.TECH RESEARCH  |  FINAL PROJECT DEFENSE"
    p.font.name = FONT_HEAD
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "Molecular Toxicity Prediction via Chemical\nFingerprints & Molecular Graph Networks"
    p.font.name = FONT_HEAD
    p.font.size = Pt(36)
    p.font.bold = True
    p.font.color.rgb = WHITE

    p2 = tf.add_paragraph()
    p2.text = "Classical ML vs. edge-aware GNNs on Tox21, under a leak-free scaffold split"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    p2.space_before = Pt(12)

    tb_foot = s1.shapes.add_textbox(Inches(1.0), Inches(5.5), Inches(11.3), Inches(1.2))
    tf_f = tb_foot.text_frame
    tf_f.word_wrap = True
    p = tf_f.paragraphs[0]
    p.text = "Presenting Team: Amogha Raj Sindur  •  Bulbul Agarwalla  •  Shourya Pratap  •  Kanishka Dubey"
    p.font.name = FONT_HEAD
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = WHITE

    p2 = tf_f.add_paragraph()
    p2.text = "Contributing Member: Aditya Yadav"
    p2.font.name = FONT_BODY
    p2.font.size = Pt(12)
    p2.font.color.rgb = RGBColor(0x94, 0xA3, 0xB8)
    p2.space_before = Pt(4)

    p3 = tf_f.add_paragraph()
    p3.text = "Newton School of Technology  |  Department of Computer Science & AI  |  Academic Year 2026"
    p3.font.name = FONT_BODY
    p3.font.size = Pt(12)
    p3.font.color.rgb = RGBColor(0x64, 0x74, 0x8B)
    p3.space_before = Pt(4)

    add_notes(s1, "Welcome the viva committee. Introduce the project: an empirical comparison of classical descriptor-based ML against edge-aware graph neural networks on the Tox21 benchmark under a leak-free Bemis-Murcko scaffold split.")

    # =========================================================================
    # SLIDE 2: PROJECT VISION (Executive Summary)
    # =========================================================================
    s2 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s2, BG_LIGHT)
    add_header(s2, "Executive summary: two paradigms, one benchmark", "PROJECT VISION", COLOR_VISION)

    add_kpi(s2, 0.8, 1.45, 2.7, 1.4, "7,823", "Cleaned compounds", "RDKit-sanitized from 7,831", COLOR_DATA)
    add_kpi(s2, 3.8, 1.45, 2.7, 1.4, "12", "Tox21 endpoints", "7 nuclear receptor + 5 stress", COLOR_ARCH)
    add_kpi(s2, 6.8, 1.45, 2.7, 1.4, "0.7488", "Phase 1 champion", "Random Forest, ROC-AUC", COLOR_TRAIN)
    add_kpi(s2, 9.8, 1.45, 2.7, 1.4, "0.7395", "Phase 2 GNN", "MolecularGNN, ROC-AUC", COLOR_BENCH)

    add_card(s2, 0.8, 3.1, 5.7, 3.9, "Phase 1: Classical ML", COLOR_ARCH)
    tb = s2.shapes.add_textbox(Inches(1.0), Inches(3.7), Inches(5.3), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    p1_items = [
        "2,048-bit Morgan fingerprints + 9 RDKit descriptors",
        "Logistic Regression, Random Forest, XGBoost",
        "Best: RF + combined features at 0.7488",
        "Limit: hand-crafted features, no bond geometry"
    ]
    for itm in p1_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    add_card(s2, 6.8, 3.1, 5.7, 3.9, "Phase 2: Graph Learning", COLOR_BENCH)
    tb = s2.shapes.add_textbox(Inches(7.0), Inches(3.7), Inches(5.3), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    p2_items = [
        "Molecules as graphs: atoms and bonds",
        "19 atom + 18 bond features, 3-hop message passing",
        "MolecularGNN reaches 0.7395, statistical parity",
        "End-to-end learning, no hand-crafted features"
    ]
    for itm in p2_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    add_notes(s2, "Executive summary comparing the two paradigms. Emphasize that GNN achieves statistical parity with the classical champion without manual feature engineering.")

    # =========================================================================
    # SLIDE 3: CONTRIBUTORS
    # =========================================================================
    s3 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s3, BG_LIGHT)
    add_header(s3, "Team and module ownership", "CONTRIBUTORS", COLOR_CONTRIB)

    team_members = [
        ("AS", "Amogha Raj Sindur", "Student 1: Data Engineering",
         ["RDKit sanitization, 7,823 mols", "Bemis-Murcko scaffold splitting", "Atom and bond featurization"], COLOR_DATA),
        ("BA", "Bulbul Agarwalla", "Student 2: GNN Architecture",
         ["MolecularGNN implementation", "Edge-conditioned message passing", "Pooling and 12-task head"], COLOR_ARCH),
        ("SP", "Shourya Pratap", "Student 3: Training Pipeline",
         ["Masked BCEWithLogits loss", "AdamW, scheduler, early stopping", "SHA-256 verified checkpoints"], COLOR_TRAIN),
        ("KD", "Kanishka Dubey", "Student 4: Evaluation & Validation",
         ["Quarantined test evaluation (N=783)", "Mask-aware ROC-AUC and PR-AUC", "Zero test leakage, figures"], COLOR_EVAL),
        ("AY", "Aditya Yadav", "Student 5: Integration & Paper",
         ["YAML experiment orchestrator", "Ablations and benchmark comparison", "IEEE manuscript and deck"], COLOR_BENCH),
    ]

    for idx, (initials, name, role, tasks, col) in enumerate(team_members):
        c_left = 0.8 + idx * 2.4
        c = s3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(c_left), Inches(1.5), Inches(2.25), Inches(5.5))
        c.fill.solid()
        c.fill.fore_color.rgb = WHITE
        c.line.color.rgb = col
        c.line.width = Pt(1.5)

        # Avatar circle
        circ = s3.shapes.add_shape(MSO_SHAPE.OVAL, Inches(c_left + 0.72), Inches(1.75), Inches(0.8), Inches(0.8))
        circ.fill.solid()
        circ.fill.fore_color.rgb = col
        circ.line.fill.background()
        p = circ.text_frame.paragraphs[0]
        p.text = initials
        p.font.name = FONT_HEAD
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER

        tb = s3.shapes.add_textbox(Inches(c_left + 0.1), Inches(2.7), Inches(2.05), Inches(4.1))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = name
        p1.font.name = FONT_HEAD
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = TEXT_MAIN
        p1.alignment = PP_ALIGN.CENTER

        p2 = tf.add_paragraph()
        p2.text = role
        p2.font.name = FONT_BODY
        p2.font.size = Pt(11)
        p2.font.bold = True
        p2.font.color.rgb = col
        p2.alignment = PP_ALIGN.CENTER
        p2.space_after = Pt(12)

        for tsk in tasks:
            p = tf.add_paragraph()
            p.text = f"• {tsk}"
            p.font.name = FONT_BODY
            p.font.size = Pt(10)
            p.font.color.rgb = TEXT_MAIN
            p.space_after = Pt(4)

    add_notes(s3, "Present team roles and division of labor across the 5 modules. Each student owned an independent component with strict interface boundaries.")

    # =========================================================================
    # SLIDE 4: DATA ENGINEERING (Curating Tox21)
    # =========================================================================
    s4 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s4, BG_LIGHT)
    add_header(s4, "Curating Tox21 and handling missing labels", "DATA ENGINEERING", COLOR_DATA)

    # 2 Stat Cards
    add_kpi(s4, 0.8, 1.45, 2.7, 1.4, "7,823", "Valid molecules", "of 7,831; 8 unparseable removed", COLOR_DATA)
    add_kpi(s4, 3.8, 1.45, 2.7, 1.4, "23.6%", "Labels missing", "untested assays", COLOR_BENCH)

    # Text Card
    add_card(s4, 0.8, 3.1, 5.7, 3.9, "Curation & Missingness Policy", COLOR_DATA)
    tb = s4.shapes.add_textbox(Inches(1.0), Inches(3.7), Inches(5.3), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    c_items = [
        "Sanitization: salt stripping, counter-ion removal, valence and tautomer normalization.",
        "Masked labels: missing values stay NaN with a boolean observation mask m in {0,1}.",
        "No zero-imputation: treating untested as inactive adds false negatives and distorts calibration."
    ]
    for itm in c_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(10)

    # Image Card
    add_card(s4, 6.8, 1.45, 5.7, 5.5, "Positive Prevalence vs. Missingness", COLOR_DATA)
    fig_p = FIG_DIR / "fig_label_statistics.png"
    if fig_p.exists():
        s4.shapes.add_picture(str(fig_p), Inches(7.0), Inches(2.1), width=Inches(5.3))

    add_notes(s4, "Student 1: Detail the RDKit sanitization and explain why missing labels are kept as masks rather than imputed as zeroes.")

    # =========================================================================
    # SLIDE 5: DATA ENGINEERING (Scaffold Splitting)
    # =========================================================================
    s5 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s5, BG_LIGHT)
    add_header(s5, "Scaffold splitting with a zero-leakage protocol", "DATA ENGINEERING", COLOR_DATA)

    add_kpi(s5, 0.8, 1.45, 3.6, 1.4, "6,258", "Training graphs", "80%, model optimization", COLOR_DATA)
    add_kpi(s5, 4.8, 1.45, 3.6, 1.4, "782", "Validation graphs", "10%, hyperparameter tuning", COLOR_ARCH)
    add_kpi(s5, 8.8, 1.45, 3.7, 1.4, "783", "Quarantined test", "10%, untouched final benchmark", COLOR_EVAL)

    add_card(s5, 0.8, 3.1, 11.7, 3.9, "Why not a random split?", COLOR_DATA)
    tb = s5.shapes.add_textbox(Inches(1.1), Inches(3.7), Inches(11.1), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    sc_items = [
        "Leakage hazard: shared ring cores in train and test let models memorize scaffolds.",
        "Bemis-Murcko cores: side chains stripped, molecules clustered by framework.",
        "Deterministic 80/10/10: clusters sorted by size and allocated whole.",
        "Train ∩ Test = ∅: Zero scaffolds overlap between partitions, so the test measures generalization to novel chemical families."
    ]
    for itm in sc_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    add_notes(s5, "Student 1: Explain Bemis-Murcko scaffold splitting. Emphasize why random splitting leads to inflated metrics due to scaffold leakage.")

    # =========================================================================
    # SLIDE 6: DATA ENGINEERING (19 Node & 18 Edge Features)
    # =========================================================================
    s6 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s6, BG_LIGHT)
    add_header(s6, "Molecular graph: 19 atom and 18 bond features", "DATA ENGINEERING", COLOR_DATA)

    add_card(s6, 0.8, 1.45, 5.7, 4.5, "19 node features (atoms)", COLOR_DATA)
    tb = s6.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.3), Inches(3.6))
    tf = tb.text_frame
    tf.word_wrap = True
    node_items = [
        "6 scalars: atomic number, degree, formal charge, aromatic, attached H, in-ring",
        "9 dims: hybridization one-hot (S, SP, SP2, SP3, SP2D, SP3D, SP3D2, other)",
        "4 dims: chirality one-hot (unspecified, CW, CCW, other)"
    ]
    for itm in node_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(10)

    add_card(s6, 6.8, 1.45, 5.7, 4.5, "18 edge features (bonds)", COLOR_DATA)
    tb = s6.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(3.6))
    tf = tb.text_frame
    tf.word_wrap = True
    edge_items = [
        "2 flags: conjugated, in-ring",
        "7 dims: bond type one-hot (single, double, triple, aromatic, dative, zero, other)",
        "9 dims: stereo one-hot (none, any, Z, E, cis, trans, atropisomers, other)"
    ]
    for itm in edge_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(10)

    # Bottom Callout Card
    add_card(s6, 0.8, 6.1, 11.7, 0.9, border_color=COLOR_DATA, bg_color=WHITE)
    tb_b = s6.shapes.add_textbox(Inches(1.0), Inches(6.2), Inches(11.3), Inches(0.7))
    p = tb_b.text_frame.paragraphs[0]
    p.text = "Bidirectional directed graph: every bond becomes two directed edges (u→v and v→u) for message passing."
    p.font.name = FONT_HEAD
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_DATA
    p.alignment = PP_ALIGN.CENTER

    add_notes(s6, "Student 1: Walk through the exact node and edge features. Point out the bidirectional edge formulation for message passing.")

    # =========================================================================
    # SLIDE 7: GNN ARCHITECTURE (MolecularGNN Overview)
    # =========================================================================
    s7 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s7, BG_LIGHT)
    add_header(s7, "MolecularGNN: an edge-conditioned MPNN", "GNN ARCHITECTURE", COLOR_ARCH)

    # Pipeline Flow Bar
    pipe_box = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.4), Inches(11.7), Inches(0.65))
    pipe_box.fill.solid()
    pipe_box.fill.fore_color.rgb = BG_DARK
    pipe_box.line.fill.background()
    p = pipe_box.text_frame.paragraphs[0]
    p.text = "SMILES (7,823)  →  Graph (19/18)  →  Encoders (128-d)  →  3x MPNN (Res+LN)  →  Mean Pool  →  Head (12 logits)"
    p.font.name = "Consolas"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    # 6 Numbered Cards
    components = [
        ("1", "Feature encoders", "19-d atom and 18-d bond features projected to 128-d"),
        ("2", "3 message-passing layers", "3-bond topological radius, edge-aware"),
        ("3", "Residual + LayerNorm", "h = LayerNorm(h + update), stable deeper layers"),
        ("4", "Dropout 10%", "Regularizes across variable molecule sizes"),
        ("5", "Mean readout", "Size-invariant molecular embedding"),
        ("6", "12-task head", "Linear projection to 12 unconstrained logits"),
    ]

    for idx, (num, title, desc) in enumerate(components):
        row = idx // 3
        col = idx % 3
        c_left = 0.8 + col * 4.0
        c_top = 2.25 + row * 2.45
        c = s7.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(c_left), Inches(c_top), Inches(3.7), Inches(2.2))
        c.fill.solid()
        c.fill.fore_color.rgb = WHITE
        c.line.color.rgb = COLOR_ARCH
        c.line.width = Pt(1.5)

        tb = s7.shapes.add_textbox(Inches(c_left + 0.2), Inches(c_top + 0.2), Inches(3.3), Inches(1.8))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = f"{num}  {title}"
        p1.font.name = FONT_HEAD
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_ARCH

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.name = FONT_BODY
        p2.font.size = Pt(12)
        p2.font.color.rgb = TEXT_MAIN
        p2.space_before = Pt(8)

    add_notes(s7, "Student 2: Present the overall MolecularGNN architecture. Explain each of the 6 key building blocks.")

    # =========================================================================
    # SLIDE 8: GNN ARCHITECTURE (Step by Step Message Passing)
    # =========================================================================
    s8 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s8, BG_LIGHT)
    add_header(s8, "Message passing and aggregation, step by step", "GNN ARCHITECTURE", COLOR_ARCH)

    steps = [
        ("1  Feature encoders",
         ["h_i⁰ = ReLU(W_v x_i + b_v)", "h_ij = ReLU(W_e e_ij + b_e)", "Atoms 19→128, bonds 18→128"]),
        ("2  Edge-conditioned message",
         ["m_ij = MLP([ h_j || h_ij ])", "Neighbor state concatenated with its bond"]),
        ("3  Permutation-invariant sum",
         ["m_i = Σ_{j∈N(i)} m_ij", "Messages summed over bonded neighbors"]),
        ("4  Residual update",
         ["h_i = LayerNorm( h_i + Dropout(  ReLU( W_u [ h_i || m_i ] ) ) )", "Skip connection preserves atom identity"]),
    ]

    for idx, (title, lines) in enumerate(steps):
        row = idx // 2
        col = idx % 2
        c_left = 0.8 + col * 6.0
        c_top = 1.45 + row * 2.85
        c = s8.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(c_left), Inches(c_top), Inches(5.7), Inches(2.65))
        c.fill.solid()
        c.fill.fore_color.rgb = WHITE
        c.line.color.rgb = COLOR_ARCH
        c.line.width = Pt(1.5)

        tb = s8.shapes.add_textbox(Inches(c_left + 0.25), Inches(c_top + 0.2), Inches(5.2), Inches(2.25))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = title
        p1.font.name = FONT_HEAD
        p1.font.size = Pt(15)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_ARCH

        for ln in lines:
            p = tf.add_paragraph()
            p.text = ln
            if "ReLU" in ln or "MLP" in ln or "Σ" in ln or "LayerNorm" in ln:
                p.font.name = "Consolas"
                p.font.size = Pt(12)
                p.font.bold = True
                p.font.color.rgb = TEXT_MAIN
                p.space_before = Pt(4)
            else:
                p.font.name = FONT_BODY
                p.font.size = Pt(11)
                p.font.color.rgb = TEXT_MUTED
                p.space_before = Pt(2)

    add_notes(s8, "Student 2: Explain the four steps of message passing: linear projections, edge-conditioned messaging, sum aggregation, and residual update with LayerNorm.")

    # =========================================================================
    # SLIDE 9: GNN ARCHITECTURE (Readout Pooling & Prediction Head)
    # =========================================================================
    s9 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s9, BG_LIGHT)
    add_header(s9, "Readout pooling and the 12-task prediction head", "GNN ARCHITECTURE", COLOR_ARCH)

    # Left: Graph Readout
    add_card(s9, 0.8, 1.45, 5.7, 5.5, "Graph-level readout", COLOR_ARCH)
    tb = s9.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "h_G = (1/|V|) Σ_i h_i"
    p.font.name = "Consolas"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_ARCH

    # 2 small KPI boxes for Mean vs Sum
    p = tf.add_paragraph()
    p.text = "Ablation Validation Loss:"
    p.font.name = FONT_HEAD
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN
    p.space_before = Pt(10)

    p = tf.add_paragraph()
    p.text = "• 0.2771 — Mean pooling"
    p.font.name = FONT_BODY
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_DATA

    p = tf.add_paragraph()
    p.text = "• 0.3047 — Sum pooling"
    p.font.name = FONT_BODY
    p.font.size = Pt(13)
    p.font.color.rgb = COLOR_EVAL

    p = tf.add_paragraph()
    p.text = "Sum pooling scales with molecule size (2 to 50+ atoms), adding size bias. Mean pooling is size-invariant."
    p.font.name = FONT_BODY
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MUTED
    p.space_before = Pt(12)

    # Right: 12-Task Head
    add_card(s9, 6.8, 1.45, 5.7, 5.5, "12-task prediction head", COLOR_ARCH)
    tb = s9.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "z = W_out h_G + b_out  →  [batch, 12]"
    p.font.name = "Consolas"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLOR_ARCH

    head_points = [
        "Raw logits: no sigmoid inside the model.",
        "Stability: BCEWithLogitsLoss uses the log-sum-exp trick.",
        "Probabilities: σ(z) applied only at evaluation.",
        "Endpoint order: fixed, NR-AR through SR-p53."
    ]
    for itm in head_points:
        p = tf.add_paragraph()
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_before = Pt(8)

    add_notes(s9, "Student 2: Explain readout pooling and why mean pooling outperforms sum pooling (size invariance). Point out why the head outputs raw logits.")

    # =========================================================================
    # SLIDE 10: TRAINING PIPELINE (Masked Loss)
    # =========================================================================
    s10 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s10, BG_LIGHT)
    add_header(s10, "Training on incomplete labels: the masked loss", "TRAINING PIPELINE", COLOR_TRAIN)

    # Top banner equation
    loss_banner = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.45), Inches(11.7), Inches(1.4))
    loss_banner.fill.solid()
    loss_banner.fill.fore_color.rgb = WHITE
    loss_banner.line.color.rgb = COLOR_TRAIN
    loss_banner.line.width = Pt(1.5)

    tb = s10.shapes.add_textbox(Inches(1.0), Inches(1.55), Inches(11.3), Inches(1.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "23.6% of label entries are unobserved, so the loss only counts observed entries:"
    p.font.name = FONT_BODY
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT_MUTED

    p2 = tf.add_paragraph()
    p2.text = "L = Σ_i Σ_k m_ik · [ −y_ik·log σ(z_ik) − (1−y_ik)·log(1−σ(z_ik)) ] / Σ m_ik"
    p2.font.name = "Consolas"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = COLOR_TRAIN
    p2.space_before = Pt(6)

    # 4 Cards for Principles
    loss_cards = [
        ("1", "Strict masking", "m_ik = 0 gives zero loss and zero gradient"),
        ("2", "Observed-only norm", "Divide by observed count, not batch × 12"),
        ("3", "Numerical stability", "BCEWithLogitsLoss with binary mask"),
        ("4", "Pos-weight support", "w_k = N_neg / N_pos for rare toxic labels"),
    ]

    for idx, (num, title, desc) in enumerate(loss_cards):
        c_left = 0.8 + idx * 3.0
        c = s10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(c_left), Inches(3.1), Inches(2.7), Inches(3.9))
        c.fill.solid()
        c.fill.fore_color.rgb = WHITE
        c.line.color.rgb = COLOR_TRAIN
        c.line.width = Pt(1.5)

        tb = s10.shapes.add_textbox(Inches(c_left + 0.15), Inches(3.3), Inches(2.4), Inches(3.5))
        tf = tb.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = f"{num}  {title}"
        p1.font.name = FONT_HEAD
        p1.font.size = Pt(14)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_TRAIN

        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.name = FONT_BODY
        p2.font.size = Pt(12)
        p2.font.color.rgb = TEXT_MAIN
        p2.space_before = Pt(10)

    add_notes(s10, "Student 3: Explain the masked loss formulation. Missing entries have m=0, ensuring zero gradient flow.")

    # =========================================================================
    # SLIDE 11: TRAINING PIPELINE (Optimization & Early Stopping)
    # =========================================================================
    s11 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s11, BG_LIGHT)
    add_header(s11, "Optimization: AdamW, scheduler and early stopping", "TRAINING PIPELINE", COLOR_TRAIN)

    add_kpi(s11, 0.8, 1.45, 2.7, 1.4, "AdamW", "Optimizer", "lr 1e-3, wd 1e-4", COLOR_TRAIN)
    add_kpi(s11, 3.8, 1.45, 2.7, 1.4, "Plateau", "LR scheduler", "factor 0.5, patience 4", COLOR_TRAIN)
    add_kpi(s11, 6.8, 1.45, 2.7, 1.4, "1.0", "Gradient clip", "max L2 norm", COLOR_TRAIN)
    add_kpi(s11, 9.8, 1.45, 2.7, 1.4, "8", "Early stopping", "patience on val loss", COLOR_TRAIN)

    add_card(s11, 0.8, 3.1, 11.7, 3.9, "Convergence safeguards", COLOR_TRAIN)
    tb = s11.shapes.add_textbox(Inches(1.1), Inches(3.7), Inches(11.1), Inches(3.1))
    tf = tb.text_frame
    tf.word_wrap = True
    opt_items = [
        "LR schedule: halves the rate (1e-3 → 5e-4 → 2.5e-4) after 4 flat epochs.",
        "Clipping: L2 norm capped at 1.0 against exploding gradients.",
        "Early stop: halts after 8 epochs without improvement on 6,258 training graphs.",
        "Logging: batch size 32, per-epoch CSV and JSON logs."
    ]
    for itm in opt_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(8)

    add_notes(s11, "Student 3: Detail the optimization choices: AdamW, learning rate plateaus, gradient clipping at 1.0, and early stopping.")

    # =========================================================================
    # SLIDE 12: TRAINING PIPELINE (Checkpointing & Reproducibility)
    # =========================================================================
    s12 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s12, BG_LIGHT)
    add_header(s12, "Checkpoint integrity and reproducibility", "TRAINING PIPELINE", COLOR_TRAIN)

    add_card(s12, 0.8, 1.45, 5.7, 5.5, "Atomic checkpointing", COLOR_TRAIN)
    tb = s12.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    ckpt_items = [
        "Best model only: saved on a new validation minimum.",
        "Atomic writes: temp file then rename, safe on interrupt.",
        "Full metadata: weights, optimizer, epoch, val loss, hyperparameters.",
        "SHA-256 verified: bit-for-bit weight preservation."
    ]
    for itm in ckpt_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(10)

    add_card(s12, 6.8, 1.45, 5.7, 5.5, "Multi-seed verification", COLOR_TRAIN)
    tb = s12.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    seed_items = [
        "Global seeding: seed 42 across Python, NumPy, PyTorch CPU and CUDA.",
        "Seed 43 check: val loss 0.2820 vs 0.2771 (Δ < 0.005).",
        "Deterministic inference: repeat runs differ by < 1e-6."
    ]
    for itm in seed_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = f"• {itm}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN
        p.space_after = Pt(10)

    add_notes(s12, "Student 3: Explain atomic checkpoint writing and SHA-256 checksums. Show multi-seed stability (seed 42 vs 43).")

    # =========================================================================
    # SLIDE 13: EVALUATION & VALIDATION (Quarantined Test Protocol)
    # =========================================================================
    s13 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s13, BG_LIGHT)
    add_header(s13, "Independent test evaluation under quarantine", "EVALUATION & VALIDATION", COLOR_EVAL)

    add_kpi(s13, 0.8, 1.45, 3.6, 1.4, "783", "Quarantined test graphs", "zero scaffold overlap", COLOR_EVAL)
    add_kpi(s13, 4.8, 1.45, 3.6, 1.4, "7,175", "Observed labels", "76.4% assay density", COLOR_BENCH)
    add_kpi(s13, 8.8, 1.45, 3.7, 1.4, "0.0", "Data leakage", "test never seen in training", COLOR_DATA)

    # 4 Numbered Cards
    eval_cards = [
        ("1", "Strict quarantine:", "Test molecules never used for backprop, LR decisions or early stopping."),
        ("2", "Frozen model:", "model.eval() with torch.no_grad(); no fine-tuning or threshold tuning."),
        ("3", "Mask-aware metrics:", "Computed over the 7,175 observed labels only."),
        ("4", "Threshold-free, with uncertainty:", "ROC-AUC and PR-AUC, macro mean over 12 endpoints with SEM and 95% CI."),
    ]

    for idx, (num, title, desc) in enumerate(eval_cards):
        c_top = 3.1 + idx * 0.95
        c = s13.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(c_top), Inches(11.7), Inches(0.85))
        c.fill.solid()
        c.fill.fore_color.rgb = WHITE
        c.line.color.rgb = COLOR_EVAL
        c.line.width = Pt(1.5)

        tb = s13.shapes.add_textbox(Inches(1.0), Inches(c_top + 0.12), Inches(11.3), Inches(0.6))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        p.text = f"{num}  {title}  {desc}"
        p.font.name = FONT_BODY
        p.font.size = Pt(13)
        p.font.color.rgb = TEXT_MAIN

    add_notes(s13, "Student 4: Emphasize the strict quarantine protocol. The test set was never accessed during model development.")

    # =========================================================================
    # SLIDE 14: EVALUATION & VALIDATION (ROC-AUC Across 12 Endpoints)
    # =========================================================================
    s14 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s14, BG_LIGHT)
    add_header(s14, "ROC-AUC across all 12 endpoints", "EVALUATION & VALIDATION", COLOR_EVAL)

    fig_roc = FIG_DIR / "endpoint_roc_auc.png"
    if fig_roc.exists():
        s14.shapes.add_picture(str(fig_roc), Inches(0.8), Inches(1.4), width=Inches(11.7))

    # Bottom summary card
    add_card(s14, 0.8, 5.75, 11.7, 1.35, border_color=COLOR_EVAL, bg_color=WHITE)
    tb = s14.shapes.add_textbox(Inches(1.0), Inches(5.82), Inches(11.3), Inches(1.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Macro mean 0.7395 ± 0.0182 (SEM)  |  95% CI [0.6995, 0.7795]"
    p.font.name = FONT_HEAD
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_EVAL

    p = tf.add_paragraph()
    p.text = "Best assays:  SR-MMP 0.849 (mitochondrial potential)  •  NR-AhR 0.832 (aromatic rings)  •  NR-Aromatase 0.803 (steroid synthesis)"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN
    p.space_before = Pt(2)

    p = tf.add_paragraph()
    p.text = "Hardest assay:  NR-AR 0.590 (only 1.5% positives, severe class scarcity)"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MUTED

    add_notes(s14, "Student 4: Walk through the ROC-AUC bar chart. Highlight macro mean of 0.7395 and performance on best vs hardest endpoints.")

    # =========================================================================
    # SLIDE 15: EVALUATION & VALIDATION (PR-AUC vs Random Baseline)
    # =========================================================================
    s15 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s15, BG_LIGHT)
    add_header(s15, "Precision-recall against the random baseline", "EVALUATION & VALIDATION", COLOR_EVAL)

    # Top KPI box
    add_card(s15, 0.8, 1.4, 11.7, 0.85, border_color=COLOR_EVAL, bg_color=WHITE)
    tb = s15.shapes.add_textbox(Inches(1.0), Inches(1.48), Inches(11.3), Inches(0.7))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = "0.3118 Macro PR-AUC ± 0.0331 (SEM)  |  95% CI [0.2389, 0.3847]"
    p.font.name = FONT_HEAD
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_EVAL

    p = tf.add_paragraph()
    p.text = "Random baseline equals prevalence (1.5% to 15.2%)  •  GNN beats baseline on every endpoint  •  SR-MMP: 0.589 vs 0.152 (3.9x gain)"
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    fig_pr = FIG_DIR / "endpoint_pr_auc.png"
    if fig_pr.exists():
        s15.shapes.add_picture(str(fig_pr), Inches(0.8), Inches(2.35), width=Inches(11.7))

    add_notes(s15, "Student 4: Explain PR-AUC. The red diamonds show baseline prevalence; the green bars show GNN performance beating it on all 12 endpoints.")

    # =========================================================================
    # SLIDE 16: EVALUATION & VALIDATION (Prediction Distributions)
    # =========================================================================
    s16 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s16, BG_LIGHT)
    add_header(s16, "Prediction distributions and threshold collapse", "EVALUATION & VALIDATION", COLOR_EVAL)

    fig_dist = FIG_DIR / "prediction_probability_distributions.png"
    if fig_dist.exists():
        s16.shapes.add_picture(str(fig_dist), Inches(0.8), Inches(1.4), width=Inches(6.8))

    add_card(s16, 7.8, 1.4, 4.7, 5.7, "What the plot shows", COLOR_EVAL)
    tb = s16.shapes.add_textbox(Inches(8.0), Inches(2.0), Inches(4.3), Inches(4.9))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Blue: true negatives cluster near 0.0\nRed: true positives shift to higher probabilities"
    p.font.name = FONT_BODY
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MAIN

    p = tf.add_paragraph()
    p.text = "Threshold collapse at t = 0.5:"
    p.font.name = FONT_HEAD
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = COLOR_EVAL
    p.space_before = Pt(12)

    p = tf.add_paragraph()
    p.text = "• Balanced accuracy 0.5599, F1 0.1777\n• With ~5% prevalence, positives peak near 0.2 to 0.4\n• So rank metrics (ROC-AUC) are the right yardstick"
    p.font.name = FONT_BODY
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_MAIN
    p.space_before = Pt(4)

    add_notes(s16, "Student 4: Show continuous probability distributions. Explain threshold collapse: fixing t=0.5 fails because calibrated positive probabilities naturally peak around 0.2-0.4.")

    # =========================================================================
    # SLIDE 17: INTEGRATION & BENCHMARKS (Config-Driven Experiments)
    # =========================================================================
    s17 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s17, BG_LIGHT)
    add_header(s17, "One-command, config-driven experiments", "INTEGRATION & BENCHMARKS", COLOR_BENCH)

    add_card(s17, 0.8, 1.45, 5.7, 5.5, "Modular Experiment Workflow", COLOR_BENCH)
    tb = s17.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True
    mod_items = [
        "1  run_experiment.py — unified CLI and programmatic runner",
        "2  experiment_config.py — typed dataclass schema for YAML configs",
        "3  experiment_registry.py — logs git hash, runtime, device, metrics",
        "4  configs/phase2/ — 8 YAML configs: baseline and ablations",
        "",
        "Links data, model, training and evaluation modules seamlessly."
    ]
    for itm in mod_items:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = itm
        if itm.startswith(("1", "2", "3", "4")):
            p.font.name = FONT_HEAD
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = COLOR_BENCH
            p.space_before = Pt(6)
        else:
            p.font.name = FONT_BODY
            p.font.size = Pt(12)
            p.font.color.rgb = TEXT_MAIN

    # Code block on right
    add_card(s17, 6.8, 1.45, 5.7, 5.5, "configs/phase2/gnn_baseline.yaml", COLOR_BENCH)
    tb_c = s17.shapes.add_textbox(Inches(7.0), Inches(2.1), Inches(5.3), Inches(4.6))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True
    yaml_lines = [
        "experiment_id: gnn_baseline",
        "",
        "# architecture",
        "hidden_dim: 128",
        "num_layers: 3",
        "pooling: mean",
        "dropout: 0.1",
        "residual: true",
        "norm: layer",
        "",
        "# optimization",
        "epochs: 10",
        "batch_size: 32",
        "learning_rate: 0.001",
        "weight_decay: 0.0001",
        "early_stopping_patience: 8",
        "",
        "# reproducibility",
        "seed: 42"
    ]
    for ln in yaml_lines:
        p = tf_c.add_paragraph() if tf_c.paragraphs[0].text else tf_c.paragraphs[0]
        p.text = ln
        p.font.name = "Consolas"
        p.font.size = Pt(10)
        p.font.color.rgb = COLOR_BENCH if ln.startswith(("#", "experiment_id")) else TEXT_MAIN

    add_notes(s17, "Student 5: Present the experiment runner and declarative YAML configuration. Explain how it orchestrates the work of Students 1-4.")

    # =========================================================================
    # SLIDE 18: INTEGRATION & BENCHMARKS (Ablations)
    # =========================================================================
    s18 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s18, BG_LIGHT)
    add_header(s18, "Ablations: every design choice was tested", "INTEGRATION & BENCHMARKS", COLOR_BENCH)

    fig_abl = FIG_DIR / "ablation_validation_loss_comparison.png"
    if fig_abl.exists():
        s18.shapes.add_picture(str(fig_abl), Inches(0.8), Inches(1.4), width=Inches(11.7))

    # Bottom ablation summary
    add_card(s18, 0.8, 5.75, 11.7, 1.35, border_color=COLOR_BENCH, bg_color=WHITE)
    tb = s18.shapes.add_textbox(Inches(1.0), Inches(5.82), Inches(11.3), Inches(1.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Validation loss, lower is better:"
    p.font.name = FONT_HEAD
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    abl_items = [
        "1  Pooling: Mean 0.2771 vs Sum 0.3047",
        "2  Depth: L=3 0.2771 vs L=2 0.2783",
        "3  Width: d=128 0.2771 vs d=64 0.2809",
        "4  Seeds: 42: 0.2771 vs 43: 0.2820"
    ]
    p = tf.add_paragraph()
    p.text = "   •   ".join(abl_items)
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = COLOR_BENCH
    p.space_before = Pt(4)

    add_notes(s18, "Student 5: Walk through the ablation results. Show that depth=3, hidden_dim=128, and mean pooling were experimentally validated.")

    # =========================================================================
    # SLIDE 19: INTEGRATION & BENCHMARKS (Cross-Paradigm Benchmark)
    # =========================================================================
    s19 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s19, BG_LIGHT)
    add_header(s19, "Cross-paradigm benchmark: classical ML vs GNN", "INTEGRATION & BENCHMARKS", COLOR_BENCH)

    fig_mc = FIG_DIR / "fig_model_comparison.png"
    if fig_mc.exists():
        s19.shapes.add_picture(str(fig_mc), Inches(0.8), Inches(1.4), width=Inches(6.2))

    # Right Card: Table + Insights
    add_card(s19, 7.2, 1.4, 5.3, 5.7, border_color=COLOR_BENCH, bg_color=WHITE)
    tb = s19.shapes.add_textbox(Inches(7.4), Inches(1.5), Inches(4.9), Inches(5.4))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Model                      ROC-AUC    PR-AUC\n" \
             "LR (Morgan)                0.6865     0.2866\n" \
             "XGBoost (Comb.)            0.7456     0.3789\n" \
             "RF champion (Comb.)        0.7488     0.3891\n" \
             "MolecularGNN               0.7395     0.3118"
    p.font.name = "Consolas"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = TEXT_MAIN

    p = tf.add_paragraph()
    p.text = "Overlapping CIs:"
    p.font.name = FONT_HEAD
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_BENCH
    p.space_before = Pt(10)

    p = tf.add_paragraph()
    p.text = "RF [0.7125, 0.7851] vs GNN [0.6995, 0.7795]; Δ = 0.0093 is within standard error."
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf.add_paragraph()
    p.text = "Representation learning:"
    p.font.name = FONT_HEAD
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_BENCH
    p.space_before = Pt(8)

    p = tf.add_paragraph()
    p.text = "GNN matches a 2,057-feature engineered RF using raw atom and bond features only."
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    p = tf.add_paragraph()
    p.text = "PR-AUC gap:"
    p.font.name = FONT_HEAD
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLOR_BENCH
    p.space_before = Pt(8)

    p = tf.add_paragraph()
    p.text = "RF keeps higher PR-AUC via axis-aligned splits on rare positive fingerprint bits."
    p.font.name = FONT_BODY
    p.font.size = Pt(11)
    p.font.color.rgb = TEXT_MAIN

    add_notes(s19, "Student 5: The headline comparison. Point out the statistical parity in ROC-AUC and explain the PR-AUC trade-off.")

    # =========================================================================
    # SLIDE 20: Conclusion and scientific contributions
    # =========================================================================
    s20 = prs.slides.add_slide(prs.slide_layouts[6])
    set_bg(s20, BG_DARK)

    bar = s20.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = COLOR_VISION
    bar.line.fill.background()

    tb = s20.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.3), Inches(0.8))
    p = tb.text_frame.paragraphs[0]
    p.text = "Conclusion and scientific contributions"
    p.font.name = FONT_HEAD
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE

    # 3 Summary Cards
    cards_data = [
        ("1  Rigorous benchmark",
         ["7,823 molecules curated via RDKit",
          "Zero-leakage scaffold split (6,258 / 782 / 783)",
          "Masked multi-task training, no imputation",
          "212 automated tests passed"], COLOR_DATA),
        ("2  Parity achieved",
         ["Phase 1 RF: 0.7488 ± 0.0185",
          "Phase 2 GNN: 0.7395 ± 0.0182",
          "Overlapping 95% confidence intervals",
          "No hand-crafted features for the GNN"], COLOR_ARCH),
        ("3  Future horizons",
         ["3D equivariant GNNs (EGNN, SchNet)",
          "Self-supervised pre-training on ChEMBL/ZINC",
          "RF + GNN ensemble for complementary signal",
          "Extended ADMET endpoint evaluations"], COLOR_BENCH),
    ]

    for idx, (title, items, col) in enumerate(cards_data):
        c_left = 1.0 + idx * 3.85
        c = s20.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(c_left), Inches(1.8), Inches(3.6), Inches(4.5))
        c.fill.solid()
        c.fill.fore_color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        c.line.color.rgb = col
        c.line.width = Pt(1.5)

        tb = s20.shapes.add_textbox(Inches(c_left + 0.2), Inches(2.0), Inches(3.2), Inches(4.1))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = FONT_HEAD
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = col

        for itm in items:
            p = tf.add_paragraph()
            p.text = f"• {itm}"
            p.font.name = FONT_BODY
            p.font.size = Pt(12)
            p.font.color.rgb = TEXT_LIGHT
            p.space_before = Pt(8)

    tb_end = s20.shapes.add_textbox(Inches(1.0), Inches(6.5), Inches(11.3), Inches(0.5))
    p = tb_end.text_frame.paragraphs[0]
    p.text = "Thank you. Questions welcome."
    p.font.name = FONT_HEAD
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLOR_VISION
    p.alignment = PP_ALIGN.CENTER

    add_notes(s20, "Conclude the presentation. Thank the viva committee and faculty. Open the floor for questions.")

    prs.save(str(OUT_PATH))
    print(f"Generated {OUT_PATH} successfully! ({len(prs.slides)} slides)")


if __name__ == "__main__":
    create_deck()
