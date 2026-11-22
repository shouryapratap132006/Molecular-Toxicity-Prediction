"""Build the Phase 2 Final Project Presentation deck (18 slides).

Student 5 — Integration, Experiments, Final IEEE Report & Presentation Owner.

Slides:
  1. Title & Team Roles
  2. Problem & Motivation
  3. Dataset: Tox21 Curation & Endpoints
  4. End-to-End Two-Phase Pipeline
  5. Phase 1: Feature Engineering (Morgan FP + Descriptors)
  6. Phase 1: Classical ML Models & Scaffold Splitting
  7. Phase 1: Benchmark Results (Table)
  8. Phase 2: Molecular Graph Representation
  9. Phase 2: GNN Architecture (MolecularGNN)
 10. Phase 2: Message Passing Mechanism
 11. Phase 2: Training Pipeline & Mask-Aware Loss
 12. Phase 2: Ablation Matrix Results
 13. Phase 2: Final Test Evaluation
 14. Cross-Phase Comparison: Phase 1 vs Phase 2
 15. Endpoint-Level Analysis
 16. Discussion & Limitations
 17. Conclusion & Future Work
 18. Q&A / Thank You
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt, Emu

OUT = Path("reports/final_presentation/phase2_final_presentation.pptx")

# Color palette
INK = RGBColor(0x1B, 0x2A, 0x41)
INK_SOFT = RGBColor(0x2C, 0x3E, 0x5A)
TEXT = RGBColor(0x22, 0x2B, 0x36)
MUTED = RGBColor(0x5C, 0x67, 0x73)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TINT = RGBColor(0xF1, 0xF4, 0xF8)
BLUE = RGBColor(0x00, 0x72, 0xB2)
ORANGE = RGBColor(0xE6, 0x9F, 0x00)
GREEN = RGBColor(0x00, 0x9E, 0x73)
VERMIL = RGBColor(0xD5, 0x5E, 0x00)
HEAD_FONT, BODY_FONT = "Cambria", "Calibri"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _add_bg(slide, color=TINT):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def _tb(slide, left, top, width, height):
    return slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))


def _set_para(para, text, font_name=BODY_FONT, size=18, bold=False, color=TEXT, align=PP_ALIGN.LEFT):
    para.text = text
    para.font.name = font_name
    para.font.size = Pt(size)
    para.font.bold = bold
    para.font.color.rgb = color
    para.alignment = align
    para.space_after = Pt(4)


def _add_title_bar(slide, text, subtitle=None):
    from pptx.enum.shapes import MSO_SHAPE
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, Inches(1.15)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = INK
    shape.line.fill.background()

    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.name = HEAD_FONT
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.LEFT
    tf.margin_left = Inches(0.6)
    tf.margin_top = Inches(0.15)

    if subtitle:
        tb = _tb(slide, 0.6, 0.75, 10, 0.4)
        _set_para(tb.text_frame.paragraphs[0], subtitle, size=14, color=RGBColor(0xA0, 0xB0, 0xC0))


def _bullet_list(slide, left, top, width, height, items, size=16, bold_first=False):
    tb = _tb(slide, left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = item
        p.font.name = BODY_FONT
        p.font.size = Pt(size)
        p.font.color.rgb = TEXT
        p.space_after = Pt(6)
        p.level = 0
        if bold_first and i == 0:
            p.font.bold = True


def _notes(slide, text):
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = text


# ── Slide 1: Title ──────────────────────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl, INK)

tb = _tb(sl, 1.0, 1.5, 11.3, 1.5)
_set_para(tb.text_frame.paragraphs[0],
          "Molecular Toxicity Prediction via Chemical\nFingerprints and Molecular Graph Networks",
          font_name=HEAD_FONT, size=36, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

tb = _tb(sl, 1.0, 3.3, 11.3, 0.6)
_set_para(tb.text_frame.paragraphs[0],
          "B.Tech Research Project — Phase 1 & Phase 2 Final Presentation",
          size=20, color=RGBColor(0xA0, 0xC0, 0xE0), align=PP_ALIGN.CENTER)

team = [
    "Amogha Raj Sindur  •  Bulbul Agarwalla  •  Shourya Pratap  •  Kanishka Dubey  •  Aditya Yadav",
    "Department of Computer Science & AI — Newton School of Technology",
]
tb = _tb(sl, 1.0, 4.3, 11.3, 1.2)
tf = tb.text_frame
tf.word_wrap = True
for i, t in enumerate(team):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = t
    p.font.name = BODY_FONT
    p.font.size = Pt(16) if i == 0 else Pt(14)
    p.font.color.rgb = RGBColor(0xCC, 0xDD, 0xEE) if i == 0 else MUTED
    p.alignment = PP_ALIGN.CENTER

_notes(sl, "Welcome and introduce team members. Mention the two-phase structure of the project.")


# ── Slide 2: Problem & Motivation ───────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Problem Statement & Motivation")

items = [
    "▸ Animal-based toxicity testing is slow (~2 years), expensive (~$2M/compound), and raises ethical concerns",
    "▸ U.S. NRC (2007) proposed shifting to high-throughput in vitro screening of toxicity pathways",
    "▸ Tox21 programme: screened ~10,000 chemicals across 12 nuclear-receptor & stress-response assays",
    "▸ Challenge: Can computational models predict toxicity from molecular structure alone?",
    "",
    "Key ML Challenges in Tox21:",
    "  • Extreme class imbalance — only 1.5–15% of compounds are active/toxic per endpoint",
    "  • Sparse label matrix — 23.6% of assay labels are missing (untested compounds)",
    "  • Structural clustering — analogues inflate accuracy under random splits",
    "  • Solution: Bemis–Murcko scaffold split guarantees out-of-distribution generalization",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.5, items)

_notes(sl, "Explain why in-silico toxicity prediction matters. Emphasize the three ML challenges: imbalance, missingness, and scaffold clustering.")


# ── Slide 3: Dataset Curation & Endpoints ────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Tox21 Dataset: Curation & Canonical Endpoints")

items = [
    "Dataset: 7,831 raw → 7,823 cleaned molecules after RDKit sanitization (8 removed)",
    "",
    "12 Toxicity Endpoints (7 Nuclear Receptor + 5 Stress Response):",
    "  NR: AR, AR-LBD, AhR, Aromatase, ER, ER-LBD, PPAR-gamma",
    "  SR: ARE, ATAD5, HSE, MMP, p53",
    "",
    "Label Handling:",
    "  • Missing labels preserved as NaN with boolean mask — never zero-imputed",
    "  • Observed assay density: 76.4% (valid labels / total possible)",
    "  • Positive prevalence range: 1.5% (NR-AR) to 15.2% (SR-MMP)",
    "",
    "Canonical Bemis–Murcko Scaffold Split:",
    "  • Train: 6,258 (80%) | Validation: 782 (10%) | Test: 783 (10%)",
    "  • Zero scaffold overlap between any two partitions",
    "  • Test set quarantined — never seen during training or tuning",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.8, items, size=15)

_notes(sl, "Walk through the curation pipeline. Stress that missing labels are masked, not imputed. Explain scaffold splitting vs random splitting.")


# ── Slide 4: Two-Phase Pipeline ──────────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "End-to-End Two-Phase Experimental Pipeline")

items = [
    "Phase 1 — Classical ML Baseline:",
    "  SMILES → RDKit Sanitization → Morgan FP (2048-bit) + 9 Descriptors →",
    "  Scaffold Split → LR / RF / XGBoost → Masked BCE → ROC-AUC & PR-AUC",
    "",
    "Phase 2 — Graph Neural Network:",
    "  SMILES → RDKit → Atom Graph (19 node feat + 18 edge feat) →",
    "  Scaffold Split → MolecularGNN (Edge-MPNN) → Masked BCE → ROC-AUC & PR-AUC",
    "",
    "Shared Infrastructure:",
    "  • Identical canonical scaffold split across both phases",
    "  • Same 783 quarantined test molecules for final evaluation",
    "  • Same 12 endpoint ordering and mask-aware metric computation",
    "  • Reproducible seeds (42), deterministic data loading",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.5, items, size=15)

_notes(sl, "Emphasize that both phases share the same scaffold split, test set, and evaluation protocol for fair comparison.")


# ── Slide 5: Phase 1 Feature Engineering ─────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Phase 1: Feature Engineering — Morgan Fingerprints & Descriptors")

items = [
    "1. Morgan Circular Fingerprints (ECFP4):",
    "   • 2,048-bit binary vectors, radius=2",
    "   • Each bit encodes presence/absence of a circular molecular substructure",
    "   • Captures local chemical environment within 2-bond neighborhood of each atom",
    "",
    "2. Physicochemical Descriptors (9 features):",
    "   • MolWt, LogP, TPSA, HBD, HBA, NumRotatableBonds, RingCount, HeavyAtomCount, FractionCSP3",
    "   • StandardScaler fitted on training data only (no test leakage)",
    "",
    "3. Combined Representation (2,057 dimensions):",
    "   • Concatenation of Morgan FP + scaled descriptors",
    "   • Combines structural topology with global physicochemical context",
    "   • Improved ROC-AUC on 11/12 endpoints vs fingerprints alone",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.5, items, size=15)

_notes(sl, "Explain Morgan fingerprints with the circular substructure concept. Note that combining FP + descriptors consistently helped.")


# ── Slide 6: Phase 1 Models ─────────────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Phase 1: Classical ML Models & Evaluation Protocol")

items = [
    "Three Model Architectures (× 3 representations = 9 configurations):",
    "",
    "  1. L2-Regularized Logistic Regression — linear baseline",
    "  2. Random Forest (500 trees, balanced subsampling) — ensemble of decision trees",
    "  3. XGBoost (max_depth=6, lr=0.05) — gradient-boosted decision trees",
    "",
    "Multi-Task Training:",
    "  • Independent binary classifier per endpoint",
    "  • Missing labels excluded via boolean masking (not zero-filled)",
    "",
    "Evaluation Metrics (threshold-free, rank-based):",
    "  • ROC-AUC: Probability that a random positive ranks above a random negative",
    "  • PR-AUC (Average Precision): Area under precision-recall curve",
    "  • Both metrics robust to class imbalance vs simple accuracy",
    "  • Macro-averaged across 12 endpoints ± SEM",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.5, items, size=15)

_notes(sl, "Stress why we use ROC-AUC and PR-AUC instead of accuracy. Accuracy is misleading when 97% of labels are negative.")


# ── Slide 7: Phase 1 Results ────────────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Phase 1: Benchmark Results — All 9 Configurations")

# Build a simple table-like text
header = "Model                        Representation      Macro ROC-AUC      Macro PR-AUC"
rows = [
    "Logistic Regression          Morgan FP           0.6865 ± 0.0210    0.2866 ± 0.0384",
    "Logistic Regression          Descriptors         0.5769 ± 0.0255    0.1721 ± 0.0196",
    "Logistic Regression          Combined            0.6823 ± 0.0215    0.2740 ± 0.0343",
    "Random Forest                Morgan FP           0.7312 ± 0.0202    0.3617 ± 0.0442",
    "Random Forest                Descriptors         0.6265 ± 0.0215    0.2130 ± 0.0271",
    "★ Random Forest              Combined            0.7488 ± 0.0185    0.3891 ± 0.0442",
    "XGBoost                      Morgan FP           0.7261 ± 0.0177    0.3400 ± 0.0381",
    "XGBoost                      Descriptors         0.6291 ± 0.0225    0.2250 ± 0.0313",
    "XGBoost                      Combined            0.7456 ± 0.0175    0.3789 ± 0.0421",
]

tb = _tb(sl, 0.6, 1.5, 12, 0.5)
_set_para(tb.text_frame.paragraphs[0], header, font_name="Consolas", size=13, bold=True, color=INK)

tb = _tb(sl, 0.6, 2.0, 12, 5.0)
tf = tb.text_frame
tf.word_wrap = True
for i, row in enumerate(rows):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = row
    p.font.name = "Consolas"
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT
    if "★" in row:
        p.font.bold = True
        p.font.color.rgb = BLUE
    p.space_after = Pt(4)

tb = _tb(sl, 0.6, 6.3, 12, 0.5)
_set_para(tb.text_frame.paragraphs[0],
          "Champion: Random Forest + Combined Features — Macro ROC-AUC 0.7488 ± 0.0185",
          size=16, bold=True, color=BLUE, align=PP_ALIGN.CENTER)

_notes(sl, "Walk through the 9 configurations. Highlight that combined features consistently improve both tree ensembles.")


# ── Slide 8: Phase 2 Molecular Graph Representation ─────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Phase 2: From Fingerprints to Molecular Graphs")

items = [
    "Key Idea: Instead of fixed-length fingerprints, represent each molecule as a graph",
    "",
    "Molecular Graph G = (V, E):",
    "  • Nodes (V) = Atoms → 19-dimensional feature vectors",
    "     – Atomic number (one-hot: C, N, O, S, F, P, Cl, Br, I, other)",
    "     – Degree (0–5), formal charge, aromaticity, hydrogens, hybridization",
    "  • Edges (E) = Bonds → 18-dimensional feature vectors",
    "     – Bond type (single, double, triple, aromatic), conjugation, ring membership, stereo",
    "",
    "Advantages over fingerprints:",
    "  • No information loss from hashing collisions",
    "  • Preserves molecular topology and connectivity",
    "  • Learnable, task-specific representations (no manual engineering)",
    "",
    "Graph Preprocessing: 7,823 molecules → 6,258 train + 782 val + 783 test graphs",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.8, items, size=14)

_notes(sl, "Explain the transition from fingerprints to graphs. Emphasize that the same molecules and splits are used.")


# ── Slide 9: GNN Architecture ───────────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Phase 2: MolecularGNN — Edge-Aware Message Passing Architecture")

items = [
    "Architecture Components:",
    "",
    "1. Feature Encoders — Linear projection of raw atom/bond features to hidden dim (d=128)",
    "     h_i⁰ = ReLU(W_v · x_i + b_v),   h_ij^e = ReLU(W_e · e_ij + b_e)",
    "",
    "2. Edge-Conditioned Message Passing (L=3 layers) —",
    "     Message: m_ij = MLP([h_j || h_ij^e])     (concatenate atom + bond features)",
    "     Aggregate: m̄_i = Σ_j∈N(i) m_ij             (sum over neighbors)",
    "     Update: h_i^l = LayerNorm(h_i^(l-1) + Dropout(ReLU(W_u · [h_i^(l-1) || m̄_i])))",
    "",
    "3. Graph-Level Mean Pooling — h_G = (1/|V|) Σ h_i^L",
    "     Size-invariant: molecules with 2 atoms pooled same way as 50-atom molecules",
    "",
    "4. Multi-Task Head — 12 independent logits: ẑ = W_out · h_G + b_out",
    "     Output: [batch_size, 12] raw logits (no sigmoid — handled by BCEWithLogitsLoss)",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.8, items, size=13)

_notes(sl, "Walk through the four architectural components. Stress residual connections, layer norm, and why mean pooling is preferred.")


# ── Slide 10: Message Passing Mechanism ──────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Phase 2: How Message Passing Works on Molecules")

items = [
    "Intuition: Each atom 'talks to' its neighbors through bonds",
    "",
    "Layer 1: Each atom sees its immediate bonded neighbors (1-hop)",
    "  → Carbon learns it's bonded to an OH group",
    "",
    "Layer 2: Information propagates 2 bonds away (2-hop)",
    "  → The carbon now 'knows' about the whole functional group",
    "",
    "Layer 3: Information covers 3-bond radius (3-hop)",
    "  → Captures pharmacophore-scale patterns, ring systems, conjugation",
    "",
    "Key Design Decisions:",
    "  • Edge features (bond type) are incorporated into messages — critical for chemistry",
    "  • Residual connections prevent information loss in deeper layers",
    "  • Layer normalization stabilizes training across variable molecular sizes",
    "  • Dropout (10%) prevents overfitting on small training set (6,258 graphs)",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.8, items, size=14)

_notes(sl, "Use an analogy: message passing is like a game of telephone along chemical bonds. After 3 rounds, every atom knows its 3-hop neighborhood.")


# ── Slide 11: Training Pipeline ──────────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Phase 2: Training Pipeline — Masked Loss & Optimization")

items = [
    "Mask-Aware Binary Cross-Entropy Loss:",
    "  • For each molecule and endpoint: only compute loss where label exists (mask=1)",
    "  • Missing labels (mask=0) contribute zero gradient — no false negative bias",
    "  • Loss = Σ(mask_k × BCE(y_k, σ(ẑ_k))) / Σ(mask_k)",
    "",
    "Optimization Configuration:",
    "  • Optimizer: Adam (lr=0.001, weight_decay=0.0001)",
    "  • Scheduler: ReduceLROnPlateau (factor=0.5, patience=4)",
    "  • Gradient clipping: max_norm=1.0",
    "  • Early stopping: patience=8 epochs on validation loss",
    "  • Batch size: 32 molecular graphs per mini-batch",
    "",
    "Reproducibility Controls:",
    "  • Fixed seed (42) for PyTorch, NumPy, Python random",
    "  • Deterministic collation and batching",
    "  • SHA-256 verified checkpoint saving and loading",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.8, items, size=14)

_notes(sl, "Explain masked loss carefully — this is a common viva question. The mask ensures we never train on missing labels.")


# ── Slide 12: Ablation Matrix ────────────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Phase 2: Ablation Study — Architecture & Optimization Sensitivity")

header2 = "Experiment             Seed  Dim(d)  Layers(L)  Pooling  PosWt  Best Val Loss"
rows2 = [
    "★ gnn_baseline         42    128     3          Mean     No     0.2771",
    "  gnn_depth2            42    128     2          Mean     No     0.2783",
    "  gnn_hidden64          42     64     3          Mean     No     0.2809",
    "  gnn_seed43            43    128     3          Mean     No     0.2820",
    "  gnn_sum_pooling       42    128     3          Sum      No     0.3047",
    "  gnn_pos_weighted      42    128     3          Mean     Yes    1.4557*",
]
findings = [
    "Findings:",
    "• Mean pooling >> Sum pooling (0.2771 vs 0.3047) — sum pooling introduces molecular-size variance",
    "• 3-hop > 2-hop message passing (0.2771 vs 0.2783) — captures functional group patterns",
    "• d=128 > d=64 (0.2771 vs 0.2809) — multi-task learning needs sufficient capacity",
    "• Seed 42 vs 43: Δ < 0.005 — confirms optimization stability",
    "• *Loss scale differs for pos_weighted due to class weighting factors (w ≈ 5.6–39.4)",
]

tb = _tb(sl, 0.4, 1.5, 12.5, 0.4)
_set_para(tb.text_frame.paragraphs[0], header2, font_name="Consolas", size=12, bold=True, color=INK)

tb = _tb(sl, 0.4, 1.9, 12.5, 2.5)
tf = tb.text_frame
tf.word_wrap = True
for i, row in enumerate(rows2):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = row
    p.font.name = "Consolas"
    p.font.size = Pt(12)
    p.font.color.rgb = BLUE if "★" in row else TEXT
    p.font.bold = "★" in row
    p.space_after = Pt(3)

_bullet_list(sl, 0.4, 4.2, 12.5, 3, findings, size=13)

_notes(sl, "Walk through each ablation and why it matters. The key takeaway is that architecture choices are validated, not arbitrary.")


# ── Slide 13: Final Test Evaluation ──────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Phase 2: Final Quarantined Test Evaluation (N=783)")

items = [
    "Evaluation Protocol:",
    "  • Best validation checkpoint loaded in eval() mode — no retraining",
    "  • Quarantined test set (783 graphs) seen for the first and only time",
    "  • Missing labels strictly masked — 7,175 valid observations out of 9,396 possible",
    "",
    "Phase 2 MolecularGNN Results:",
    "  • Macro ROC-AUC: 0.7395 ± 0.0182 (SEM), 95% CI [0.6995, 0.7795]",
    "  • Macro PR-AUC:  0.3118 ± 0.0331 (SEM), 95% CI [0.2389, 0.3847]",
    "  • All 12 endpoints evaluated successfully (0 excluded)",
    "",
    "Verification Guarantees:",
    "  • Zero data leakage — test labels never seen during training",
    "  • Deterministic — repeated inference yields identical predictions (Δ < 10⁻⁶)",
    "  • 783 unique molecule IDs verified — no duplicates, no missing",
    "  • SHA-256 checkpoint integrity verified before inference",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.8, items, size=14)

_notes(sl, "This is the most important results slide for Phase 2. Stress the quarantine protocol and leak-free evaluation.")


# ── Slide 14: Cross-Phase Comparison ─────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Cross-Phase Comparison: Classical ML vs Graph Neural Networks")

header3 = "Phase   Model                     Input                   Macro ROC-AUC        Macro PR-AUC"
rows3 = [
    "P1      Logistic Regression       Morgan FP (2048)        0.6865 ± 0.0210      0.2866 ± 0.0384",
    "P1      XGBoost                   Morgan FP + Desc        0.7456 ± 0.0175      0.3789 ± 0.0421",
    "P1      Random Forest (Champion)  Morgan FP + Desc        0.7488 ± 0.0185      0.3891 ± 0.0442",
    "P2      MolecularGNN (Edge-MPNN)  Raw Molecular Graphs    0.7395 ± 0.0182      0.3118 ± 0.0331",
]

tb = _tb(sl, 0.3, 1.5, 12.7, 0.4)
_set_para(tb.text_frame.paragraphs[0], header3, font_name="Consolas", size=12, bold=True, color=INK)

tb = _tb(sl, 0.3, 1.9, 12.7, 1.6)
tf = tb.text_frame
tf.word_wrap = True
for i, row in enumerate(rows3):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = row
    p.font.name = "Consolas"
    p.font.size = Pt(13)
    p.font.color.rgb = TEXT
    p.space_after = Pt(5)

items2 = [
    "Key Finding: Statistical Parity in ROC-AUC Discrimination",
    "  • RF ROC-AUC CI: [0.7125, 0.7851] vs GNN CI: [0.6995, 0.7795] → overlapping",
    "  • Difference (0.0093) is well within standard error (~0.018)",
    "  • GNN achieves this WITHOUT any manual feature engineering",
    "",
    "PR-AUC Gap:",
    "  • RF maintains advantage in PR-AUC (0.3891 vs 0.3118)",
    "  • Tree ensembles create sharp orthogonal decision boundaries in sparse fingerprint space",
    "  • GNN's continuous latent space produces smoother, less sharp probability gradients",
]
_bullet_list(sl, 0.6, 3.5, 12, 3.8, items2, size=14)

_notes(sl, "This is THE key slide. Emphasize statistical parity. The GNN matches the champion RF without feature engineering.")


# ── Slide 15: Endpoint-Level Analysis ────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Endpoint-Level Performance Analysis")

items = [
    "Endpoint Performance Variation:",
    "  • Best GNN endpoint: SR-MMP (highest prevalence ~15%) — ROC-AUC ~0.85",
    "  • Worst GNN endpoint: NR-AR (lowest prevalence ~1.5%) — ROC-AUC ~0.59",
    "  • Strong positive correlation between endpoint prevalence and both metrics",
    "",
    "Class Imbalance Impact:",
    "  • Endpoints with <3% positive prevalence show degraded PR-AUC (<0.15)",
    "  • ROC-AUC is more robust to class imbalance than PR-AUC",
    "  • This validates using ROC-AUC as primary ranking metric for model comparison",
    "",
    "Interesting Patterns:",
    "  • GNN competitive or better on endpoints where molecular topology matters most",
    "  • RF stronger on endpoints where specific functional group fingerprint bits are discriminative",
    "  • Both approaches complement each other — potential for future ensemble methods",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.8, items, size=14)

_notes(sl, "Discuss endpoint-level variation. Viva tip: Be ready to explain why some endpoints are harder than others.")


# ── Slide 16: Discussion & Limitations ───────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Discussion, Threats to Validity & Limitations")

items = [
    "Scientific Contributions:",
    "  • Rigorous leak-free cross-paradigm benchmark on identical scaffold split",
    "  • Demonstrated GNN statistical parity with classical baselines on Tox21",
    "  • Systematic ablation study validating each architectural choice",
    "  • Full reproducibility: deterministic seeds, verified checkpoints, documented pipeline",
    "",
    "Limitations & Threats to Validity:",
    "  • Single-dataset scope: Results specific to Tox21; cross-dataset transfer untested",
    "  • 2D topology only: No 3D conformational or stereoelectronic features",
    "  • Single scaffold split: Variance across different scaffold clusterings not explored",
    "  • Limited epochs (10): More training time may improve GNN convergence",
    "  • Small model: Larger GNNs (attention, graph transformers) may perform better",
    "",
    "Reproducibility Note:",
    "  • All code, configs, checkpoints, and results available in GitHub repository",
    "  • Seeds, library versions, and evaluation protocols fully documented",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.8, items, size=14)

_notes(sl, "Be honest about limitations. Reviewers and viva panels respect intellectual honesty more than inflated claims.")


# ── Slide 17: Conclusion & Future Work ───────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl)
_add_title_bar(sl, "Conclusion & Future Work")

items = [
    "Conclusions:",
    "  ✓ Established rigorous classical ML baseline (RF: ROC-AUC 0.7488)",
    "  ✓ Designed and ablated edge-aware MolecularGNN achieving ROC-AUC 0.7395",
    "  ✓ Demonstrated statistical parity between learned graph representations",
    "    and engineered fingerprint baselines under strict scaffold evaluation",
    "  ✓ Validated that GNNs can match classical models without manual feature engineering",
    "  ✓ All results on quarantined test set with zero data leakage",
    "",
    "Future Work:",
    "  • 3D conformational features — incorporate spatial geometry via RDKit Allinger/ETKDG",
    "  • Graph attention mechanisms — learn weighted neighbor importance",
    "  • Pre-training on larger molecular databases (ZINC, ChEMBL) → fine-tune on Tox21",
    "  • Ensemble of RF + GNN for complementary strengths",
    "  • Extend to additional ADMET endpoints (CYP inhibition, hERG, AMES)",
    "  • Interpretability: attention/gradient-based identification of toxic substructures",
]
_bullet_list(sl, 0.6, 1.4, 12, 5.8, items, size=14)

_notes(sl, "Summarize the three key contributions. Future work should sound realistic and grounded.")


# ── Slide 18: Q&A / Thank You ────────────────────────────────────────────────

sl = prs.slides.add_slide(prs.slide_layouts[6])
_add_bg(sl, INK)

tb = _tb(sl, 1, 2.0, 11.3, 1.5)
_set_para(tb.text_frame.paragraphs[0],
          "Thank You",
          font_name=HEAD_FONT, size=48, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

tb = _tb(sl, 1, 3.5, 11.3, 1.0)
_set_para(tb.text_frame.paragraphs[0],
          "Questions & Discussion",
          size=28, color=RGBColor(0xA0, 0xC0, 0xE0), align=PP_ALIGN.CENTER)

tb = _tb(sl, 1, 5.0, 11.3, 1.5)
tf = tb.text_frame
tf.word_wrap = True
lines = [
    "GitHub: github.com/shouryapratap132006/Molecular-Toxicity-Prediction",
    "Team: Amogha • Bulbul • Shourya • Kanishka • Aditya",
]
for i, line in enumerate(lines):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = line
    p.font.name = BODY_FONT
    p.font.size = Pt(14)
    p.font.color.rgb = MUTED
    p.alignment = PP_ALIGN.CENTER

_notes(sl, "Open the floor for questions. Be prepared for questions about: scaffold splitting, missing label handling, GNN vs RF tradeoffs, and future work.")


# ── Save ─────────────────────────────────────────────────────────────────────

OUT.parent.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT))
print(f"Saved {OUT} ({OUT.stat().st_size / 1024:.1f} KiB, {len(prs.slides)} slides)")
