#!/usr/bin/env python
"""
Figure 1: the FunnelPlan benchmark as layered bands (read bottom to top), using the FunnelPlan palette
(figures_src/palette.py). Writes results_full/figures_final/fig_framework.{drawio,svg,pdf,png}.
Usage: python figures_src/fig_framework.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scene import Scene, text_width  # noqa: E402
from palette import (AMBER, CARD_LINE, INK, MUTED, SAND, SAND_LINE, TEAL, TEAL_LINE,  # noqa: E402
                     TEAL_TINT)

OUT = Path(__file__).parent.parent / "results_full" / "figures_final" / "fig_framework"
BAND_F, BAND_S, CARD_S = SAND, SAND_LINE, CARD_LINE   # layers and cards: warm neutral
LABEL, TITLE = TEAL, TEAL                              # layer labels and card titles: teal
CHIP_F, CHIP_S = TEAL_TINT, TEAL_LINE                  # protocol chips: teal tint
W, H = 1400, 772
LX, LW = 14, 214                 # label column
X0, X1 = 238, 1388               # card area
GAP = 12
FS_LABEL, FS_TITLE, FS_ITEM, LH = 28, 22, 19, 1.3


def robot(sc, x, y, s=20):
    """small LLM icon (amber, the same colour as decisions): head, eyes, antenna."""
    sc.line([(x + s / 2, y + 1), (x + s / 2, y + 5)], color=AMBER, sw=1.6, arrow=False)
    sc.box(x + s / 2 - 2, y - 2, 4, 4, fill=AMBER, stroke=AMBER, sw=1, kind="ellipse")
    sc.box(x, y + 5, s, s * 0.72, fill="#ffffff", stroke=AMBER, sw=1.6, r=4)
    for dx in (0.3, 0.7):
        sc.box(x + s * dx - 2, y + 5 + s * 0.3, 4, 4, fill=AMBER, stroke=AMBER, sw=1, kind="ellipse")


def band(sc, y, h, label, sub=None):
    sc.box(LX - 6, y, W - 2 * (LX - 6), h, fill=BAND_F, stroke=BAND_S, sw=1.5, r=14)
    lines = [(label, FS_LABEL, True, LABEL)]
    if sub:
        lines += [(s, 17, False, MUTED) for s in sub]
    sc.text(LX, y, LW, h, lines, align="left", pad=8, lh=1.18)


def card(sc, x, y, w, h, title, items, cols=1, tag=None, robots=(), item_size=FS_ITEM):
    sc.box(x, y, w, h, fill="#ffffff", stroke=CARD_S, sw=1.3, r=10, shadow=True)
    sc.text(x, y + 4, w, FS_TITLE * 1.4, [(title, FS_TITLE, True, TITLE)], align="left", valign="top", pad=12)
    if tag:
        tw = text_width(tag, 16, True) + 18
        sc.box(x + w - tw - 8, y + 9, tw, 24, fill=TEAL, stroke=TEAL, sw=1, r=12)
        sc.text(x + w - tw - 8, y + 9, tw, 24, [(tag, 16, True, "#ffffff")], pad=2)
    per = -(-len(items) // cols)
    cw = (w - 24) / cols
    ty = y + 8 + FS_TITLE * 1.45
    for c in range(cols):
        col = items[c * per:(c + 1) * per]
        if not col:
            continue
        lines = [(("    " + s[1:]) if s.startswith("~") else ("• " + s), item_size, False, INK) for s in col]
        sc.text(x + 12 + c * cw, ty, cw, len(col) * item_size * LH + 4, lines, align="left", valign="top", pad=0)
        for k, s in enumerate(col):
            if s in robots:
                rx = x + 12 + c * cw + text_width("• " + s, item_size) + 8
                robot(sc, rx, ty + k * item_size * LH + 1, 18)


def chips(sc, y, h, labels):
    n = len(labels)
    cw = (X1 - X0 - (n - 1) * GAP) / n
    for k, s in enumerate(labels):
        x = X0 + k * (cw + GAP)
        sc.box(x, y, cw, h, fill=CHIP_F, stroke=CHIP_S, sw=1.2, r=6)
        sc.text(x, y, cw, h, [(t, FS_ITEM, False, INK) for t in s.split("\n")], pad=8, lh=1.2)


def row(widths):
    """x positions for cards of the given widths, filling the card area with equal gaps."""
    free = (X1 - X0) - sum(widths)
    g = free / (len(widths) - 1) if len(widths) > 1 else 0
    xs, x = [], X0
    for w in widths:
        xs.append(x)
        x += w + g
    return xs


def main():
    sc = Scene(W, H)
    y = 8
    # ---- analyses (top)
    h = 160
    band(sc, y, h, "Analyses")
    ws = [218] * 5
    xs = row(ws)
    for x, w, (t, it) in zip(xs, ws, [
            ("H1: policy vs hull", ["27 primary tests", "30 secondary tests", "10 stronger policies"]),
            ("Robustness", ["3 grading rules", "5 cost axes", "single plan baseline", "margin test at 0.03"]),
            ("H3: multi step", ["IRCoT vs the hull", "3 readers x 3 sets"]),
            ("Mechanism", ["oracle headroom", "what planners learn"]),
            ("Weak protocol", ["Always RAG baseline", "one seed, strict EM"])]):
        card(sc, x, y + 8, w, h - 16, t, it)
    y += h + GAP
    # ---- protocol
    h = 70
    band(sc, y, h, "Protocol")
    chips(sc, y + 9, h - 18, ["hull fitted on\ntraining folds", "5 folds x 5 seeds", "t test, bootstrap CI",
                              "Holm per family", "must pass in\nboth plan sets"])
    y += h + GAP
    # ---- policies
    h = 132
    band(sc, y, h, "Policies", ["chosen per", "question"])
    ws = [404, 352, 206, 164]
    xs = row(ws)
    card(sc, xs[0], y + 8, ws[0], h - 16, "Primary", ["TSR planner", "Adaptive RAG", "Adaptive k", "Closed book gate",
                                                      "Confidence cascade", "Reader cascades"], cols=2)
    card(sc, xs[1], y + 8, ws[1], h - 16, "Secondary", ["Adaptive k, rule", "AR, BGE M3", "AR, FLAN T5",
                                                        "Per plan LR", "Per plan kNN", "METIS port"], cols=2,
         robots={"METIS port"})
    card(sc, xs[2], y + 8, ws[2], h - 16, "Stronger", ["Classifier gate", "Planner with", "~confidence"])
    card(sc, xs[3], y + 8, ws[3], h - 16, "Reference", ["IRCoT", "~multi step"], robots={"IRCoT"})
    y += h + GAP
    # ---- measurement
    h = 104
    band(sc, y, h, "Measurement")
    ws = [560, 566]
    xs = row(ws)
    card(sc, xs[0], y + 8, ws[0], h - 16, "Grade every answer",
         ["label_final: strict rule, Qwen3 14B judge if ambiguous",
          "human audit: 2 x 200 answers, 3 independent raters"])
    card(sc, xs[1], y + 8, ws[1], h - 16, "Cost of every answer",
         ["mean measured latency per question (main axis)", "tokens, output priced 4x, FLOPs, calls"])
    y += h + GAP
    # ---- plan matrix
    h = 150
    band(sc, y, h, "Plan matrix", ["every question", "x every plan", "x every reader"])
    ws = [452, 228, 434]
    xs = row(ws)
    card(sc, xs[0], y + 8, ws[0], h - 16, "6 retrievers x depth 0 to 20",
         ["BM25", "SPLADE v3", "BGE M3", "Stella 1.5B", "BM25 + BGE M3 (RRF)", "Qwen3 Reranker 0.6B"], cols=2)
    card(sc, xs[1], y + 8, ws[1], h - 16, "Two hop plans", ["BM25, 3 or 5 per hop", "reader writes the", "~second query"],
         tag=None)
    card(sc, xs[2], y + 8, ws[2], h - 16, "5 readers, greedy",
         ["Qwen2.5 7B, bf16", "Llama 3.1 8B, bf16", "Qwen3 14B, int8", "Gemma 3 27B, 4 bit", "Qwen3 32B, 4 bit"],
         cols=2)
    y += h + GAP
    # ---- data (bottom)
    h = 104
    band(sc, y, h, "Data")
    ws = [600, 526]
    xs = row(ws)
    card(sc, xs[0], y + 8, ws[0], h - 16, "4 QA datasets, 500 questions each",
         ["Natural Questions, HotpotQA (one and two hop)", "2WikiMultihopQA, MuSiQue (multi hop)"])
    card(sc, xs[1], y + 8, ws[1], h - 16, "Corpora",
         ["BEIR Wikipedia for NQ (2.7M) and HotpotQA (5.2M)", "pooled paragraphs for 2Wiki and MuSiQue"])
    y += h + 8
    sc.h = y
    print(f"canvas {W} x {y}")
    sc.save(OUT)
    print(f"written {OUT}.drawio/.svg/.pdf/.png")


if __name__ == "__main__":
    main()
