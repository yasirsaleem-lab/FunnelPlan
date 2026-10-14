"""
Figure 2: the baseline and the policy families as small flow charts with one visual grammar, in the
FunnelPlan palette (figures_src/palette.py): teal filled start and end, outlined steps, amber decisions, slate
arrows, dashed paths not taken. Column colour gives the role: warm gold for the baseline, light teal for the
policies, cool grey with a dashed border for the reference. Writes results_full/figures_final/fig_policies.{drawio,svg,pdf,png}.
Usage: python figures_src/fig_policies.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scene import SYMBOL_FONT, Scene, wrap  # noqa: E402
from palette import (AMBER, ARROW, BASE_LINE, BASE_TINT, INK, MUTED, POLICY_LINE, POLICY_TINT,  # noqa: E402
                     REF_LINE, REF_TINT, STEP_LINE, TEAL)

OUT = Path(__file__).parent.parent / "results_full" / "figures_final" / "fig_policies"
TITLE = TEAL
BASE, POLICY, REF = (BASE_TINT, BASE_LINE, False), (POLICY_TINT, POLICY_LINE, False), (REF_TINT, REF_LINE, True)
ROLES = [BASE, POLICY, POLICY, POLICY, POLICY, REF]     # (fill, border, dashed border) per column
W, COLW, GAP, X0 = 1400, 218, 12.4, 8
FS, FS_T = 19, 22
Y_TITLE, Y_Q, Y_B1, Y_D, Y_B2, Y_R, Y_A, Y_FOOT, H = 8, 136, 206, 270, 370, 440, 512, 572, 684


def term(sc, cx, y, s):
    sc.box(cx - 70, y, 140, 40, fill=TEAL, stroke=TEAL, sw=2.2, r=20)
    sc.text(cx - 70, y, 140, 40, [(s, FS, True, "#ffffff")], pad=4)


def step(sc, cx, y, s, w=176, h=48):
    sc.box(cx - w / 2, y, w, h, fill="#ffffff", stroke=STEP_LINE, sw=1.6, r=7, shadow=True)
    lines = wrap(s, FS, w - 12)
    sc.text(cx - w / 2, y, w, h, [(t, FS, False, INK) for t in lines], pad=5, lh=1.15)


def decision(sc, cx, y, s, w=176, h=74):
    sc.box(cx - w / 2, y, w, h, fill="#ffffff", stroke=AMBER, sw=2.2, kind="diamond")
    lines = wrap(s, FS, w * 0.62)
    sc.text(cx - w * 0.36, y, w * 0.72, h, [(t, FS, True, AMBER) for t in lines], pad=2, lh=1.1)


def arrow(sc, pts, dash=False):
    sc.line(pts, color=ARROW, sw=2.2, dash=dash)


def label(sc, x, y, s, align="center", w=60):
    sc.text(x - w / 2 if align == "center" else x, y, w, 24, [(s, 18, True, MUTED)], align=align, pad=1)


def column(sc, k, title, desc, foot):
    x = X0 + k * (COLW + GAP)
    fill, stroke, dashed = ROLES[k]
    sc.box(x, Y_TITLE + 44, COLW, H - Y_TITLE - 52, fill=fill, stroke=stroke, sw=1.5, r=16, dash=dashed)
    lines = [(t, FS_T, True, TITLE) for t in wrap(title, FS_T, COLW - 34, bold=True)] + \
        [(t, FS, False, INK) for t in wrap(desc, FS, COLW - 34)]
    sc.box(x + 10, Y_TITLE, COLW - 20, 112, fill="#ffffff", stroke=stroke, sw=1.5, r=8, dash=dashed)
    sc.text(x + 10, Y_TITLE, COLW - 20, 112, lines, pad=6, lh=1.18)
    fl = [(t, 18, False, MUTED) for t in wrap(foot, 18, COLW - 24)]
    sc.text(x + 6, Y_FOOT, COLW - 12, H - Y_FOOT - 12, fl, valign="top", pad=4, lh=1.18)
    return x + COLW / 2


def gate_column(sc, k, title, desc, first, second, foot):
    cx = column(sc, k, title, desc, foot)
    term(sc, cx, Y_Q, "Question")
    arrow(sc, [(cx, Y_Q + 40), (cx, Y_B1)])
    step(sc, cx, Y_B1, first)
    arrow(sc, [(cx, Y_B1 + 48), (cx, Y_D + 8)])
    decision(sc, cx, Y_D + 8, "confident?")
    arrow(sc, [(cx, Y_D + 82), (cx, Y_B2 + 10)])
    label(sc, cx - 22, Y_D + 84, "no", w=40)
    step(sc, cx, Y_B2 + 10, second)
    arrow(sc, [(cx, Y_B2 + 58), (cx, Y_A)])
    # yes path: right vertex, down the side, into the answer box
    xr = cx + 98
    arrow(sc, [(cx + 88, Y_D + 45), (xr, Y_D + 45), (xr, Y_A + 20), (cx + 70, Y_A + 20)])
    label(sc, cx + 96, Y_D + 14, "yes", w=40)
    term(sc, cx, Y_A, "Answer")


def main():
    sc = Scene(W, H)
    # 1. baseline: the fixed plan hull, a coin flip between two fixed plans
    cx = column(sc, 0, "Fixed plan hull", "baseline: random mix of two fixed plans",
                "fitted on training folds; no question features")
    term(sc, cx, Y_Q, "Question")
    arrow(sc, [(cx, Y_Q + 40), (cx, Y_D - 40)])
    decision(sc, cx, Y_D - 40, "coin flip", h=70)
    for side, name, lab in ((-1, "Plan A", "p"), (1, "Plan B", "q")):
        bx = cx + side * 52
        y_edge = Y_D - 5 + 35 * (1 - 52 / 88)          # point on the lower diamond edge above the plan box
        arrow(sc, [(bx, y_edge), (bx, Y_B2 - 30)])
        label(sc, bx + side * 16, y_edge + 8, lab, w=24)
        step(sc, bx, Y_B2 - 30, name, w=96)
        arrow(sc, [(bx, Y_B2 + 18), (bx, Y_R - 18), (cx, Y_R - 18), (cx, Y_R - 4)])
    step(sc, cx, Y_R - 4, "Reader")
    arrow(sc, [(cx, Y_R + 44), (cx, Y_A)])
    term(sc, cx, Y_A, "Answer")

    # 2 to 3, 5: gates and cascades
    gate_column(sc, 1, "Closed book gate", "retrieve only when the reader is unsure", "Reader, no passages",
                "Retrieval plan, then reader", "Closed book gate (primary); classifier gate (stronger)")
    gate_column(sc, 2, "Confidence cascade", "stronger plan when unsure", "Cheap plan, then reader",
                "Strong plan, then reader", "Confidence cascade (primary)")

    # 4. router or planner
    cx = column(sc, 3, "Router or planner", "predict one plan for each question",
                "TSR planner, Adaptive RAG, Adaptive k, per plan LR and kNN, METIS port")
    term(sc, cx, Y_Q, "Question")
    arrow(sc, [(cx, Y_Q + 40), (cx, Y_D - 44)])
    decision(sc, cx, Y_D - 44, "pick a plan", h=74)
    for j, (dx, name) in enumerate(((-72, "Plan 1"), (0, "Plan 2"), (72, "Plan k"))):
        chosen = j == 1
        arrow(sc, [(cx, Y_D + 30), (cx, Y_D + 46), (cx + dx, Y_D + 46), (cx + dx, Y_B2 - 26)], dash=not chosen)
        step(sc, cx + dx, Y_B2 - 26, name, w=68, h=46)
        arrow(sc, [(cx + dx, Y_B2 + 20), (cx + dx, Y_R - 18), (cx, Y_R - 18), (cx, Y_R - 4)], dash=not chosen)
    step(sc, cx, Y_R - 4, "Reader")
    arrow(sc, [(cx, Y_R + 44), (cx, Y_A)])
    term(sc, cx, Y_A, "Answer")

    gate_column(sc, 4, "Reader cascade", "move to a larger reader when unsure", "Plan x, small reader",
                "Plan y, large reader", "7B to Llama 3.1 8B; 7B to Qwen3 14B (primary)")

    # 6. IRCoT, the multi step reference
    cx = column(sc, 5, "IRCoT", "reference: retrieve and reason in turns",
                "compared at its own measured cost (H3)")
    term(sc, cx, Y_Q, "Question")
    arrow(sc, [(cx, Y_Q + 40), (cx, Y_B1)])
    step(sc, cx + 14, Y_B1, "Retrieve (BM25)", w=156)
    arrow(sc, [(cx + 14, Y_B1 + 48), (cx + 14, Y_D + 2)])
    step(sc, cx + 14, Y_D + 2, "Reason one step", w=156)
    arrow(sc, [(cx + 14, Y_D + 50), (cx + 14, Y_B2 - 4)])
    decision(sc, cx + 14, Y_B2 - 4, "answer?", w=150, h=70)
    arrow(sc, [(cx + 14, Y_B2 + 66), (cx + 14, Y_A)])
    label(sc, cx + 38, Y_B2 + 70, "yes", w=40)
    term(sc, cx + 14, Y_A, "Answer")
    xl = cx - 88
    arrow(sc, [(cx - 61, Y_B2 + 31), (xl, Y_B2 + 31), (xl, Y_B1 + 24), (cx - 64, Y_B1 + 24)], dash=True)
    label(sc, xl + 2, Y_B2 + 34, "no", align="left", w=40)
    sc.box(xl - 15, Y_D + 8, 30, 34, fill=REF_TINT, stroke=REF_TINT, sw=0.5, r=6)
    sc.text(xl - 15, Y_D + 6, 30, 38, [("↻", 28, True, ARROW)], pad=0, family=SYMBOL_FONT)

    sc.save(OUT)
    print(f"written {OUT}.drawio/.svg/.pdf/.png ({W} x {H})")


if __name__ == "__main__":
    main()
