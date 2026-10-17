"""
FunnelPlan figure palette, shared by every diagram.

Two accents only, each with one job:
  * teal  : structure (layer labels, card titles, start and end of a flow, the baseline)
  * amber : the point where a policy decides, and the LLM marker
Backgrounds are tints of the same two hues; everything else is ink or slate. Teal and amber sit on the blue and yellow ends of the colour axis
that the common colour vision deficiencies keep, and they differ in lightness, so they stay apart in grayscale.
Text colours pass WCAG AA (contrast 4.5 or more) on white; check with `python figures_src/palette.py`.
"""

INK = "#22262E"         # body text
MUTED = "#5B6170"       # captions, branch labels, footers
TEAL = "#0F5E63"        # titles, layer labels, terminals
TEAL_TINT = "#E4F0EF"   # chips, baseline column
TEAL_LINE = "#9CC4C1"
AMBER = "#965600"       # decisions, LLM marker
AMBER_TINT = "#FBF0DE"
AMBER_LINE = "#D9B77E"
SAND = "#F5F2EC"        # neutral fill (kept for small elements)
SAND_LINE = "#DCD4C6"
# Figure 1 layers: one teal ramp, lightest at the bottom (Data) and deepest at the top (Analyses),
# so the colour follows the reading order. Teal text keeps contrast 4.8 or more on every step.
LAYER_RAMP = ["#E6F4F2", "#DAEFEC", "#CEE9E5", "#C2E3DE", "#B6DDD8", "#AAD7D1"]
LAYER_LINE = "#8FC3BC"
# Figure 2 columns: warm gold for the baseline every policy is compared with, light teal for the policies,
# cool grey with a dashed amber border for the reference.
BASE_TINT, BASE_LINE = "#F9E6BF", "#DDB46A"
POLICY_TINT, POLICY_LINE = "#DDF0EC", "#8FC3BC"
REF_TINT, REF_LINE = "#EEF0F3", "#D9B77E"
CARD_LINE = "#CFC7B9"
STEP_LINE = "#3A3F4A"   # outline of process steps
ARROW = "#3F4756"       # flow arrows (slate)


# Data figures (scripts/23_figures.py): the same two hues, plus neutrals and a five colour set for readers or policies.
DATA_TEAL = TEAL               # main series: single step plans, our protocol, IRCoT, audit 1
DATA_AMBER = "#C27C0E"         # second series: with two hop plans, weak protocol, audit 2 (brighter than AMBER for marks)
DATA_GREY = "#B8BCC2"          # fixed plans and other context marks
DATA_SLATE = ARROW             # hull lines, reference lines
TEAL_RAMP = ["#A9D3CF", "#4E9C97", "#0F5E63"]   # light to dark: 2 plans, 3 plans, all plans
CATEGORICAL = ["#0F5E63", "#C27C0E", "#5D5BA6", "#B4495E", "#6F8A2B"]   # teal, amber, indigo, rose, olive


def _lum(hexc):
    c = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b="#FFFFFF"):
    la, lb = sorted((_lum(a), _lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


if __name__ == "__main__":
    for name in ("INK", "MUTED", "TEAL", "AMBER", "ARROW"):
        c = globals()[name]
        print(f"{name:6s} {c}  on white {contrast(c):4.1f}  on sand {contrast(c, SAND):4.1f}")
    print(f"white on TEAL {contrast('#FFFFFF', TEAL):4.1f}   AMBER on AMBER_TINT {contrast(AMBER, AMBER_TINT):4.1f}")
    for c in LAYER_RAMP + [BASE_TINT, POLICY_TINT, REF_TINT]:
        print(f"{c}: TEAL {contrast(TEAL, c):4.1f}  MUTED {contrast(MUTED, c):4.1f}  INK {contrast(INK, c):4.1f}")
