"""
FunnelPlan figure palette, shared by every diagram.

Two accents only, each with one job:
  * teal  : structure (layer labels, card titles, start and end of a flow, the baseline)
  * amber : the point where a policy decides, and the LLM marker
Everything else is warm neutral or ink. Teal and amber sit on the blue and yellow ends of the colour axis
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
SAND = "#F5F2EC"        # layer and column fill
SAND_LINE = "#DCD4C6"
CARD_LINE = "#CFC7B9"
STEP_LINE = "#3A3F4A"   # outline of process steps
ARROW = "#3F4756"       # flow arrows (slate)


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
