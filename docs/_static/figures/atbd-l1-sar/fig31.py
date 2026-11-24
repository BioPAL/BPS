"""Fig. 31 — Geometry of rescaling to the ground. Due to the imaging geometry of
SAR, the misalignments of the ionospheric feature at the ionospheric height
exaggerated on SAR image (left side). Its effect on the ellipse is shown on the
right side. The rescaling is associated also with additional rotation.

ATBD L1 SAR (atbd-l1-sar), section 4.10.5 (Step 3.f, equation 4.121). This is
the lower image (right side) of the two-image figure; the upper image (left
side, 3D geometry) is img1-4.10.5.py. Redrawn from the original figure and the
text of the section: the ellipse of fig. 30 (grey) and the same ellipse
rescaled to the ground (black), stretched and rotated; the range (green,
along y) and azimuth (green) directions and the two parallel black lines that
bound the rescaled ellipse along the range direction. The red arrows are the
principal semi-axes of the two ellipses (dashed for the grey one). This is a
schematic: the axis lengths and the angles are qualitative, taken from the
original figure, not measured.

To modify: change the numbers in GEOMETRY (pixels of the original figure,
x to the right, y downwards), the labels in LABELS or the colours in COLOURS,
then run ``python3 fig31.py`` (matplotlib needed): it rewrites
fig31.svg (web page) and fig31.pdf (PDF export) next to this file. Commit the
three files together. Keep a dark text colour: the page puts a white
background behind the figure in both site themes.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

# -- colours and sizes -------------------------------------------------------
COLOURS = {"text": "#000000", "black": "#000000", "light": "#8faadc",
           "grey": "#7f7f7f", "green": "#00b050", "red": "#ff0000"}
WIDTH_IN = 6.2  # PDF text width
FONT_PT = 11.5  # the PDF export prints it at about 65 %: subscripts stay >= 5 pt

# -- geometry (pixels of the original figure, y downwards) -------------------
X_MIN, X_MAX, Y_MIN, Y_MAX = -15, 725, -15, 705  # canvas, with room for the larger text
ORIGIN = (329, 363)  # origin of the B-aligned axes
X_AXIS = (0, 655)  # x extent of the x axis (arrow on the right)
Y_AXIS = (692, 35)  # y extent of the y axis (arrow at the top)
RANGE_TIP = (311, 35)  # green arrow, range direction (LOS)
AZIMUTH_FROM, AZIMUTH_TIP = (0, 380), (655, 347)  # green arrow, azimuth direction
NORTH_TIP, EAST_TIP = (506, 88), (605, 540)
GREY = {"a": 220, "b": 27, "angle": 41.9}  # ellipse of fig. 30 (semi-axes, tilt)
BLACK = {"a": 314, "b": 26.5, "angle": 57.5}  # the same ellipse rescaled to the ground
# the two black lines are parallel to the range direction and touch the tips
# of the black ellipse; each is drawn between two heights
LINE_Y = {"left": (35, 692), "right": (10, 670)}

# -- labels (mathtext), word for word from the original figure ----------------
LABELS = {
    "x": (r"$x$", (676, 369), "center", 0),
    "y": (r"$y$", (322, 18), "center", 0),
    "north": ("North", (523, 62), "center", 0),
    "east": ("East", (640, 547), "center", 0),
    "azimuth": ("Azimuth", (598, 321), "center", 0),
    "range": ("Range", (291, 95), "center", 90),
}

# -- drawing -------------------------------------------------------------------
plt.rcParams.update({"mathtext.fontset": "dejavusans", "font.size": FONT_PT})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * (Y_MAX - Y_MIN) / (X_MAX - X_MIN)))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MAX, Y_MIN)
ax.axis("off")


def arrow(p0, p1, colour, lw=1.0, head=9):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(
        arrowstyle="-|>", color=COLOURS[colour], lw=lw, shrinkA=0, shrinkB=0,
        mutation_scale=head))


def polar(r, deg):
    """Point at distance r from ORIGIN, deg degrees below the +x axis (y down)."""
    a = math.radians(deg)
    return (ORIGIN[0] + r * math.cos(a), ORIGIN[1] + r * math.sin(a))


ax.add_patch(Ellipse(ORIGIN, 2 * GREY["a"], 2 * GREY["b"], angle=GREY["angle"], fill=False,
                     edgecolor=COLOURS["grey"], lw=1.4))
ax.add_patch(Ellipse(ORIGIN, 2 * BLACK["a"], 2 * BLACK["b"], angle=BLACK["angle"], fill=False,
                     edgecolor=COLOURS["black"], lw=1.4))
# the two black lines, parallel to the range direction, through the tips of the black ellipse
slope = (RANGE_TIP[0] - ORIGIN[0]) / (RANGE_TIP[1] - ORIGIN[1])  # dx / dy of the range direction
for side, sign in (("left", -1), ("right", 1)):
    tip = polar(sign * BLACK["a"], BLACK["angle"])
    y0, y1 = LINE_Y[side]
    ax.plot([tip[0] + slope * (y0 - tip[1]), tip[0] + slope * (y1 - tip[1])], [y0, y1],
            color=COLOURS["black"], lw=1.0)
# axes x, y (B-aligned), green range and azimuth arrows
ax.plot([X_AXIS[0], ORIGIN[0]], [ORIGIN[1], ORIGIN[1]], color=COLOURS["black"], lw=1.0)
arrow(ORIGIN, (X_AXIS[1], ORIGIN[1]), "black")
ax.plot([ORIGIN[0], ORIGIN[0]], [Y_AXIS[0], ORIGIN[1]], color=COLOURS["black"], lw=1.0)
arrow(ORIGIN, (ORIGIN[0], Y_AXIS[1]), "black")
arrow(ORIGIN, RANGE_TIP, "green", lw=0.9)
ax.plot([AZIMUTH_FROM[0], ORIGIN[0]], [AZIMUTH_FROM[1], ORIGIN[1]], color=COLOURS["green"], lw=0.9)
arrow(ORIGIN, AZIMUTH_TIP, "green", lw=0.9)
# east and north, for information
arrow(ORIGIN, NORTH_TIP, "light", lw=1.0, head=11)
arrow(ORIGIN, EAST_TIP, "light", lw=1.0, head=11)
# principal semi-axes: grey ellipse dashed, black ellipse solid
for e, dashed in ((GREY, True), (BLACK, False)):
    for r, d in ((e["a"], e["angle"]), (e["b"], e["angle"] - 90)):
        p = polar(r, d)
        ax.annotate("", xy=p, xytext=ORIGIN, arrowprops=dict(
            arrowstyle="-|>", color=COLOURS["red"], lw=1.0, shrinkA=0, shrinkB=0,
            mutation_scale=9, linestyle=(0, (3, 2)) if dashed else "-"))

for s, (x, y), ha, rot in LABELS.values():
    ax.text(x, y, s, color=COLOURS["text"], ha=ha, va="center", rotation=rot)

# -- output (same bytes for the same script: fixed ids and dates) -------------
if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "esa-doc2open", "svg.fonttype": "path",
                         "pdf.fonttype": 42})
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.svg", format="svg", transparent=True, metadata={"Date": None})
    fig.savefig(f"{stem}.pdf", format="pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
