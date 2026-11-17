"""Fig. 31 (upper image) — Geometry of rescaling to the ground. Due to the
imaging geometry of SAR, the misalignments of the ionospheric feature at the
ionospheric height exaggerated on SAR image (left side). Its effect on the
ellipse is shown on the right side. The rescaling is associated also with
additional rotation.

ATBD L1 SAR (atbd-l1-sar), section 4.10.5 (Step 3.f, equation 4.121). This is
the upper image of the two-image figure (image_1_section_4.10.5); the lower
image (fig. 31) is fig31.py. Redrawn from the original figure and the text of
the section: the satellite, at height h_sat, looks down towards the ground
through the ionosphere at height h_iono; the feature seen at the ionospheric
height (middle plane) is projected along the lines of sight onto the ground,
which scales it by r = h_sat / (h_sat - h_iono). The black oblique lines are the
azimuth direction at the satellite, ionosphere and ground levels, the blue
arrow is the range direction. This is a schematic perspective sketch: the
positions are qualitative, taken from the original figure, not measured.

To modify: change the positions in GEOMETRY (pixels of the original figure,
x to the right, y downwards), the labels in LABELS or the colours in COLOURS,
then run ``python3 img1-4.10.5.py`` (matplotlib needed): it rewrites
img1-4.10.5.svg (web page) and img1-4.10.5.pdf (PDF export) next to this
file. Commit the three files together. Keep a dark text colour: the page puts
a white background behind the figure in both site themes.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# -- colours and sizes -------------------------------------------------------
COLOURS = {"text": "#000000", "black": "#000000", "blue": "#4472c4",
           "green": "#00b050", "red": "#ff0000"}
WIDTH_IN = 6.2  # PDF text width
FONT_PT = 11.5  # the PDF export prints it at about 65 %: subscripts stay >= 5 pt

# -- geometry (pixels of the original figure, y downwards) -------------------
X_MIN, X_MAX, Y_MIN, Y_MAX = -10, 725, -5, 745  # canvas, with room for the larger text
# black lines along the azimuth direction (satellite, ionosphere, ground, ground
# at the foot of the lines of sight); an arrow marks the end given in ARROWED
SAT_LINE = ((167, 30), (609, 205))
IONO_LINE = ((153, 192), (527, 340))
GROUND_LINE = ((112, 396), (609, 594))
SIDE_LINE = ((13, 543), (403, 695))  # lower line, parallel, with an arrowhead
ARROWED = [SAT_LINE[0], SIDE_LINE[0]]
# three satellite positions, their vertical (nadir) feet on the ground line and
# their lines of sight to the lower line
SAT_POINTS = [(262, 68), (368, 110), (472, 152)]
FEET = [(263, 457), (368, 498), (473, 540)]
LOS_ENDS = [(96, 573), (202, 616), (306, 657)]
# blue range-direction lines on the ground (ends); the middle one has the
# arrowhead (Range), the last one leaves the figure at the bottom
RANGE_LINES = [((96, 573), (263, 457)), ((68, 708), (368, 498)), ((190, 740), (473, 540))]
RANGE_ARROW_TIP = (68, 708)
# green lines (satellite point -> ground) and the red / dashed red segments
GREEN_LINES = [((262, 68), (171, 521)), ((472, 152), (233, 708))]
RED_GROUND = ((171, 521), (233, 708))
RED_DASHED = ((126, 552), (275, 676))
# the feature at the ionospheric height: blue lines and the red diagonal
IONO_BLUE = [((262, 236), (195, 283)), ((195, 283), (402, 366)), ((297, 325), (368, 275)),
             ((402, 366), (472, 318)), ((402, 366), (372, 386))]
IONO_RED = ((222, 263), (372, 386))
# height arrows, drawn from the ground line upwards: (x, from y, to y)
H_SAT = (609, 594, 205)
H_IONO = (492, 546, 326)

# -- labels (mathtext), word for word from the original figure ----------------
# text, position, alignment, rotation (degrees, counterclockwise on screen)
LABELS = {
    "azimuth": ("Azimuth", (255, 45), "center", -21.7),
    "range": ("Range", (122, 644), "center", 36),
    "h_iono": (r"$h_{iono}$", (505, 420), "left", 0),
    "h_sat": (r"$h_{sat}$", (622, 390), "left", 0),
}

# -- drawing -------------------------------------------------------------------
plt.rcParams.update({"mathtext.fontset": "dejavusans", "font.size": FONT_PT})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * (Y_MAX - Y_MIN) / (X_MAX - X_MIN)))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MAX, Y_MIN)
ax.axis("off")


def line(p0, p1, colour, lw=0.9, **kw):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=COLOURS[colour], lw=lw,
            solid_capstyle="butt", **kw)


def arrow(p0, p1, colour, lw=0.9, head=9):
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(
        arrowstyle="-|>", color=COLOURS[colour], lw=lw, shrinkA=0, shrinkB=0,
        mutation_scale=head))


# blue: lines of sight, verticals, range lines on the ground, feature at the iono height
for sat, foot, end in zip(SAT_POINTS, FEET, LOS_ENDS, strict=True):
    line(sat, end, "blue")
    line(sat, foot, "blue")
for p0, p1 in RANGE_LINES:
    line(p0, p1, "blue")
for p0, p1 in IONO_BLUE:
    line(p0, p1, "blue")
for p0, p1 in GREEN_LINES:
    line(p0, p1, "green")
line(*IONO_RED, "red")
line(*RED_GROUND, "red")
line(*RED_DASHED, "red", dashes=(3, 2))
# black lines along the azimuth direction
for p0, p1 in (SAT_LINE, IONO_LINE, GROUND_LINE, SIDE_LINE):
    line(p0, p1, "black")
for (p0, p1) in (SAT_LINE, SIDE_LINE):
    arrow(p1, p0, "black", head=10)  # same line, arrowhead at its first point
arrow(RANGE_LINES[1][1], RANGE_ARROW_TIP, "blue", head=10)
# heights
for x, y0, y1 in (H_SAT, H_IONO):
    arrow((x, y0), (x, y1), "black", head=9)

for s, (x, y), ha, rot in LABELS.values():
    ax.text(x, y, s, color=COLOURS["text"], ha=ha, va="center", rotation=rot,
            rotation_mode="anchor")

# -- output (same bytes for the same script: fixed ids and dates) -------------
if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "esa-doc2open", "svg.fonttype": "path",
                         "pdf.fonttype": 42})
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.svg", format="svg", transparent=True, metadata={"Date": None})
    fig.savefig(f"{stem}.pdf", format="pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
