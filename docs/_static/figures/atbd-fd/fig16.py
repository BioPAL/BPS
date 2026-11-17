"""Fig. 16 — Time of validity for each acquisition among two global cycles,
with overlaps shown in RGB colours. The FD relative to various acquisition
times within the global cycle are reported. See also Tab.1.

ATBD L2 FD (atbd-l2-fd), section 4.1.4. Redrawn from the original figure: the
bars FD t1, FD t2 and FD t3 mark the acquisition times of each global cycle
("n-1" and "n"); the rectangles between them are the validity intervals, whose
colour (red, yellow, white, cyan, blue) encodes which of the three FD maps (R,
G, B in the additive-colour disks) overlap there. This is a schematic: the
horizontal positions are qualitative.

To modify: change the positions in GEOMETRY (units of the original figure,
x to the right, y downwards), the labels in LABELS or the colours in COLOURS,
then run ``python3 fig16.py`` (matplotlib needed): it rewrites
fig16.svg (web page) and fig16.pdf (PDF export) next to this file. Commit the
three files together. Keep a dark text colour: the page puts a white
background behind the figure in both site themes.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

# -- colours and sizes -------------------------------------------------------
COLOURS = {
    "text": "#000000", "line": "#000000",
    "red": "#ff0000", "green": "#00b050", "blue": "#0000ff",
    "yellow": "#ffff00", "cyan": "#00ffff", "white": "#ffffff",
    "magenta": "#ff00ff", "disk_g": "#00ff00",
}
WIDTH_IN = 6.2  # PDF text width
FONT_PT = 11.5  # the PDF export prints it at about 65 %

# -- geometry (pixels of the original figure + margin, y downwards) -----------
W, H = 1180, 520
Y_AXIS = 430             # time axis
Y_BAR = (315, 425)       # acquisition bars (top, bottom)
Y_BOX = (362, 420)       # validity rectangles
Y_SEP = (155, 490)       # black cycle separator
X_SEP = 500
BAR_W = 12
BARS = {"t1a": 68, "t2a": 242, "t3a": 393, "t1b": 593, "t2b": 767, "t3b": 918}
BAR_COLOUR = {"t1": "red", "t2": "green", "t3": "blue"}
# validity rectangles: (x0, x1, colour)
BOXES = [(74, 236, "red"), (248, 386, "yellow"), (400, 586, "white"),
         (598, 762, "cyan"), (773, 912, "blue")]
DISKS = {"R": (106, 56, "red"), "G": (78, 112, "disk_g"), "B": (134, 112, "blue")}
DISK_R = 46  # kept clear of the "Global cycle" title

# -- labels --------------------------------------------------------------------
LABELS = {
    "t1": "FD t1", "t2": "FD t2", "t3": "FD t3", "time": "Time",
    "cycle_prev": "Global cycle “n-1”", "cycle": "Global cycle “n”",
    "R": "R", "G": "G", "B": "B",
}

# -- drawing -------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": FONT_PT})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.set_aspect("equal")
ax.axis("off")
C = COLOURS
LINE = C["line"]


def rect(x0, x1, y0, y1, colour, edge=None, lw=1.2, z=2):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=C[colour],
                           edgecolor=edge or "none", lw=lw, zorder=z))


# additive colour disks; each overlap is the first disk clipped by the others
def disk(key, colour, clips=()):
    x, y = DISKS[key][:2]
    p = Circle((x, y), DISK_R, facecolor=C[colour], edgecolor="none", zorder=2 + len(clips))
    ax.add_patch(p)
    return p


for k, (_x, _y, col) in DISKS.items():
    disk(k, col)
for a, b, col in (("R", "G", "yellow"), ("R", "B", "magenta"), ("G", "B", "cyan")):
    disk(a, col).set_clip_path(Circle(DISKS[b][:2], DISK_R, transform=ax.transData))
# centre (R, G and B): polygon bounded by the three arcs, drawn in white
th = np.linspace(0, 2 * np.pi, 1440)
tri = []
for k, others in (("R", "GB"), ("G", "RB"), ("B", "RG")):
    cx, cy = DISKS[k][:2]
    for t in th:
        p = (cx + DISK_R * np.cos(t), cy + DISK_R * np.sin(t))
        if all((p[0] - DISKS[o][0]) ** 2 + (p[1] - DISKS[o][1]) ** 2 <= DISK_R ** 2
               for o in others):
            tri.append(p)
c0 = np.mean(tri, axis=0)
tri.sort(key=lambda p: np.arctan2(p[1] - c0[1], p[0] - c0[0]))
ax.fill([p[0] for p in tri], [p[1] for p in tri], color=C["white"], zorder=6, lw=0)
ax.text(DISKS["R"][0], DISKS["R"][1] - 22, LABELS["R"], ha="center", va="center",
        fontsize=FONT_PT * 1.25, color=C["text"], zorder=7)
ax.text(DISKS["G"][0] - 22, DISKS["G"][1] + 5, LABELS["G"], ha="center", va="center",
        fontsize=FONT_PT * 1.25, color=C["text"], zorder=7)
ax.text(DISKS["B"][0] + 22, DISKS["B"][1] + 5, LABELS["B"], ha="center", va="center",
        fontsize=FONT_PT * 1.25, color=C["text"], zorder=7)

# time axis with arrowhead
ax.plot([0, 1060], [Y_AXIS, Y_AXIS], color=LINE, lw=2.2, zorder=4, solid_capstyle="butt")
ax.annotate("", xy=(1085, Y_AXIS), xytext=(1050, Y_AXIS), zorder=4, arrowprops=dict(
    arrowstyle="-|>,head_length=0.7,head_width=0.3", color=LINE, lw=2.2,
    shrinkA=0, shrinkB=0))
ax.text(1095, Y_AXIS + 38, LABELS["time"], ha="center", va="center", color=C["text"])

# validity rectangles, then bars on top, then the cycle separator
for x0, x1, col in BOXES:
    rect(x0, x1, Y_BOX[0], Y_BOX[1], col, edge=LINE, lw=1.0, z=2)
for name, x in BARS.items():
    k = name[:2]
    rect(x - BAR_W / 2, x + BAR_W / 2, Y_BAR[0], Y_BAR[1] + 8, BAR_COLOUR[k], z=5)
    ax.text(x, Y_AXIS + 38, LABELS[k], ha="center", va="center",
            color=C[BAR_COLOUR[k]])
rect(X_SEP - 7, X_SEP + 7, Y_SEP[0], Y_SEP[1], "line", z=6)

# cycle titles (bold)
ax.text(X_SEP - 20, 187, LABELS["cycle_prev"], ha="right", va="center",
        fontweight="bold", color=C["text"])
ax.text(X_SEP + 20, 187, LABELS["cycle"], ha="left", va="center",
        fontweight="bold", color=C["text"])

# -- output (same bytes for the same script: fixed ids and dates) -------------
if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "esa-doc2open", "svg.fonttype": "path",
                         "pdf.fonttype": 42})
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.svg", format="svg", transparent=True, metadata={"Date": None})
    fig.savefig(f"{stem}.pdf", format="pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
