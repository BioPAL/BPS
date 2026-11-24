"""Fig. 10 — Schematic representation of the SWP alignment strategy implemented
during the Range Compression step.

ATBD L1 SAR (atbd-l1-sar), section 4.2.2 (SWP Alignment). Redrawn from the
original figure and the text of the section: the echoes (ECHO 1 ... ECHO N) are
received at times t[n], one every 1/PRF; each one is recorded on a grid of
samples spaced 1/f_s along the fast time tau. When the Sampling Window
Position changes along the orbit, a group of echoes starts at another tau, so
the echoes are shifted w.r.t. the first one (SWP[1], at tau_0). The delta SWP
w.r.t. the first echo, converted to samples and rounded up (SWP_sq), spans from
min(Delta SWP_sq) to max(Delta SWP_sq). This is a schematic: the levels and the
group widths are qualitative, taken from the original figure, not measured.

To modify: change the positions in GEOMETRY (pixels of the original figure,
x to the right, y downwards), the labels in LABELS or the colours in COLOURS,
then run ``python3 fig10.py`` (matplotlib needed): it rewrites
fig10.svg (web page) and fig10.pdf (PDF export) next to this file. Commit the
three files together. Keep a dark text colour for the labels: the page puts a white
background behind the figure in both site themes. The ECHO letters are white,
as in the original, on a dark blue that keeps them readable.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# -- colours and sizes -------------------------------------------------------
COLOURS = {"text": "#000000", "axis": "#262626", "grid": "#d4d4d4",
           "echo": "#2a68ab", "edge": "#0b3a6e", "echo_text": "#ffffff", "swp": "#ff0000"}
WIDTH_IN = 6.5  # PDF text width
FONT_PT = 11.5  # the PDF export prints it at about 65 %: subscripts stay >= 5 pt
ECHO_PT = 8.2  # the letters inside the echoes

# -- geometry (pixels of the original figure, y downwards) -------------------
X_MIN, X_MAX, Y_MIN, Y_MAX = -150, 1440, -55, 775  # canvas, with room for the larger text
ORIGIN = (221, 184)  # crossing of the t axis (y = 184) and the tau axis (x = 221)
T_AXIS = (198, 953)  # x extent of the t axis
TAU_AXIS = (160, 752)  # y extent of the tau axis
GRID_X = [279 + 46.5 * i for i in range(14)]  # echo boundaries (1/PRF)
GRID_Y = [230 + 46.5 * i for i in range(11)]  # sample boundaries (1/f_s)
GRID_X_END, GRID_Y_END = 884, 908  # last grid column / right end of the rows
# groups of echoes: (first boundary index, number of echoes, top y, bottom y)
GROUPS = [(0, 4, 276, 602), (4, 5, 346, 672), (9, 3, 246, 572)]
Y_TAU0P, Y_TAU0, Y_LOW = 230, 276, 369  # dash-dot lines
SWP_LINE = {Y_TAU0P: (256, 965), Y_TAU0: (256, 1275), Y_LOW: (256, 965)}
BRACKET_MIN = (978, 230, 273)  # x, y from, y to : min(Delta SWP_sq)
BRACKET_MAX = (1002, 276, 369)  # max(Delta SWP_sq)
ECHO_LETTERS_Y = 400  # first letter of the vertical captions, then every 30 px
ECHO_STEP = 30

# -- labels (mathtext), word for word from the original figure ----------------
LABELS = {
    "t": r"$t$", "tau": r"$\tau$", "t0": r"$t_0$",
    "tn": r"$t[n]$", "tn1": r"$t[n+1]$",
    "one": r"$1$", "prf": r"$PRF$",
    "tau0p": r"$\tau_0^{\prime}$", "tau0": r"$\tau_0$",
    "fs_num": r"$1$", "fs_den": r"$f_s$",
    "taum": r"$\tau[m]$", "taum1": r"$\tau[m+1]$",
    "min": r"$min(\Delta SWP_{sq})$", "max": r"$max(\Delta SWP_{sq})$",
    "swp1": r"$SWP[1]$",
    "echo1": ["E", "C", "H", "O", "", "1"],
    "echo2": ["E", "C", "H", "O", "", "2"],
    "dots": "...",
    "echoN": ["E", "C", "H", "O", "", "N"],
}

# -- drawing -------------------------------------------------------------------
plt.rcParams.update({"mathtext.fontset": "cm", "font.size": FONT_PT})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN * (Y_MAX - Y_MIN) / (X_MAX - X_MIN)))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(X_MIN, X_MAX)
ax.set_ylim(Y_MAX, Y_MIN)
ax.axis("off")
TXT, AXC = COLOURS["text"], COLOURS["axis"]


def text(x, y, s, ha="center", va="center", **kw):
    ax.text(x, y, s, color=TXT, ha=ha, va=va, **kw)


# grid, faint, as in the original
for x in GRID_X[:GRID_X.index(min(GRID_X, key=lambda v: abs(v - GRID_X_END))) + 1]:
    ax.plot([x, x], [160, GRID_Y[-1]], color=COLOURS["grid"], lw=0.6, zorder=0)
for y in GRID_Y:
    ax.plot([T_AXIS[0], GRID_Y_END], [y, y], color=COLOURS["grid"], lw=0.6, zorder=0)

# axes
ax.plot(T_AXIS, [ORIGIN[1]] * 2, color=AXC, lw=1.6, solid_capstyle="butt")
ax.plot([ORIGIN[0]] * 2, TAU_AXIS, color=AXC, lw=1.6, solid_capstyle="butt")

# the echoes: one rectangle per echo, grouped by SWP
for first, n, top, bot in GROUPS:
    for i in range(first, first + n):
        ax.add_patch(Rectangle((GRID_X[i], top), GRID_X[i + 1] - GRID_X[i], bot - top,
                               facecolor=COLOURS["echo"], edgecolor=COLOURS["edge"],
                               lw=2.0, zorder=2))


def vertical_caption(x, letters):
    for k, ch in enumerate(letters):
        if ch:
            ax.text(x, ECHO_LETTERS_Y + k * ECHO_STEP, ch, color=COLOURS["echo_text"],
                    ha="center", va="center", fontsize=ECHO_PT, fontweight="bold", zorder=4)


col = lambda i: (GRID_X[i] + GRID_X[i + 1]) / 2  # noqa: E731  centre of echo column i
vertical_caption(col(0), LABELS["echo1"])
vertical_caption(col(1), LABELS["echo2"])
ax.text(col(2), ECHO_LETTERS_Y + 2 * ECHO_STEP, LABELS["dots"], color=COLOURS["echo_text"],
        ha="center", va="center", fontsize=ECHO_PT, fontweight="bold", zorder=4)
vertical_caption(col(10), LABELS["echoN"])

# dash-dot SWP lines
for y, (x0, x1) in SWP_LINE.items():
    ax.plot([x0, x1], [y, y], color=COLOURS["swp"], lw=1.5, zorder=3,
            dashes=(7, 2.5, 1.5, 2.5))

# brackets min / max
for (x, y0, y1), key, ty in ((BRACKET_MIN, "min", 234), (BRACKET_MAX, "max", 328)):
    ax.plot([x, x], [y0, y1], color=AXC, lw=1.0)
    text(x + 22, ty, LABELS[key], ha="left")
text(SWP_LINE[Y_TAU0][1] + 26, Y_TAU0, LABELS["swp1"], ha="left")

# axis labels and times
text(941, 148, LABELS["t"])
text(178, 742, LABELS["tau"])
text(275, 136, LABELS["t0"])
text(578, 136, LABELS["tn"], ha="right")
text(598, 136, LABELS["tn1"], ha="left")
text(583, -24, LABELS["one"])
text(583, 52, LABELS["prf"])
ax.plot([547, 620], [10, 10], color=AXC, lw=1.0)  # fraction bar of 1/PRF
ax.plot([558, 604], [90, 90], color=AXC, lw=1.0)  # one echo period, t[n] to t[n+1]

# SWP levels and the sample spacing
text(185, 232, LABELS["tau0p"], ha="right")
text(185, 285, LABELS["tau0"], ha="right")
text(-95, 412, LABELS["fs_num"])
text(-95, 466, LABELS["fs_den"])
ax.plot([-125, -65], [440, 440], color=AXC, lw=1.0)  # fraction bar of 1/f_s
ax.plot([-20, -20], [405, 462], color=AXC, lw=1.0)
text(185, 415, LABELS["taum"], ha="right")
text(185, 466, LABELS["taum1"], ha="right")

# -- output (same bytes for the same script: fixed ids and dates) -------------
if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "esa-doc2open", "svg.fonttype": "path",
                         "pdf.fonttype": 42})
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.svg", format="svg", transparent=True, metadata={"Date": None})
    fig.savefig(f"{stem}.pdf", format="pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
