"""Fig. 12 (bottom image) — World covered by 3°x3° processing blocks (top) and
5°x5° neighbourhood for AGB estimation (bottom).

ATBD L2 AGB (atbd-l2-agb), section 4. Only the bottom image is redrawn here
(the top one, a world map, stays an image). From the text of section 4: each
3°×3° processing block (the blue 3 × 3 square, AGB estimation area) is
estimated from a 5°×5° neighbourhood (the green outer ring, AGB neighbourhood),
i.e. the block plus a 1° buffer zone. The 5 × 5 grid has one cell per degree;
the label "DGG Tile" names a cell of the grid, as in the original.

To modify: change the sizes in GEOMETRY, the labels in LABELS or the colours
in COLOURS, then run ``python3 fig12.py`` (matplotlib needed): it
rewrites fig12.svg (web page) and fig12.pdf (PDF export) next to this file.
Commit the three files together. Keep a dark text colour: the page puts a
white background behind the figure in both site themes.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# -- geometry (grid of N x N cells, one per degree; the estimation area is
# the central block of INNER x INNER cells) ----------------------------------
GEOMETRY = {"N": 5, "INNER": 3}
# -- labels, word for word from the original figure ---------------------------
LABELS = {"neigh": "AGB neighbourhood", "area": "AGB estimation area",
          "tile": "DGG\nTile"}
# -- colours and sizes -------------------------------------------------------
COLOURS = {
    "ring": "#b5d6a2", "ring_edge": "#375623", "ring_text": "#375623",
    "area": "#a2b8e0", "area_edge": "#203864", "area_text": "#1f4e79",
    "tile_text": "#000000",
}
WIDTH_IN = 4.6
FONT_PT = 11.5  # the PDF export prints it at about 65 %: text stays >= 5 pt

# -- drawing -------------------------------------------------------------------
N, INNER = GEOMETRY["N"], GEOMETRY["INNER"]
m = (N - INNER) // 2  # width of the ring, in cells
plt.rcParams.update({"font.size": FONT_PT})
fig = plt.figure(figsize=(WIDTH_IN, WIDTH_IN))
ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
ax.set_xlim(0, N)
ax.set_ylim(N, 0)
ax.set_aspect("equal")
ax.axis("off")
for i in range(N):
    for j in range(N):
        inner = m <= i < N - m and m <= j < N - m
        ax.add_patch(Rectangle(
            (i, j), 1, 1, lw=1.3,
            fc=COLOURS["area" if inner else "ring"],
            ec=COLOURS["area_edge" if inner else "ring_edge"]))
ax.text(N / 2, 0.38, LABELS["neigh"], color=COLOURS["ring_text"],
        ha="center", va="center", fontsize=FONT_PT * 1.25)
ax.text(N / 2, m + 0.38, LABELS["area"], color=COLOURS["area_text"],
        ha="center", va="center", fontsize=FONT_PT * 1.25)
ax.text(N / 2, N / 2, LABELS["tile"], color=COLOURS["tile_text"],
        ha="center", va="center", fontsize=FONT_PT * 1.25, linespacing=1.3)

# -- output (same bytes for the same script: fixed ids and dates) -------------
if __name__ == "__main__":
    plt.rcParams.update({"svg.hashsalt": "esa-doc2open", "svg.fonttype": "path",
                         "pdf.fonttype": 42})
    stem = Path(__file__).with_suffix("")
    fig.savefig(f"{stem}.svg", format="svg", transparent=True, metadata={"Date": None})
    fig.savefig(f"{stem}.pdf", format="pdf", transparent=True,
                metadata={"CreationDate": None, "ModDate": None})
