<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Editing the figures of the Science Guide

Most drawings of the web ATBDs and PFDs (flowcharts, product structures,
schematic plots) are **text sources**, not images: you can correct a label, add
a processing step or change a value with a text editor. This page tells you
where each source is, how to change it and how to check the result. You do not
need to know Sphinx.

```{important}
**Read this first.** These pages are generated from the Word documents by the
converter esa-doc2open. It keeps the master copy of every figure source, listed
in its mapping file `diagram_figures.yaml` (see "Keeping your change"). A
change made only in this repository is **overwritten at the next conversion**:
after you edit a figure here, send the same change to esa-doc2open, or open an
issue there with your patch. Some figures have one source shared by several
documents: a change to that source changes the figure in all of them.
```

## Where the figures are

Each document is a folder `docs/science-guide/<document>/` (for example
`atbd-l2-agb/`) with one Markdown page (`.md`) per chapter. A figure is a block
that starts with its label and ends with its caption (here Fig. 6 of the L2 AGB
ATBD, shortened):

````text
(fig:atbd-agb-calibration)=
::::{div} atbd-mermaid-figure atbd-drawn-figure

<div class="atbd-figure-content">

```{mermaid}
%%{init: {"flowchart": {"useMaxWidth": false}, …}}%%
flowchart TB
  ...
```

<!-- source image: figure_6_section_3.3.1 sha256 6ba29c6b… -->

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 6</span>…

::::
````

The comment `<!-- source image: … -->` gives the key of the figure in the
mapping file of the converter (`figure_6_section_3.3.1`) and the fingerprint
of the original image. Keep it.

What you edit depends on the kind of figure:

**Flowchart, block diagram, class diagram**
: the `{mermaid}` code block in the page `.md`.

**Product structure drawn with box characters**
: the `text` code block in the page `.md`.

**Coloured text tree (PFD metadata header)**
: nothing in this repository: see "Coloured text trees" below.

**Table drawn as a picture**
: the Markdown table in the page `.md`.

**Schematic plot (axes, windows, geometry)**
: the Python script `fig<N>.py` in `docs/_static/figures/<folder>/`, then
  regenerate its `.svg` and `.pdf` (see "Editing a Python figure").

**Measured map, image of real data, figure taken from another publication**
: nothing: the image cannot be redrawn without inventing data. Ask the
  document owner for a new image (see "Images").

A figure drawn in several parts, (a), (b), \(c\)…, is several `{mermaid}`
blocks one after the other, each preceded by a part label such as:

```text
<p class="atbd-subfigure-label">(c)</p>
```

The parts share one label and one caption. Edit only the block of the part you
want (the caption comes after all the parts: read upwards to the part label).
Each part is an independent diagram: an arrow cannot join two parts, so a link
to another part is drawn as a dotted reference box, "from \(c\): …".

The figure folder of a script is named after the product, not always after the
document: `atbd-l2-agb` uses `agb/`, `atbd-l2-fd` uses `atbd-fd/`,
`atbd-l2-fh` uses `atbd-fh/`. The script name is not always the figure number
either: Fig. 12 of `atbd-l2-fh` is drawn by `atbd-fh/fig11.py` (the same
drawing as Fig. 11 of the L2 AGB and L2 FD ATBDs), and the L1 SAR ATBD has
`img1-4.10.5.py`. The image line of the figure block gives both:

```text
![…](../../_static/figures/<folder>/fig<N>.*)
```

A figure can show a fixed image (a map, a measured product) above a drawn one:
Fig. 12 of `atbd-l2-agb` has the land cover map on top (an image) and the
block grid below (`fig12.py`).

The image of the approved Word document is not published next to a redrawn
figure: to compare, open the Word or PDF document (the converter keeps a
fingerprint of each original image and reports a change).

To find a figure, open the chapter page that shows it on the site (its address
ends with the page name, for example `05-calibration.html` →
`05-calibration.md`) and search the file for its number as written in the
caption, `Fig. 5</span>`; from a terminal:

```bash
grep -rn "Fig. 5</span>" docs/science-guide/atbd-l1-stack/
```

The source block is just above that line.

## Editing a table

A table is a Markdown table in the page: one line per row, cells separated by
`|`. Write a star as `\*` (a bare `*` makes italics). To add a row, copy a line
and change its cells; to add a column, add a cell to the header, to the
`|---|` line and to every row, so that each line has the same number of `|`.

## Editing a folder tree

A product structure is a `text` block drawn with box characters; it is shown
as is.

```text
BIOMASS FH L2b Product/
├── Main product header           [XML]
├── Preview folder/
│   ├── Quick-look ADS            [PNG]
│   └── Overlay ADS               [KMZ]
└── Schema folder/
    └── Support schemas RDS       [XSD]
```

- **Add a file**: copy a neighbouring line (the box characters `├── └── │` are
  not on the keyboard), change the name and the `[format]`. The last line of
  a folder starts with `└──`, all the others with `├──`: when you add after
  the last line, change the old last line to `├──` and give `└──` to the new
  one.
- **Remove a file**: delete its line; if it was the last of its folder, change
  the new last line to `└──`.
- **Rename a folder**: change its name (a folder name ends with `/`); its
  content is indented under it with `│   ` (four spaces under a `└──`).
- **Move a file**: delete its line and add it in the other folder, with the
  indentation of its new neighbours.
- Keep the `[format]` column aligned with spaces. Nothing checks the tree:
  look at the page.

## Coloured text trees

The main product header (MPH) of each PFD (Fig. 2) and the other metadata
trees of the PFD chapters are typed as text in the Word document, not drawn.
The converter copies them from the Word text:

- in the L1 PFD, Fig. 2 is a plain `text` block;
- in the L2 PFDs, a coloured tree (a figure with the classes
  `atbd-mermaid-figure` and `atbd-text-figure`, or a tree without caption)
  holds two blocks: a `{raw} html` block (the site: colours, bold and italics in
  `<span>`, `<b>`, `<i>` tags) followed by a `{raw} latex` block (the PDF
  export: the same tree with `\textcolor`, `\textbf`, `\textit`).

The colours, bold and italics follow the notation given just above the tree
(for example blue for a container of other elements). The dark theme
of the site lightens the red and the blue; the meaning is the same.

These trees have no source file in the converter: the next conversion writes
them again from the Word text, and **any edit made in the page is lost**. To
correct one, either ask the document owner to correct the Word document (an
error of the approved document), or change the converter esa-doc2open when the
conversion itself is wrong. Do not edit them in this repository.

## Editing a Mermaid diagram

A Mermaid diagram is plain text: nodes with an identifier and a label, then
the arrows between them. Vocabulary: a *node* is a box; its *identifier* is the
short name before the bracket (`cal`), used by the arrows; its *label* is the
text between quotes. A `classDef` line defines a *style class* (fill, border
and text colours) that nodes use with `:::name`. A line that starts with `%%`
is a comment.

```text
flowchart TB
  %% a processing step (rectangle)
  cal["Calibration"]
  %% data (parallelogram)
  dem[/"DEM"/]
  %% a product (cylinder)
  prod[("L2a product")]
  %% an arrow
  dem --> cal
  %% an arrow with a label
  cal -- "sigma nought" --> prod
```

Write a comment on its own line: Mermaid rejects `%%` after the code of a line.

- **Rename a step**: change the text between quotes; keep the identifier
  (`cal`), which the arrows use.
- **Add a step**: add a node line, then the arrows to and from it.
- **Line break, index, exponent**: `<br/>`, `G<sub>i</sub>`, `m<sup>2</sup>`.
- **Two boxes with the same text** need two identifiers (`ion1["Ionosphere
  Removal"]`, `ion2["Ionosphere Removal"]`).
- **Colours**: the `classDef` lines at the top define the styles; a node uses
  one with `:::name` (`cal["Calibration"]:::step`). A colour is written
  `#rrggbb` (`#ffd9a0`); two classes may have the same colour (`data` and
  `step`): change each of them to recolour the whole figure. Always give both
  `fill` (box) and `color` (text) so that the text stays readable in the light
  and the dark themes of the site. After recolouring a class, update the
  comment above it (`%%   step: …`).
- **Groups**: `subgraph id["Title"]` … `end` draws a box around nodes (the
  coloured lanes); `style id fill:…,color:…` sets its colours.
- **Add a step inside a frame**: write its line between the `subgraph` line
  and its `end`; a node declared outside the frame is drawn outside it.
- **Rename a frame**: keep the `&nbsp;` series after its title, they keep the
  title at the left, away from the arrows.
- **Invisible links** `a ~~~ b` only set the order of boxes (for example inside
  a group with no arrows); they draw nothing. When you add a step to such a
  group, add it to the chain: `a ~~~ new ~~~ b`.
- **Remove an arrow**: delete its line (`a --> b`). A chain such as
  `a --> b --> c` is one line but two arrows: to remove only `b --> c`,
  rewrite it as `a --> b`. A node left without any arrow is still drawn:
  delete its line too if it must disappear. Arrows are counted from 0 in the
  order they are read (each arrow of a chain counts, invisible links `~~~`
  too) by the `linkStyle n` lines that colour them: after adding or removing an
  arrow, check those numbers (most figures have no `linkStyle`).
- **Arrow styles**: `a --> b` solid, `a -.-> b` dotted (optional input),
  `a ==> b` thick, `a --- b` without head; a label goes in the middle:
  `a -- "text" --> b`.
- **Description for screen readers**: `accTitle` and `accDescr` are the title
  and the description read aloud to blind readers. When you change the content
  of a diagram (or of one part), update its `accDescr` so that it stays true to
  the figure.
- Keep the first line, `%%{init: …}%%`: it keeps the colours of the original
  figure in both site themes, and on large diagrams it also selects the layout
  engine that keeps them compact (`"layout": "elk"`).

To try a change, paste the whole block (from its first line to its last) into
<https://mermaid.live>: the drawing updates as you type, and an error message
names the faulty line. The layout there may differ from the site's (large
diagrams use the ELK engine of the site): the built page is the reference.

## Editing a Python figure

The script `fig<N>.py` draws the figure with matplotlib. Its first lines (the
docstring) say which figure it is, where its values come from and which
constants to change. The values are named constants at the top of the script:

- `LABELS`: the texts. Text between `$…$` is a formula (`$\sigma^0$`).
- `COLOURS`: the colours. Keep a dark text colour: the page puts a white
  background behind the figure in both site themes.
- The positions and values: `GEOMETRY` in most scripts, or constants named
  after what they draw (`ORIGIN`, `X_AXIS`, `BARS`…), as the docstring says.
- `WIDTH_IN` and `FONT_PT`: the size of the drawing and of its text.

1. Change a value or a label at the top of the script. In some scripts the
   label positions are written apart from the points they name (`LABEL_AT`,
   in pixels of the original figure): after moving a point, move its label
   too.
2. Regenerate the images, in the figure folder:

   ```bash
   cd docs/_static/figures/<folder>      # for atbd-l2-agb: agb
   python3 fig<N>.py
   ```

   This rewrites `fig<N>.svg` (web page) and `fig<N>.pdf` (PDF export). You
   need Python 3 with matplotlib (`pip install matplotlib`). A warning about a
   "timestamp" is harmless (the files get a fixed date so that the same script
   always gives the same files).
3. Look at the new `fig<N>.svg`: the drawing area has a fixed size, so a
   longer label may be cut at the edge; then move it, or enlarge the drawing
   (`WIDTH_IN`).
4. If the script is shared (see "Keeping your change"), copy the three files
   to the other figure folders that have the same script: `agb/fig11.py`,
   `atbd-fd/fig11.py` and `atbd-fh/fig11.py` are the same drawing.
5. Commit the three files together (with Git, or send them to the document
   owner).

## Editing a schema diagram

The structures of the product formats (XML schemas) are Mermaid class
diagrams:

```text
classDiagram
  direction LR
  class lutAnnotationType {
    <<complexType>>
    Attributes [18]
    History [0..1]
    FNF : fnf_variables [1]
  }
  lutAnnotationType --> fnf_variables : FNF
```

- One line per element: its name, `: type` when the figure gives it, then its
  occurrences: `[n]` exactly n, `[0..1]` optional, `[1..30]` 1 to 30.
  `@count` is an attribute; `<<complexType>>` is the kind of box. An arrow
  links a type to the type of one of its elements.
- **Add an element**: add its line inside the braces of its class. **Change
  a cardinality**: edit the brackets (plain text, not checked).
- `direction LR` draws the levels from left to right, `direction TB` from top
  to bottom.
- After a change, update the `%%` comments above the classes.

A schema diagram with a line "Generated from biomass-xsd" under it is produced
from the XSD files by the converter: change the schema in biomass-xsd, then
convert the documents again with esa-doc2open (an edit of the page is lost at
the next conversion). The converter lays such a diagram out from left to
right, unless it would be wider than about 1600 px: it then stacks the levels
from top to bottom when that is narrower. To force the top-to-bottom layout,
give the figure `direction: TB` in the mapping file `xsd_figures.yaml` of
esa-doc2open (as for Fig. 33 of the L1 PFD, Overlay ADS).

## Images

A figure that stays an image (measured data, a figure taken from another
publication) is a PNG file in `docs/_static/figures/<folder>/`, named after
its place in the Word document, for example:

```text
figure_<N>_section_<S>.png
```

- To replace it, ask the document owner for the new image and keep the file
  name. Every image of the folder must be shown by a page: the converter check
  "no stray raster" fails on an image that no page uses.
- A figure taken from another publication keeps its credit. The caption cites
  the source (L1 SAR ATBD Fig. 12, "image taken from [AD2]"), or a credit line
  follows the caption ("Reproduced from [RD6]." after Fig. 8 of the L2 FH
  ATBD). That line comes from the field `credit` of the figure in
  `diagram_figures.yaml`: keep it, or change it there.
- An image drawn in Word as EMF or WMF is rendered on a whole page, then
  published cut to its content with a 12 px margin. The fingerprint
  `media_sha256` kept in the mapping is that of the whole page: the published
  file has another fingerprint, which is normal.

## Checking the result

Build the site from the `docs` folder, as described in the
[contributing guide](../contributing/index.md):

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

The build of the whole site takes about a minute; warnings about `grid-item`
or about a page "not included in any toctree" are not related to your change.
Then open the chapter page of the figure (reload it without the cache,
Ctrl+Shift+R, after regenerating a Python figure: run the build again first, it
copies the new `.svg`):

```text
_build/html/science-guide/<document>/<chapter>.html
```

that is, the `.md` file you edited, with `.html`. Look at the figure in both
themes (the sun/moon button at the top right). A Mermaid error shows as a red
message in place of the diagram; for a Python figure, check that the build
output mentions no missing image.

The PDF of a document (needs LaTeX) prints the same diagrams:

```bash
make -C docs atbd-pdf ATBD=<document>     # for example ATBD=pfd-l2-fh
```

A very large diagram is printed on a page of its own; the build warns when its
text would be smaller than 4 pt: then split the diagram into parts, (a),
(b)…

## Keeping your change

Commit the edited page (and, for a Python figure, the `.py`, `.svg` and `.pdf`
together), then make the same change in esa-doc2open. Its figure sources are
in `src/esa_doc2open/emit/`:

- `diagrams/<document>/`: one file per figure, `.mmd` (Mermaid), `.txt`
  (folder tree), `.md` (table), or `.py` with its `.svg` and `.pdf`; the
  parts of a figure are `fig<N>a.mmd`, `fig<N>b.mmd`…;
- `diagram_figures.yaml`: the mapping, one entry per figure, found by the key
  of the comment `<!-- source image: <key> … -->` of the page.

Copy the whole Mermaid block into the `.mmd` file named by the field `source`
(or `parts`) of the figure, the Markdown table into its `.md`, the tree into
its `.txt`, or the script and its two images into the folder of the `.py`. A
new part needs its own `.mmd` file, declared in the field `parts` of the
figure: the converter emits only the parts listed there.

**Sources shared by several documents.** A figure copied in several documents
has one source, in `atbd-l2-agb/`:

- Fig. 1 of the 5 ATBDs: `fig1.mmd`;
- Fig. 2 of the 5 ATBDs: `fig2a.mmd` to `fig2e.mmd` (five parts);
- Fig. 4 and 5 of the L2 AGB, FD and FH ATBDs: `fig4.md`, `fig5.md`;
- Fig. 6, 7, 8 and 10 of the L2 AGB and FD ATBDs: `fig6.mmd`, `fig7.mmd`,
  `fig8.mmd`, `fig10.mmd`;
- Fig. 11 of the L2 AGB and FD ATBDs **and Fig. 12 of the L2 FH ATBD**:
  `fig11.py` (in this repository: `agb/fig11.py`, `atbd-fd/fig11.py`,
  `atbd-fh/fig11.py`, three identical copies).

So, for the L2 FD ATBD, Fig. 1, 2, 4 to 8, 10 and 11 are in `atbd-l2-agb/`;
only the others are in `atbd-l2-fd/` (for example `fig16.py`). For the L2 FH
ATBD, Fig. 1, 2, 4, 5 and 12 are in `atbd-l2-agb/`; the others are in
`atbd-l2-fh/`. A change to a shared source changes the figure in **every
document that uses it**: make the same change in each of their pages in this
repository (for a Python figure, in each figure folder), or the pages disagree
until the next conversion.

**Checking in the converter.** Run `doc2open quality diagrams` in esa-doc2open.
It compiles every Mermaid source, and fails when:

- the number of nodes or arrows differs from the fields `nodes` and `edges` of
  the figure in `diagram_figures.yaml` (summed over the parts): after adding or
  removing a step or an arrow, update them;
- a label of the original figure, listed in `<name>.labels.txt` next to the
  source, is no longer in it: when you rename a step on purpose, update that
  list;
- the `.svg` and `.pdf` of a Python figure are not exactly what its script
  gives: run the script again.

If you cannot make the change there, open an issue in esa-doc2open with your
patch: otherwise the next conversion brings the old figure back.
