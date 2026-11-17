<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Editing the figures of the Science Guide

The drawings of the web ATBDs and PFDs (flowcharts, product structures,
schematic plots) are **text sources**, not images: you can correct a label, add
a processing step or change a value with a text editor. This page tells you
where each source is, how to change it and how to check the result. You do not
need to know Sphinx.

```{important}
**Read this first.** These pages are generated from the Word documents by the
converter esa-doc2open, which keeps the master copy of every figure source
(`src/esa_doc2open/emit/diagrams/<document>/`, one file per figure, listed in
`src/esa_doc2open/emit/diagram_figures.yaml`). A change made only in this
repository is **overwritten at the next conversion**: after you edit a figure
here, send the same change to esa-doc2open (see "Keeping your change" at the
end) or open an issue there with your patch.
```

## Where the figures are

Each document is a folder `docs/science-guide/<document>/` (for example
`atbd-l2-agb/`) with one Markdown page (`.md`) per chapter. A figure is a block
that starts with its label and ends with its caption:

````text
(fig:atbd-l2-agb-6)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

```{mermaid}
flowchart TB
  ...
```
...
<p class="atbd-figure-caption"><span class="caption-number">Fig. 6</span>...
````

| The figure is… | Its source is… | You edit… |
|---|---|---|
| a flowchart, a block diagram, a class diagram | the `{mermaid}` code block in the page | the page `.md` |
| a product structure (folder tree) | the `text` code block in the page | the page `.md` |
| a table drawn as a picture | the Markdown table in the page | the page `.md` |
| a schematic plot (axes, windows, geometry) | a Python script `_static/figures/<folder>/fig<N>.py`, next to the `fig<N>.svg` and `fig<N>.pdf` it produces | the `.py`, then regenerate (below) |
| a measured map, an image of real data, a figure taken from another publication | the image itself (it cannot be redrawn without inventing data) | nothing: ask the document owner for a new image |

A figure drawn in several parts, (a), (b), (c)…, is several `{mermaid}` blocks
one after the other, each preceded by a line
`<p class="atbd-subfigure-label">(c)</p>`; they share one label and one caption. Edit only the block of the part you
want (the caption comes after all the parts: read upwards to the part label).
Each part is an independent diagram (an arrow cannot join two parts: a link to
another part is drawn as a dotted reference box, "from (c): …"), and each part
has its own `accDescr` to update.

The figure folder of a script is named after the product, not always after the
document (`atbd-l2-agb` uses `docs/_static/figures/agb/`, `atbd-l2-fd` uses
`atbd-fd/`): the line
`![…](../../_static/figures/<folder>/fig<N>.*)` of the figure block gives the
folder and the script name (`fig<N>.py`). A figure can show a fixed image (a
map, a measured product) above a drawn one: Fig. 12 of `atbd-l2-agb` has the
land cover map on top (an image) and the block grid below (`fig12.py`).

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

## Editing a Mermaid diagram

A Mermaid diagram is plain text: nodes with an identifier and a label, then
the arrows between them. Vocabulary: a *node* is a box; its *identifier* is the
short name before the bracket (`cal`), used by the arrows; its *label* is the
text between quotes. A `classDef` line defines a *style class* (fill, border
and text colours) that nodes use with `:::name`. `accTitle` and `accDescr` are
the title and the description read aloud to blind readers: keep them true to
the figure.

```text
flowchart TB
  cal["Calibration"]            %% a processing step (rectangle)
  dem[/"DEM"/]                  %% data (parallelogram)
  prod[("L2a product")]         %% a product (cylinder)
  dem --> cal                   %% an arrow
  cal -- "sigma nought" --> prod  %% an arrow with a label
```

- **Rename a step**: change the text between quotes; keep the identifier
  (`cal`), which the arrows use.
- **Add a step**: add a node line, then the arrows to and from it.
- **Line break, index, exponent**: `<br/>`, `G<sub>i</sub>`, `m<sup>2</sup>`.
- **Two boxes with the same text** need two identifiers (`ion1["Ionosphere
  Removal"]`, `ion2["Ionosphere Removal"]`).
- **Colours**: the `classDef` lines at the top define the styles; a node uses
  one with `:::name` (`cal["Calibration"]:::step`). A colour is written
  `#rrggbb` (`#ffd9a0`); two classes may have the same colour (`data` and
  `step`): change each of them to recolour the whole figure. Always give both `fill` (box)
  and `color` (text) so that the text stays readable in the light and the dark
  themes of the site.
- **Groups**: `subgraph id["Title"]` … `end` draws a box around nodes (the
  coloured lanes); `style id fill:…,color:…` sets its colours.
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
- **Add a step inside a frame**: write its line between the `subgraph` line
  and its `end`; a node declared outside the frame is drawn outside it.
- **Renaming a frame**: keep the `&nbsp;` series after its title, they keep the
  title at the left, away from the arrows.
- After recolouring a class, update the comment above it (`%%   step: …`).
- `accTitle` and `accDescr` describe the diagram for screen readers: update
  `accDescr` when you change the content.
- Keep the first line, `%%{init: {"theme": "default", …}}%%`: it keeps the
  colours of the original figure in both site themes, and on large diagrams it
  also selects the layout engine that keeps them compact (`"layout": "elk"`).

To try a change, paste the whole block (from its first line to its last) into
<https://mermaid.live>: the drawing updates as you type, and an error message
names the faulty line. The layout there may differ from the site's (large
diagrams use the ELK engine of the site): the built page is the reference.

## Editing a Python figure

The script `fig<N>.py` draws the figure with matplotlib. Its first lines say
which figure it is and where its values come from; the values, labels and
colours are grouped at the top (`PARAMETERS`, `LABELS`, `COLOURS`).

1. Change a value or a label at the top of the script, in `PARAMETERS` or
   `GEOMETRY` (values), `LABELS` (texts) or `COLOURS`. Text between `$…$` is a
   formula (`$\sigma^0$`). In some scripts the label positions are written
   apart from the points they name (pixels of the original figure): after
   moving a point, move its label too, then look at the new `fig<N>.svg`.
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
   longer label may be cut at the edge; then move it, or enlarge the canvas
   (the size constants at the top of the script).
4. Commit the three files together (with Git, or send them to the document
   owner).

## Editing a schema diagram

The structures of the product formats (XML schemas) are Mermaid class
diagrams:

```text
classDiagram
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
- After a change, update `accDescr` and the `%%` comments above the classes.
- A schema diagram with a line "Generated from biomass-xsd" under it is
  produced from the XSD files: change the schema in biomass-xsd, not the page.

## Checking the result

Build the site from the `docs` folder, as described in the
[developer guide](../developer-guide/index.md):

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

The build takes about a minute; warnings about `grid-item` or about a page
"not included in any toctree" are not related to your change. Then open the
chapter page of the figure (reload it without the cache, Ctrl+Shift+R, after
regenerating a Python figure: run the build again first, it copies the new
`.svg`),
`_build/html/science-guide/<document>/<chapter>.html` (the `.md` file you
edited, with `.html`), and look at the figure in both themes (the sun/moon button at the top right). A Mermaid error
shows as a red message in place of the diagram; for a Python figure, check
that the build output mentions no missing image.

The PDF of a document (`make -C docs atbd-pdf SPHINX_ATBD_PDF=<document>`,
which needs LaTeX) prints the same diagrams. A very large diagram is printed on
a page of its own; the build warns when its text would be smaller than 4 pt:
then split the diagram into parts, (a), (b)…

When the meaning of a figure changed, update its `accDescr` (Mermaid) or the
caption text (table, tree) as well.

## Keeping your change

Commit the edited page (and, for a Python figure, the `.py`, `.svg` and `.pdf`
together), then make the same change in esa-doc2open: copy the whole Mermaid
block into the `.mmd` file of the figure, or the script and its two images, in
`src/esa_doc2open/emit/diagrams/<document>/`. If you cannot, open an issue
there with your patch: otherwise the next conversion brings the old figure
back.
