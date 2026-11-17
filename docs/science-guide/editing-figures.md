<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Editing the figures of the Science Guide

The drawings of the web ATBDs and PFDs (flowcharts, product structures,
schematic plots) are **text sources**, not images: you can correct a label, add
a processing step or change a value with a text editor. This page tells you
where each source is, how to change it and how to check the result. You do not
need to know Sphinx.

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

Under every redrawn figure, the **Original figure (DOCX)** dropdown keeps the
image of the approved Word document, for comparison.

```{important}
The pages are generated from the Word documents by the converter
esa-doc2open, which holds the master copy of every
figure source in `src/esa_doc2open/emit/diagrams/<document>/` (one file per
figure, listed in `src/esa_doc2open/emit/diagram_figures.yaml`). A change made
only in BPS is **overwritten at the next conversion**. When you change a figure
here, make the same change in esa-doc2open (copy the whole Mermaid block into
the `.mmd` file, or the `.py` script and its `.svg` and `.pdf`), or open an
issue there with your patch.
```

## Editing a Mermaid diagram

A Mermaid diagram is plain text: nodes with an identifier and a label, then
the arrows between them.

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
- **Colours**: the `classDef` lines at the top define the styles; a node uses
  one with `:::name` (`cal["Calibration"]:::step`). Always give both `fill` (box)
  and `color` (text) so that the text stays readable in the light and the dark
  themes of the site.
- **Groups**: `subgraph id["Title"]` … `end` draws a box around nodes (the
  coloured lanes); `style id fill:…,color:…` sets its colours.
- `accTitle` and `accDescr` describe the diagram for screen readers: update
  `accDescr` when you change the content.
- Keep the first line, `%%{init: {"theme": "default", …}}%%`: it keeps the
  colours of the original figure in both site themes, and on large diagrams it
  also selects the layout engine that keeps them compact (`"layout": "elk"`).

To try a change, paste the whole block (from its first line to its last) into
<https://mermaid.live>: the drawing updates as you type, and an error message
names the faulty line.

## Editing a Python figure

The script `fig<N>.py` draws the figure with matplotlib. Its first lines say
which figure it is and where its values come from; the values, labels and
colours are grouped at the top (`PARAMETERS`, `LABELS`, `COLOURS`).

1. Change a value or a label at the top of the script. Labels between `$…$`
   are formulas (`$\sigma^0$`).
2. Regenerate the images, in the figure folder:

   ```bash
   cd docs/_static/figures/<folder>
   python3 fig<N>.py
   ```

   This rewrites `fig<N>.svg` (web page) and `fig<N>.pdf` (PDF export). You
   need Python 3 with matplotlib (`pip install matplotlib`).
3. Commit the three files together.

## Checking the result

Build the site from the `docs` folder, as described in the
[developer guide](../developer-guide/index.md):

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html . _build/html
```

then open `_build/html/science-guide/<document>/index.html` and look at the
figure in both themes (the sun/moon button at the top right). A Mermaid error
shows as a red message in place of the diagram; for a Python figure, check
that the build output mentions no missing image.

The PDF of a document (`make -C docs atbd-pdf SPHINX_ATBD_PDF=<document>`,
which needs LaTeX) prints the same diagrams. A very large diagram is printed on
a page of its own; if its text would be smaller than 4 pt, the PDF prints the
original figure instead and the build reports it.
