<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) and ARESYS -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

:::{atbd-logo-banner}
:::


(sec:atbd-fh-4)=
# 4. Forest height aggregation

The overall scheme of the L2b processing is reported in figure below.

The key step is fusion of L2a estimates at map level. These steps are described in detail in the following subsections.

(sec:atbd-fh-4-1)=
### 4.1 L2a FH fusion

(sec:atbd-fh-4-1-1)=
#### 4.1.1 Algorithm overview and rationale

The forest height and quality maps allow the combination of different acquisitions in a simple and efficient pixelwise way. L2a products falling onto the same L2b tile within a global cycle are retrieved through geospatial query of information (tile IDs and time tags) in the Main Product Header of the product.

(fig:atbd-fh-13)=
::::{div} atbd-mermaid-figure atbd-drawn-figure

<div class="atbd-figure-content">

```{mermaid}
%%{init: {"flowchart": {"useMaxWidth": false}, "class": {"useMaxWidth": false}, "theme": "default"}}%%
flowchart TB
  accTitle: Complete flow-chart of the L2b processing.
  accDescr: The DGG, the group of Forest Height L2a, FNF and Forest Height Quality L2a for all stacks on an area in a given time, and FD L2b shown by a dashed arrow feed Combining L2a products. It gives Aggregated Forest Height L2b and Aggregated Forest Height Quality L2b.

  %% Colours of the original figure.
  %%   data: input or product (light grey parallelogram)
  %%   step: processing step (light grey box)
  classDef data fill:#f2f2f2,stroke:#000000,color:#000000
  classDef step fill:#f2f2f2,stroke:#000000,color:#000000

  dgg[/"DGG"/]:::data
  fd[/"FD L2b"/]:::data

  %% Dotted frame of the original; the &nbsp; push its title to the left.
  subgraph STACKS["All stacks, on an area, in a given time&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"]
    direction LR
    fh[/"Forest Height<br/>L2a"/]:::data
    fnf[/"FNF"/]:::data
    fhq[/"Forest Height<br/>Quality L2a"/]:::data
    fh ~~~ fnf ~~~ fhq
  end

  comb["Combining L2a<br/>products"]:::step
  afh[/"Aggregated<br/>Forest Height<br/>L2b"/]:::data
  afhq[/"Aggregated<br/>Forest Height<br/>Quality L2b"/]:::data

  dgg --> comb
  STACKS --> comb
  fd -.-> comb
  comb --> afh
  comb --> afhq

  style STACKS fill:#ffffff,stroke:#000000,color:#000000,stroke-dasharray:2 2
```

<!-- source image: figure_13_section_4.1.1 sha256 fb547cf29f87c8c8cb4f6ff1d5f552d553f2abe1fb9cd0ac1e12eaff53f6632b -->

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 13</span><span class="caption-text"> — Complete flow-chart of the L2b processing.</span></p>

::::

**The step is mandatory.**

(sec:atbd-fh-4-1-2)=
#### 4.1.2 Inputs

| Symbol | Name | Origin | Size | Variability | Default | Description |
|---|---|---|---|---|---|---|
| *FNF* {sup}`i` | FNF | FH L2a products in a DGG tile in a global cycle (`FH_PFD`) | $\left[ N_{lat}\times{}N_{lon} \right]^i$ | - | - | Annotated forest-non-forest mask |
| $H_F^i$ | forest height | FH L2a products in a DGG tile in a global cycle (`FH_PFD`) | $\left[ N_{lat}\times{}N_{lon} \right]^i$ | - | - | Forest height map |
| $h_b^i$ | forest height quality | FH L2a products in a DGG tile in a global cycle (`FH_PFD`) | $\left[ N_{lat}\times{}N_{lon} \right]^i$ | - | - | Quality index indicating the inversion performance |
| *CFM* | CFM | FD L2b product in a DGG tile in a global cycle (`FD_PFD`) | $\left[ N_{lat,L2b}\times{}N_{lon,L2b} \right]$ | optional | not provided | Computed Forest Mask |
| - | `forestMaskingFlag` | `AUX_PP2_FH` (`AUX_FMT`) | bool | - | True | True if forest/non-forest map masking should be performed |
| *DGG* | DGG | Internal lookup table | - | - | - | DGG tile |
| *BB*{sup}`i` | Footprint | L2a main product header (`FH_PFD`) | lat-lon vertices (4 couples) | - | - | The extent of the L2a input data in geographical coordinates |
| $\beta_A$ | `rollOffFactorAzimuth` | `AUX_PP2_FH` (`AUX_FMT`) | float value [scalar] | 0.0-1.0 | 0.1 | Feathering roll-off factor |
| $\beta_R$ | `rollOffFactorRange` | `AUX_PP2_FH` (`AUX_FMT`) | float value [scalar] | 0.0-1.0 | 0.1 | Feathering roll-off factor |
| *mDGG* | `minimumL2aCoverage` | `AUX_PP2_FH` (`AUX_FMT`) | float value [scalar] | 0.0-100.0 | 10.0 | The minimum [%] tile coverage that triggers the computations |

(sec:atbd-fh-4-1-3)=
#### 4.1.3 Outputs

| Symbol | Name | Destination | Size | Variability | Default | Description |
|---|---|---|---|---|---|---|
| $H_F$ | forest height | FH L2b product | $\left[ N_{lat,L2b}\times{}N_{lon,L2b} \right]$ | - | - | forest height aggregated |
| $h_b$ | forest height quality | FH L2b product | $\left[ N_{lat,L2b}\times{}N_{lon,L2b} \right]$ | - | - | forest height quality aggregated |
| *HM* | heat map | FH L2b product | $\left[ N_{lat,L2b}\times{}N_{lon,L2b} \right]$ | - | - | heat map |

(sec:atbd-fh-4-1-4)=
#### 4.1.4 Algorithm mathematical background

As a first step, the minimum required coverage over L2b DGG tile *j* is checked to proceed with computations:

$$
TR=\ \left( 100*\frac{A(\left. \bigcup_i^{}{{BB}^i} \right\vert_{DGG}\cap\left. {TT}^j \right\vert_{DGG})}{A(\left. {TT}^j \right\vert_{DGG})}>mDGG \right)
\tag{4.1}
$$ (eq:atbd-fh-4-1)

where *BB*{sup}`i` are the L2a bounding boxes, *TT*{sup}`j` is the footprint of the current L2b DGG tile and *A* operator computes the area of the corresponding polygon.

(sec:atbd-fh-4-1-4-1)=
##### 4.1.4.1 Feathering weights

Each L2a forest height map is weighted on its border (as defined by its footprint *BB*{sup}`i`) by applying raised cosine weights and accounting for transitions between swaths {rd}`14`.

Thus, the two feathering weights $w_{\mathrm{a}}^i(d)w_{\mathrm{r}}^i(d)$ are computed in the swath tilted domain, as defined by the footprint *BB*{sup}`i`:

(equation-eq-atbd-fh-4-2)=
\begin{equation}
w_{a,r}^i(d)=\left\{ \ \ \begin{matrix}
1\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ & \left\vert d \right\vert\leq\frac{1-\beta}{\mathrm{s}_{\mathrm{w}}}\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \\
\frac{1}{2}\left[ 1+\cos \left( \frac{\pi\mathrm{s}_{\mathrm{w}}}{2\beta}\left[ \left\vert d \right\vert-\frac{1-\beta}{\mathrm{s}_{\mathrm{w}}} \right] \right) \right]\ \ \ \ \ \ \ \ \ \ & \frac{1-\beta}{\mathrm{s}_{\mathrm{w}}}<\left\vert d \right\vert\leq\frac{1+\beta}{\mathrm{s}_{\mathrm{w}}}\ \ \ \\
\ 0\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ & otherwise\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
\end{matrix} \right.
\tag{4.2}
\end{equation}

where $\mathrm{s}_{\mathrm{w}}=s_{w,A},s_{w,R}$ is swath width along azimuth or range direction (computed from swath footprint *BB*{sup}`i`), $d$ is distance from swath border (computed in the tilted domain) and $\beta=\beta_A,\ \beta_R$ is the roll-off factor.

(fig:atbd-fh-14)=
:::{div} atbd-mermaid-figure

<div class="atbd-figure-content">

![Feathering weights example. The product of range and azimuth weights is represented colour-coded.](../../_static/figures/atbd-fh/figure_14_section_4.1.4.1.png)

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 14</span><span class="caption-text"> — Feathering weights example. The product of range and azimuth weights is represented colour-coded.</span></p>

:::

(sec:atbd-fh-4-1-4-2)=
##### 4.1.4.2 Aggregation rule

The aggregation at pixel map level of the various L2a forest height maps is done by performing an average weighted by quality information represented by the percentage bias $h_b^i$, computed in {sec}`3.5.4.1.6`. The estimation quality $h_b^i$ allows the combination of K different estimates $H_F^i$ in a simple and efficient way to generate a global $H_F$ estimate. A heat map *HM* is also stored as representing the sum of weights used in computation (including feathering weights). Two different sets of relations are proposed, depending on the availability of FD L2b product as input. When FD L2b product is not present, annotated forest/non-forest map is used to remove deforested areas:

$$
H_F=\frac{\sum_{i=1}^K{\frac{1}{{\varepsilon+h}_b^i}w_{\mathrm{a}}^i(d)w_{\mathrm{r}}^i(d)H_F^i{FNF}^i}}{\sum_{i=1}^K{\frac{1}{{\varepsilon+h}_b^i}}w_{\mathrm{a}}^i(d)w_{\mathrm{r}}^i(d)}
\tag{4.3}
$$ (eq:atbd-fh-4-3)

$$
h_b=\frac{\sum_{i=1}^K{\frac{1}{{\varepsilon+h}_b^i}w_{\mathrm{a}}^i(d)w_{\mathrm{r}}^i(d)h_b^i{FNF}^i}}{\sum_{i=1}^K{\frac{1}{{\varepsilon+h}_b^i}}w_{\mathrm{a}}^i(d)w_{\mathrm{r}}^i(d)}
$$

$$
HM=\sum_{i=1}^K{\frac{1}{{\varepsilon+h}_b^i}w_{\mathrm{a}}^i(d)w_{\mathrm{r}}^i(d){FNF}^i}
$$

where $\varepsilon=0.01$ is an adjustment factor.

If the FD L2b product is provided as input, *CFM* supersedes *FNF*, and the rule is instead:

$$
H_F=CFM\ \frac{\sum_{i=1}^K{\frac{1}{{\varepsilon+h}_b^i}w_{\mathrm{a}}^i(d)w_{\mathrm{r}}^i(d)H_F^i}}{\sum_{i=1}^K{\frac{1}{{\varepsilon+h}_b^i}}w_{\mathrm{a}}^i(d)w_{\mathrm{r}}^i(d)}
\tag{4.4}
$$ (eq:atbd-fh-4-4)

$$
h_b=CFM\ \frac{\sum_{i=1}^K{\frac{1}{{\varepsilon+h}_b^i}w_{\mathrm{a}}^i(d)w_{\mathrm{r}}^i(d)h_b^i}}{\sum_{i=1}^K{\frac{1}{{\varepsilon+h}_b^i}}w_{\mathrm{a}}^i(d)w_{\mathrm{r}}^i(d)}
$$

$$
HM=CFM\sum_{i=1}^K{\frac{1}{{\varepsilon+h}_b^i}w_{\mathrm{a}}^i(d)w_{\mathrm{r}}^i(d)}
$$

It must be noted that previous relations are actually computed only where at least one FH L2A input is present.
