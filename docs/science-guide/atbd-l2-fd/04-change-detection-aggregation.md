<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:atbd-fd-4)=
# 4. Change detection aggregation

The overall scheme of the L2b processing is reported in figure below.

The key step is fusion of L2a estimates at map level. These steps are described in detail in the following subsections.

**The step is mandatory.**

(fig:atbd-fd-14)=
::::{div} atbd-mermaid-figure atbd-drawn-figure

<div class="atbd-figure-content">

```{mermaid}
%%{init: {"flowchart": {"useMaxWidth": false}, "class": {"useMaxWidth": false}, "theme": "default"}}%%
flowchart TB
  accTitle: Complete flow-chart of the L2b processing.
  accDescr: Within the group All stacks, on an area, in a given time, Forest Disturbance L2a, Probability of change L2a and CFM L2a and DGG feed the L2b processing group, which contains Combining L2a products. The L2b processing group outputs Aggregated Forest Disturbance L2b, Aggregated probability of change L2b and Aggregated CFM L2b.

  %% Colours of the original figure. Each class sets its text colour too.
  %%   data:  input or output (parallelogram, light grey)
  %%   step:  processing step (light grey box)
  classDef data fill:#f2f2f2,stroke:#7f7f7f,color:#000000
  classDef step fill:#f2f2f2,stroke:#7f7f7f,color:#000000

  dgg[/"DGG"/]:::data

  subgraph stacks["All stacks, on an area, in a given time"]
    fd[/"Forest<br/>Disturbance<br/>L2a"/]:::data
    poc[/"Probability<br/>of change L2a"/]:::data
    cfm[/"CFM L2a"/]:::data
  end

  subgraph l2b["L2b processing"]
    comb["Combining L2a<br/>products"]:::step
  end

  afd[/"Aggregated<br/>Forest<br/>Disturbance L2b"/]:::data
  apoc[/"Aggregated<br/>probability of<br/>change L2b"/]:::data
  acfm[/"Aggregated<br/>CFM L2b"/]:::data

  style stacks fill:#ffffff,stroke:#000000,color:#000000,stroke-dasharray:2 2
  style l2b fill:#ffffff,stroke:#000000,color:#000000,stroke-dasharray:2 2

  fd --> l2b
  poc --> l2b
  cfm --> l2b
  dgg --> l2b
  l2b --> afd
  l2b --> apoc
  l2b --> acfm
```

<!-- source image: figure_14_section_4 sha256 ce75f05974fb472915cdb066d8ebed9c2128db6b7bcdc4f8f5c6cd4c6714736a -->

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 14</span><span class="caption-text"> — Complete flow-chart of the L2b processing.</span></p>

::::

(sec:atbd-fd-4-1)=
### 4.1 Fusion of L2a products on a Tile within a fixed timespan

(sec:atbd-fd-4-1-1)=
#### 4.1.1 Overview and rationale

The disturbance maps allow the combination of different acquisitions in a simple and efficient pixelwise way. L2a products falling onto the same L2b tile within a global cycle are retrieved through geospatial query of information (tile IDs and time tags) in the Main Product Header of the product.

From the disturbance map combined, the extraction of the combined CFM is obtained by a simple logical *and* operator.

(fig:atbd-fd-15)=
::::{div} atbd-mermaid-figure atbd-drawn-figure

<div class="atbd-figure-content">

```{mermaid}
%%{init: {"flowchart": {"useMaxWidth": false}, "class": {"useMaxWidth": false}, "theme": "default"}}%%
flowchart TB
  accTitle: L2a FD fusion.
  accDescr: Within the group All stacks, on an area, in a given time, DGG, Probability of change L2a, Forest Disturbance L2a and CFM L2a are shown. Probability of change L2a and Forest Disturbance L2a feed Temporal sorting, followed by Series time derivative, FD steps count and Consistency check. Consistency check feeds Probability of change averaging, which gives Aggregated Probability of change L2b, and gives Aggregated Forest Disturbance L2b. CFM L2a and Aggregated Forest Disturbance L2b feed Update CFM, which gives Aggregated CFM L2b.

  %% Colours of the original figure. Each class sets its text colour too.
  %%   data:  input or output (parallelogram, light grey)
  %%   step:  processing step (light grey box)
  classDef data fill:#f2f2f2,stroke:#7f7f7f,color:#000000
  classDef step fill:#f2f2f2,stroke:#7f7f7f,color:#000000

  subgraph stacks["All stacks, on an area, in a given time"]
    dgg[/"DGG"/]:::data
    poc[/"Probability of<br/>change L2a"/]:::data
    fd[/"Forest<br/>Disturbance L2a"/]:::data
    cfm[/"CFM L2a"/]:::data
  end
  style stacks fill:#ffffff,stroke:#000000,color:#000000,stroke-dasharray:2 2

  sort["Temporal sorting"]:::step
  deriv["Series time<br/>derivative"]:::step
  count["FD steps count"]:::step
  check["Consistency check"]:::step
  avg["Probability of<br/>change averaging"]:::step
  ucfm["Update CFM"]:::step

  apoc[/"Aggregated<br/>Probability of<br/>change L2b"/]:::data
  afd[/"Aggregated<br/>Forest<br/>Disturbance L2b"/]:::data
  acfm[/"Aggregated<br/>CFM L2b"/]:::data

  poc --> sort
  fd --> sort
  dgg ~~~ poc
  sort --> deriv --> count --> check
  check --> avg
  check --> afd
  avg --> apoc
  afd --> ucfm
  cfm --> ucfm
  ucfm --> acfm
```

<!-- source image: figure_15_section_4.1.1 sha256 a3620ce9038f87a755bc8b18fe61ebb522c2f5ef85ec9eb57cd93c740ac9668e -->

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 15</span><span class="caption-text"> — L2a FD fusion.</span></p>

::::

(sec:atbd-fd-4-1-2)=
#### 4.1.2 Inputs

| Symbol | Name | Origin | Size | Variability | Default | Description |
|---|---|---|---|---|---|---|
| ${{FD}_{map}}^{\mathrm{t}^{\mathrm{i}}}$ | Forest Disturbance | FD L2a products in a DGG tile in a global cycle (`FD_PFD`) | $\left[ N_{lat}\times{}N_{lon} \right]^{\mathrm{t}^{\mathrm{i}}}$ | - | - | Forest Disturbance map |
| ${CFM}^{\mathrm{t}^{\mathrm{i}}}$ | CFM | FD L2a products in a DGG tile in a global cycle (`FD_PFD`) | $\left[ N_{lat}\times{}N_{lon} \right]^{\mathrm{t}^{\mathrm{i}}}$ | - | - | CFM |
| ${P_{\mathrm{c}}}^{\mathrm{t}^{\mathrm{i}}}$ | Probability of change | FD L2a products in a DGG tile in a global cycle (`FD_PFD`) | $\left[ N_{lat}\times{}N_{lon} \right]^{\mathrm{t}^{\mathrm{i}}}$ | - | - | Probability of change map |
| *DGG* | DGG | Internal lookup table | - | - | - | DGG tile |
| *BB*{sup}`i` | Footprint | L2a main product header (`FD_PFD`) | lat-lon vertices (4 couples) | - | - | The extent of the L2a input data in geographical coordinates |
| *mDGG* | `minimumL2aCoverage` | `AUX_PP2_` FD (`AUX_FMT`) | float value [scalar] | 0.0-100.0 | 10.0 | The minimum [%] tile coverage that triggers the computations |

(sec:atbd-fd-4-1-3)=
#### 4.1.3 Outputs

| Symbol | Name | Destination | Size | Variability | Default | Description |
|---|---|---|---|---|---|---|
| ${{FD}_{map}}^{aggregated}$ | Forest Disturbance | FD L2 b product (`FD_PFD`) | $\left[ N_{lat,L2b}\times{}N_{lon,L2b} \right]$ | - | - | Forest Disturbance map aggregated |
| ${CFM}^{aggregated}$ | CFM | FD L2 b product (`FD_PFD`) | $\left[ N_{lat,L2b}\times{}N_{lon,L2b} \right]$ | - | - | CFM aggregated |
| ${P_{\mathrm{c}}}^{aggregated}$ | Probability of change | FD L2 b product (`FD_PFD`) | $\left[ N_{lat,L2b}\times{}N_{lon,L2b} \right]$ | - | - | Probability of change map aggregated |
| *HM* | heat map | FD L2b product (`FD_PFD`) | $2\left[ N_{lat,L2b}\times{}N_{lon,L2b} \right]$ | 0- | - | heat map (couple of integer values per pixel) |

(sec:atbd-fd-4-1-4)=
#### 4.1.4 Algorithm mathematical background

As a first step, the minimum required coverage over L2b DGG tile *j* is checked to proceed with computations:

$$
TR=\ \left( 100*\frac{A(\left. \bigcup_i^{}{{BB}^i} \right\vert_{DGG}\cap\left. {TT}^j \right\vert_{DGG})}{A(\left. {TT}^j \right\vert_{DGG})}>mDGG \right)
\tag{4.1}
$$ (eq:atbd-fd-4-1)

where *BB*{sup}`i` are the L2a bounding boxes, *TT*{sup}`j` is the footprint of the current L2b DGG tile and *A* operator computes the area of the corresponding polygon.

The aggregation at pixel map level of the various L2a disturbance maps is done by examining the consistency of the time series of detections. First, the series of detection is temporally sorted, both ascending and descending orbits:

$$
{{FD}_{series}}^{\mathrm{sorted}}(t)=\left\{ {{FD}_{map}}^{\mathrm{t}^1},...,{{FD}_{map}}^{\mathrm{t}^{\mathrm{i}}}\ \right\}
\tag{4.2}
$$ (eq:atbd-fd-4-2)

$\left\{ \mathrm{t}^1<...<\mathrm{t}^{\mathrm{i}}\ \right\}$ being the different times at which each product is acquired in a global cycle. Then the derivative of the sorted time series is computed with respect to time and thresholded

$$
{{FD}_{series}}^{\mathrm{steps}}=\left\vert \frac{d\left[ {{FD}_{series}}^{\mathrm{sorted}}(t) \right]}{dt} \right\vert>0
\tag{4.3}
$$ (eq:atbd-fd-4-3)

Finally, the number of transitions in the derivative vector is counted

$$
{{FD}_{steps}}^{\mathrm{count}}=\sum_{\mathrm{i}}^{}{{{FD}_{series}}^{\mathrm{steps}}(\mathrm{i})}
\tag{4.4}
$$ (eq:atbd-fd-4-4)

If the sum is odd, detection is assumed, and the resulting probability of change is the average of the different probabilities pixel-wise for detection only. Otherwise, the resulting probability of change and FD map is set to zero. If the sum is zero, detection is assumed only if all the acquisitions show detection. The rationale is that change is assumed only if all the acquisitions detecting it are simultaneously consistent. The situation is depicted in {fig}`16` for the example of three acquisitions and the corresponding truth table is reported in {ref}`Tab.1 <tab:atbd-fd-1>`.

(fig:atbd-fd-16)=
:::{div} atbd-mermaid-figure atbd-drawn-figure

<div class="atbd-figure-content">

![Time of validity for each acquisition among two global cycles, with overlaps shown in RGB colours. The FD relative to various acquisition times within the global cycle are reported. See also Tab.1.](../../_static/figures/atbd-fd/fig16.*)

<!-- source image: figure_16_section_4.1.4 sha256 748207b024b04e295722ec407134e3d026e373231d7730666174025f238ac690 -->

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 16</span><span class="caption-text"> — Time of validity for each acquisition among two global cycles, with overlaps shown in RGB colours. The FD relative to various acquisition times within the global cycle are reported. See also <a class="reference internal" href="04-change-detection-aggregation.html#tab-atbd-fd-1">Tab.1</a>.</span></p>

:::

(tab:atbd-fd-1)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 1</span><span class="caption-text"> — Truth table for the example of three FDs. The color coding for FD t1/3 is the same of Fig.16, where black is the absence of validity.</span></p>

| FD t1 | FD t2 | FD t3 | FD steps count | FD map aggregated | HM: <br> contributing | HM: <br> agreeing |
|---|---|---|---|---|---|---|
| <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | 0 | 0 | 3 | 3 |
| <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | <span class="atbd-cell-fill" style="background-color:#0000FF;color:#FFFFFF">1</span> | 1 | 1 | 3 | 1 |
| <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">1</span> | <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | 2 | 0 | 3 | 2 |
| <span class="atbd-cell-fill" style="background-color:#FF0000;color:#FFFFFF">1</span> | <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | 1 | 1 | 3 | 1 |
| <span class="atbd-cell-fill" style="background-color:#FFFF00;color:#000000">1</span> | <span class="atbd-cell-fill" style="background-color:#FFFF00;color:#000000">1</span> | <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | 1 | 1 | 3 | 2 |
| <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">1</span> | <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">1</span> | 2 | 0 | 3 | 1 |
| <span class="atbd-cell-fill" style="background-color:#000000;color:#FFFFFF">0</span> | <span class="atbd-cell-fill" style="background-color:#00FFFF;color:#000000">1</span> | <span class="atbd-cell-fill" style="background-color:#00FFFF;color:#000000">1</span> | 1 | 1 | 3 | 2 |
| 1 | 1 | 1 | 0 | 1 | 3 | 3 |

:::

A heat map *HM* is also stored representing at pixel level:

- the total number of acquisitions contributing to the decision, as reported in {ref}`Tab.1 <tab:atbd-fd-1>` – column 6
- the number of acquisitions agreeing to the decision, as reported in {ref}`Tab.1 <tab:atbd-fd-1>` – column 7

From the disturbance map combined, the extraction of the forest mask is obtained by a simple update to the initial forest mask.

$$
{CFM}^{aggregated}=\left\{ 1-{{FD}_{map}}^{aggregated} \right\}\land\left\{ \cup_{\mathrm{t}^{\mathrm{i}}}{CFM}^{\mathrm{t}^{\mathrm{i}}} \right\}
\tag{4.5}
$$ (eq:atbd-fd-4-5)

$\left\{ \mathrm{t}^1<...<\mathrm{t}^{\mathrm{i}}\ \right\}$ being the different times at which each product is acquired in a global cycle, where $\cup$ is the union operator.
