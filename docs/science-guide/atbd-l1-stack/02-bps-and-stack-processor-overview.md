<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:atbd-l1-stack-2)=
# 2. BPS and Stack Processor Overview

(sec:atbd-l1-stack-2-1)=
### 2.1 BPS Overview

The BIOMASS Processing Suite (BPS) is in charge of processing BIOMASS Level-0 data up to Level-3, generating a wide set of products. Product generation scheme is depicted in the following two figures respectively for TOM phase and for INT phase. The processing foresees the following steps:

- **L1 processing**: the BIOMASS image formation step, taking in input 1 L0 slice and generating in output $N\geq1$ L1 frames;
- **Stack Processing**: the coregistration and stacking processing step that will define the required 7-images Tomographic stacks or the 3-images Interferometric stacks. It includes the stack calibration phase;
- **L2a Processing**: processing step for each stack that provides the inputs to the further products Tile Generation performed by the L2b processors. For one global cycle the number of stacks varies mainly with latitude, approximately from a minimum of two at the Equator (ascending/descending) to about six at higher latitudes (three ascending/descending pairs);
- **L2b Processing**: processing that aggregates all the stack-based L2a products insisting on the same location on ground (Tile) in the same time interval and generates a single BIOMASS L2b product. A different number of stacks may be available for each DGG tile, based on tile dimensions and swath coverage (which is not related to DGG tiling pattern);
- **L3 Processing**: consolidation processing removing discontinuities in L2b products.

(fig:atbd-l1-stack-1)=
::::{div} atbd-mermaid-figure atbd-drawn-figure

<div class="atbd-figure-content">

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "theme": "default", "layout": "elk", "elk": {"nodePlacementStrategy": "NETWORK_SIMPLEX"}, "flowchart": {"useMaxWidth": false, "diagramPadding": 4, "subGraphTitleMargin": {"top": 2, "bottom": 2}}}}%%
flowchart TB
  accTitle: Biomass high-level processing scheme from L0 data to L3 products, TOM phase.
  accDescr: Three stages, L1, L2 and L3 processing, in four lanes drawn side by side from left to right in the original and here from top to bottom. L1 Frame-based Processing (Data Driven): each L0 frame (G_i, M_j, m_k, A_l, S_m) up to A_l+6 goes through an L1 Pr. step, giving L1 (G_i, M_j, m_k, A_l, F_m) to A_l+6. L2a Stack-based Processing (Data Driven): the L1 frames are stacked; the stack, L1c(G_i, M_j, m_k, A_l, F_m) to A_l+6, feeds the FH L2a proc and AGB L2a proc steps of the L2a processor. L2b Tile-based Processing (Asynchronous): FH_a(G_i, M_j, m_k, F_m) and FH_a(G_i, *Tn, *Tn, *Tn) feed the FH L2b Tile proc, which also takes the internal forest class/mask; AGB_a(G_i, M_j, m_k, F_m), AGB_b(t_a, t_b, neighbour Tiles), AGB_a(G_i, *Tn, *Tn, *Tn) and the forest class/mask with FH_b(t_a, t_b, T_n) feed the AGB L2b Tile proc. L3 Basin-based Processing (Asynchronous): FH_b(t_a, t_b, T_n) and AGB_b(t_a, t_b, T_n) go to the FH L3 and AGB L3 consolidation steps of the L3 processor, which output FH_3(t_a, t_b, B_n) and AGB_3(t_a, t_b, B_n).

  %% Colours of the original figure. Each class sets its text colour too, so it
  %% stays readable with the light and the dark site themes.
  %%   step:  processing step (blue box, white text)
  %%   proc:  step inside a processor (grey box, white text)
  %%   data:  input or output label (text only, on the lane colour)
  %%   red A: the acquisition index that changes from frame to frame
  classDef step fill:#8faadc,stroke:#1f1f1f,color:#ffffff,font-weight:bold
  classDef proc fill:#a6a6a6,stroke:#1f1f1f,color:#ffffff,font-weight:bold
  classDef data fill:none,stroke:none,color:#000000
  classDef dots fill:none,stroke:none,color:#000000,font-weight:bold

  subgraph L1["<b>L1 processing</b><br/>L1 Frame-based Processing<br/><span style='color:#548235'>Data Driven</span>"]
    in1["L0 (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l</sub></b>, S<sub>m</sub>)"]:::data
    l1a["L1 Pr."]:::step
    more["⋮"]:::dots
    in2["L0 (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l+ 6</sub></b>, S<sub>m</sub>)"]:::data
    l1b["L1 Pr."]:::step
  end

  subgraph L2P["<b>L2 processing</b>"]
    subgraph L2A["L2a Stack-based Processing<br/><span style='color:#548235'>Data Driven</span>"]
      stack["Stack"]:::step
      subgraph L2AP["L2a processor"]
        fha["FH L2a<br/>proc"]:::proc
        agba["AGB<br/>L2a<br/>proc"]:::proc
      end
    end
    subgraph L2B["L2b Tile-based Processing<br/><span style='color:#e00000'>Asynchronous</span>"]
      mask["internal forest<br/>class/mask"]:::data
      fhin["FH<sub>a</sub>(G<sub>i</sub>, *<sub>Tn</sub>, *<sub>Tn</sub>, *<sub>Tn</sub>)"]:::data
      agbinb["AGB<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, <i>neighbour Tiles</i>)"]:::data
      agbina["AGB<sub>a</sub>(G<sub>i</sub>,*<sub>Tn</sub>, *<sub>Tn</sub>, *<sub>Tn</sub>)"]:::data
      fhb["FH L2b<br/>Tile<br/>proc"]:::step
      agbb["AGB<br/>L2b Tile<br/>proc"]:::step
    end
  end

  subgraph L3["<b>L3 processing</b><br/>L3 Basin-based Processing<br/><span style='color:#e00000'>Asynchronous</span>"]
    subgraph L3PR["L3<br/>processor"]
      fh3["FH L3<br/><small>consolidation</small>"]:::proc
      agb3["AGB L3<br/><small>consolidation</small>"]:::proc
    end
    fhout["FH<sub>3</sub>(t<sub>a</sub>, t<sub>b</sub>, B<sub>n</sub>)"]:::data
    agbout["AGB<sub>3</sub>(t<sub>a</sub>, t<sub>b</sub>, B<sub>n</sub>)"]:::data
  end

  in1 --> l1a
  in2 --> l1b
  in1 ~~~ more
  l1a -- "L1 (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l</sub></b>, F<sub>m</sub>)" --> stack
  l1b -- "L1 (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l+ 6</sub></b>, F<sub>m</sub>)" --> stack
  stack -- "L1c(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l</sub></b>, F<sub>m</sub>)<br/>L1c (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l+1</sub></b>, F<sub>m</sub>)<br/>…<br/>L1c(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l+ 6</sub></b>, F<sub>m</sub>)" --> fha
  stack --> agba
  fha -- "FH<sub>a</sub>(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, F<sub>m</sub>)" --> fhb
  agba -- "AGB<sub>a</sub>(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, F<sub>m</sub>)" --> agbb
  fhin --> fhb
  agbinb --> agbb
  mask ==> fhb
  mask -- "forest<br/>class/mask<br/>FH<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)" --> agbb
  fhb -- "FH<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)" --> fh3
  agbb -- "AGB<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)" --> agb3
  fh3 --> fhout
  agb3 --> agbout
  agbina --> agbb

  %% lane colours (fill and text colour: readable in both site themes)
  style L2P fill:none,stroke:none,color:#000000
  style L1 fill:#dae3f3,stroke:#dae3f3,color:#000000
  style L2A fill:#fbe5d6,stroke:#fbe5d6,color:#000000
  style L2B fill:#e2f0d9,stroke:#e2f0d9,color:#000000
  style L3 fill:#fff2cc,stroke:#fff2cc,color:#000000
  style L2AP fill:#8faadc,stroke:#1f1f1f,color:#ffffff
  style L3PR fill:#8faadc,stroke:#1f1f1f,color:#ffffff
  %% links 11 and 12: internal forest class/mask (green, then orange)
  linkStyle 11 stroke:#70ad47,stroke-width:3px
  linkStyle 12 stroke:#ed7d31,stroke-width:2px
```

<!-- source image: figure_1_section_2.1 sha256 33a97074fa7afe25c1d16c43cf0fa648cb1ff41c2b4b2e91559f98ef5446bb8f -->

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 1</span><span class="caption-text"> — Biomass high-level processing scheme from L0 data to L3 products, TOM phase.</span></p>

::::

(fig:atbd-l1-stack-2)=
::::{div} atbd-mermaid-figure atbd-drawn-figure

<div class="atbd-figure-content">

<p class="atbd-subfigure-label">(a)</p>

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "theme": "default", "layout": "elk", "elk": {"nodePlacementStrategy": "NETWORK_SIMPLEX"}, "flowchart": {"useMaxWidth": false, "diagramPadding": 4, "wrappingWidth": 320}}}%%
flowchart TB
  accTitle: Biomass high-level processing scheme from L0 data to L3 products, INT phase.
  accDescr: Overview of the figure, whose detail is in four sub-figures, one per lane; each lane names the sub-figure that details it. L0 frames enter the L1 Frame-based Processing (Data Driven), the L1 processing; its L1 frames go to the L2a Stack-based Processing (Data Driven), the first lane of the L2 processing, which gives the FD_a, FH_a and AGB_a products to the L2b Tile-based Processing (Asynchronous), its second lane. The L2b products FD_b, FH_b and AGB_b go to the L3 Basin-based Processing (Asynchronous), the L3 processing, which outputs FD_3, FH_3 and AGB_3.

  %% Colours of the original figure. Each class sets its text colour too, so it
  %% stays readable with the light and the dark site themes.
  %%   data:  input or output label (text only)
  %%   l1, l2a, l2b, l3: one lane of the original, in its lane colour, under its
  %%                     stage title (L1, L2 or L3 processing); its last line, in
  %%                     italics, names the sub-figure that details it: (b) to (e)
  classDef data fill:none,stroke:none,color:#000000
  classDef l1 fill:#dae3f3,stroke:#8faadc,color:#000000
  classDef l2a fill:#fbe5d6,stroke:#f4b183,color:#000000
  classDef l2b fill:#e2f0d9,stroke:#a9d18e,color:#000000
  classDef l3 fill:#fff2cc,stroke:#ffd966,color:#000000

  l0["L0"]:::data

  lane1["<b>L1 processing</b><br/>L1 Frame-based Processing<br/><b style='color:#548235'>Data Driven</b><br/><i>(b)</i>"]:::l1

  %% The original has one L2 processing stage holding two lanes
  subgraph L2["&nbsp;"]
    direction TB
    l2title["<b>L2 processing</b>"]:::data
    lane2a["L2a Stack-based Processing<br/><b style='color:#548235'>Data Driven</b><br/><i>(c)</i>"]:::l2a
    lane2b["L2b Tile-based Processing<br/><b style='color:#e00000'>Asynchronous</b><br/><i>(d)</i>"]:::l2b
  end

  lane3["<b>L3 processing</b><br/>L3 Basin-based Processing<br/><b style='color:#e00000'>Asynchronous</b><br/><i>(e)</i>"]:::l3

  out["FD<sub>3</sub>, FH<sub>3</sub>, AGB<sub>3</sub>"]:::data

  l0 --> lane1
  lane1 --> lane2a
  lane2a --> lane2b
  lane2b --> lane3
  lane3 --> out
  l2title ~~~ lane2a

  %% L2 processing stage: white frame, thin grey outline
  style L2 fill:#ffffff,stroke:#7f7f7f,color:#000000
```

<p class="atbd-subfigure-label">(b)</p>

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "theme": "default", "layout": "elk", "flowchart": {"useMaxWidth": false, "diagramPadding": 4}}}%%
flowchart LR
  accTitle: Biomass high-level processing scheme from L0 data to L3 products, INT phase.
  accDescr: Detail of the L1 Frame-based Processing (Data Driven) lane, in L1 processing. The L0 frames L0 (G_i, M_j, m_k, A_l, S_m), L0 (G_i, M_j, m_k, A_l+1, S_m) and L0 (G_i, M_j, m_k, A_l+2, S_m) each go through an L1 Pr. step, which gives L1 (G_i, M_j, m_k, A_l, F_m), L1 (G_i, M_j, m_k, A_l+1, F_m) and L1 (G_i, M_j, m_k, A_l+2, F_m). The three L1 frames go to the Stack step of sub-figure (c).

  %% Colours of the original figure. Each class sets its text colour too, so it
  %% stays readable with the light and the dark site themes.
  %%   step: processing step (blue box, white text)
  %%   data: input label (text only, on the lane colour)
  %%   ref:  cross-reference, NOT a step (white stadium, dashed outline): it points to a
  %%         step drawn in another sub-figure, so that no step is duplicated
  %%   red A: the acquisition index, which changes from one frame to the next
  classDef step fill:#8faadc,stroke:#1f1f1f,color:#ffffff,font-weight:bold
  classDef data fill:none,stroke:none,color:#000000
  classDef ref fill:#ffffff,stroke:#7f7f7f,color:#000000,stroke-dasharray:4 3

  subgraph L1["<b>L1 processing</b><br/>L1 Frame-based Processing<br/><span style='color:#548235'>Data Driven</span>"]
    direction LR
    in1["L0 (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l</sub></b>, S<sub>m</sub>)"]:::data
    in2["L0 (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l+1</sub></b>, S<sub>m</sub>)"]:::data
    in3["L0 (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l+2</sub></b>, S<sub>m</sub>)"]:::data
    l1a["L1 Pr."]:::step
    l1b["L1 Pr."]:::step
    l1c["L1 Pr."]:::step
  end

  toStack(["to (c): Stack"]):::ref

  in1 --> l1a
  in2 --> l1b
  in3 --> l1c
  l1a -- "L1 (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l</sub></b>, F<sub>m</sub>)" --> toStack
  l1b -- "L1 (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l+1</sub></b>, F<sub>m</sub>)" --> toStack
  l1c -- "L1 (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l+2</sub></b>, F<sub>m</sub>)" --> toStack

  %% lane colour of the original
  style L1 fill:#dae3f3,stroke:#dae3f3,color:#000000
```

<p class="atbd-subfigure-label">(c)</p>

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "theme": "default", "layout": "elk", "flowchart": {"useMaxWidth": false, "diagramPadding": 4}}}%%
flowchart LR
  accTitle: Biomass high-level processing scheme from L0 data to L3 products, INT phase.
  accDescr: Detail of the L2a Stack-based Processing (Data Driven) lane, in L2 processing. The three L1 frames from the L1 Pr. steps of sub-figure (b) are stacked. The Stack step gives L1c(G_i, M_j, m_k, A_l, F_m), L1c (G_i, M_j, m_k, A_l+1, F_m) and L1c(G_i, M_j, m_k, A_l+2, F_m) to the FD L2a proc, FH L2a proc and AGB L2a proc steps of the L2a processor. FD L2a proc also takes FD_a(G_i-1, M_j, m_k, F_m). The three steps give FD_a(G_i, M_j, m_k, F_m), FH_a(G_i, M_j, m_k, F_m) and AGB_a(G_i, M_j, m_k, F_m) to the FD, FH and AGB L2b Tile proc steps of sub-figure (d).

  %% Colours of the original figure. Each class sets its text colour too, so it
  %% stays readable with the light and the dark site themes.
  %%   step: processing step (blue box, white text)
  %%   proc: step inside the L2a processor (grey box, white text)
  %%   data: input label (text only, on the lane colour)
  %%   ref:  cross-reference, NOT a step (white stadium, dashed outline): it points to a
  %%         step drawn in another sub-figure, so that no step is duplicated
  %%   red A, red G: the index that changes from one acquisition or cycle to the next
  classDef step fill:#8faadc,stroke:#1f1f1f,color:#ffffff,font-weight:bold
  classDef proc fill:#a6a6a6,stroke:#1f1f1f,color:#ffffff,font-weight:bold
  classDef data fill:none,stroke:none,color:#000000
  classDef ref fill:#ffffff,stroke:#7f7f7f,color:#000000,stroke-dasharray:4 3

  fromL1(["from (b):<br/>L1 Pr."]):::ref

  subgraph L2A["<b>L2 processing</b><br/>L2a Stack-based Processing<br/><span style='color:#548235'>Data Driven</span>"]
    direction LR
    stack["Stack"]:::step
    fdprev["FD<sub>a</sub>(<b style='color:#e00000'>G<sub>i-1</sub></b>, M<sub>j</sub>, m<sub>k</sub>, F<sub>m</sub>)"]:::data
    subgraph L2AP["L2a processor"]
      direction TB
      fda["FD L2a<br/>proc"]:::proc
      fha["FH L2a<br/>proc"]:::proc
      agba["AGB<br/>L2a<br/>proc"]:::proc
    end
  end

  toFD(["to (d): FD<br/>L2b Tile proc"]):::ref
  toFH(["to (d): FH<br/>L2b Tile proc"]):::ref
  toAGB(["to (d): AGB<br/>L2b Tile proc"]):::ref

  fromL1 --> stack
  stack -- "L1c(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l</sub></b>, F<sub>m</sub>)<br/>L1c (G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l+1</sub></b>, F<sub>m</sub>)<br/>L1c(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, <b style='color:#e00000'>A<sub>l+2</sub></b>, F<sub>m</sub>)" --> fda
  stack --> fha
  stack --> agba
  fdprev --> fda
  fda -- "FD<sub>a</sub>(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, F<sub>m</sub>)" --> toFD
  fha -- "FH<sub>a</sub>(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, F<sub>m</sub>)" --> toFH
  agba -- "AGB<sub>a</sub>(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, F<sub>m</sub>)" --> toAGB

  %% lane colour and L2a processor frame of the original
  style L2A fill:#fbe5d6,stroke:#fbe5d6,color:#000000
  style L2AP fill:#8faadc,stroke:#1f1f1f,color:#ffffff
```

<p class="atbd-subfigure-label">(d)</p>

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "theme": "default", "layout": "elk", "elk": {"considerModelOrder": "NODES_AND_EDGES", "forceNodeModelOrder": true, "nodePlacementStrategy": "LINEAR_SEGMENTS"}, "flowchart": {"useMaxWidth": false, "diagramPadding": 4, "wrappingWidth": 320}}}%%
flowchart TB
  accTitle: Biomass high-level processing scheme from L0 data to L3 products, INT phase.
  accDescr: Detail of the L2b Tile-based Processing (Asynchronous) lane, in L2 processing. The FD L2b Tile proc takes FD_a(G_i, M_j, m_k, F_m) from sub-figure (c) and FD_a(G_i, *Tn, *Tn, *Tn); its FD_b(t_a, t_b, T_n) goes to the FH L2b Tile proc (green arrow) and, with FH_b(t_a, t_b, T_n), to the AGB L2b Tile proc (orange arrow). The FH L2b Tile proc also takes FH_a(G_i, M_j, m_k, F_m) from sub-figure (c) and FH_a(G_i, *Tn, *Tn, *Tn). The AGB L2b Tile proc also takes AGB_a(G_i, M_j, m_k, F_m) from sub-figure (c), AGB_b(t_a, t_b, neighbour Tiles) and AGB_a(G_i, *Tn, *Tn, *Tn). The three steps give FD_b, FH_b and AGB_b(t_a, t_b, T_n) to the FD, FH and AGB L3 consolidation steps of sub-figure (e).

  %% Colours of the original figure. Each class sets its text colour too, so it
  %% stays readable with the light and the dark site themes.
  %%   step: processing step (blue box, white text)
  %%   data: input label (text only, on the lane colour)
  %%   ref:  cross-reference, NOT a step (white stadium, dashed outline): it points to a
  %%         step drawn in another sub-figure, so that no step is duplicated
  classDef step fill:#8faadc,stroke:#1f1f1f,color:#ffffff,font-weight:bold
  classDef data fill:none,stroke:none,color:#000000
  classDef ref fill:#ffffff,stroke:#7f7f7f,color:#000000,stroke-dasharray:4 3

  fromFD(["from (c): FD L2a proc"]):::ref
  fromFH(["from (c): FH L2a proc"]):::ref
  fromAGB(["from (c): AGB L2a proc"]):::ref

  subgraph L2B["<b>L2 processing</b><br/>L2b Tile-based Processing<br/><span style='color:#e00000'>Asynchronous</span>"]
    direction TB
    fdin["FD<sub>a</sub>(G<sub>i</sub>, *<sub>Tn</sub>, *<sub>Tn</sub>, *<sub>Tn</sub>)"]:::data
    fhin["FH<sub>a</sub>(G<sub>i</sub>, *<sub>Tn</sub>, *<sub>Tn</sub>, *<sub>Tn</sub>)"]:::data
    agbinb["AGB<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, <i>neighbour Tiles</i>)"]:::data
    agbina["AGB<sub>a</sub>(G<sub>i</sub>,*<sub>Tn</sub>, *<sub>Tn</sub>, *<sub>Tn</sub>)"]:::data
    fdb["FD L2b<br/>Tile<br/>proc"]:::step
    fhb["FH L2b<br/>Tile<br/>proc"]:::step
    agbb["AGB<br/>L2b Tile<br/>proc"]:::step
  end

  toFD(["to (e): FD L3 consolidation"]):::ref
  toFH(["to (e): FH L3 consolidation"]):::ref
  toAGB(["to (e): AGB L3 consolidation"]):::ref

  fromFD -- "FD<sub>a</sub>(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, F<sub>m</sub>)" --> fdb
  fromFH -- "FH<sub>a</sub>(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, F<sub>m</sub>)" --> fhb
  fromAGB -- "AGB<sub>a</sub>(G<sub>i</sub>, M<sub>j</sub>, m<sub>k</sub>, F<sub>m</sub>)" --> agbb
  fdin --> fdb
  fhin --> fhb
  agbinb --> agbb
  agbina --> agbb
  fdb == "FD<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)" ==> fhb
  fdb -- "FD<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)<br/>FH<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)" --> agbb
  fdb -- "FD<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)" --> toFD
  fhb -- "FH<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)" --> toFH
  agbb -- "AGB<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)" --> toAGB

  %% lane colour of the original
  style L2B fill:#e2f0d9,stroke:#e2f0d9,color:#000000
  %% links 7 and 8: FD_b of the FD L2b Tile proc (green, then orange)
  linkStyle 7 stroke:#70ad47,stroke-width:3px
  linkStyle 8 stroke:#ed7d31,stroke-width:2px
```

<p class="atbd-subfigure-label">(e)</p>

```{mermaid}
%%{init: {"class": {"useMaxWidth": false}, "theme": "default", "layout": "elk", "flowchart": {"useMaxWidth": false, "diagramPadding": 4}}}%%
flowchart LR
  accTitle: Biomass high-level processing scheme from L0 data to L3 products, INT phase.
  accDescr: Detail of the L3 Basin-based Processing (Asynchronous) lane, in L3 processing. FD_b, FH_b and AGB_b(t_a, t_b, T_n), from the FD, FH and AGB L2b Tile proc steps of sub-figure (d), go to the FD L3, FH L3 and AGB L3 consolidation steps of the L3 processor, which output FD_3, FH_3 and AGB_3(t_a, t_b, B_n).

  %% Colours of the original figure. Each class sets its text colour too, so it
  %% stays readable with the light and the dark site themes.
  %%   proc: step inside the L3 processor (grey box, white text)
  %%   data: output label (text only, on the lane colour)
  %%   ref:  cross-reference, NOT a step (white stadium, dashed outline): it points to a
  %%         step drawn in another sub-figure, so that no step is duplicated
  classDef proc fill:#a6a6a6,stroke:#1f1f1f,color:#ffffff,font-weight:bold
  classDef data fill:none,stroke:none,color:#000000
  classDef ref fill:#ffffff,stroke:#7f7f7f,color:#000000,stroke-dasharray:4 3

  fromFD(["from (d): FD L2b Tile proc"]):::ref
  fromFH(["from (d): FH L2b Tile proc"]):::ref
  fromAGB(["from (d): AGB L2b Tile proc"]):::ref

  subgraph L3["<b>L3 processing</b><br/>L3 Basin-based Processing<br/><span style='color:#e00000'>Asynchronous</span>"]
    direction LR
    subgraph L3PR["L3<br/>processor"]
      direction TB
      fd3["FD L3<br/><small>consolidation</small>"]:::proc
      fh3["FH L3<br/><small>consolidation</small>"]:::proc
      agb3["AGB L3<br/><small>consolidation</small>"]:::proc
    end
    fdout["FD<sub>3</sub>(t<sub>a</sub>, t<sub>b</sub>, B<sub>n</sub>)"]:::data
    fhout["FH<sub>3</sub>(t<sub>a</sub>, t<sub>b</sub>, B<sub>n</sub>)"]:::data
    agbout["AGB<sub>3</sub>(t<sub>a</sub>, t<sub>b</sub>, B<sub>n</sub>)"]:::data
  end

  fromFD -- "FD<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)" --> fd3
  fromFH -- "FH<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)" --> fh3
  fromAGB -- "AGB<sub>b</sub>(t<sub>a</sub>, t<sub>b</sub>, T<sub>n</sub>)" --> agb3
  fd3 --> fdout
  fh3 --> fhout
  agb3 --> agbout

  %% lane colour and L3 processor frame of the original
  style L3 fill:#fff2cc,stroke:#fff2cc,color:#000000
  style L3PR fill:#8faadc,stroke:#1f1f1f,color:#ffffff
```

<!-- source image: figure_2_section_2.1 sha256 5de7fe36c9ccaa724a6a7263db7c09cdaaec4c489eafc6b7894604d6f2a1d7aa -->

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 2</span><span class="caption-text"> — Biomass high-level processing scheme from L0 data to L3 products, INT phase.</span></p>

::::

(sec:atbd-l1-stack-2-2)=
### 2.2 Overview

The Stack Processor has as its main goal the generation of stacks ready to be used by the L2 Processor. It receives as input a stack of BIOMASS Level-1a images (SCS stack), including its relevant annotated data (e.g., the ionosphere screen estimated exploring Faraday Rotation). The Stack Processor performs the coregistration of the stack, as well as its calibration:

- stacking, *i.e.*, the alignment of the individual images, which will be done via a two-step approach for coregistration;
- calibration, *i.e.*, the coregistered stack and input ionospheric phase screens delivered by the L1 Processor are fed to the calibration module which will use multi-baseline techniques to compute refined ionospheric screens and baseline errors. The stack is calibrated by removing the estimated errors. The block can also perform the Sum of Kronecker Products (SKP) calibration to deliver information required for the generation of tomography-ready stacks to the L2 Processor {rd}`3`.

Other operations, like the generation of Digital Elevation Model (DEM) and Digital Terrain Model (DTM) are outside the scope of BPS.

(sec:atbd-l1-stack-2-3)=
### 2.3 Stack Processor’s Workflow and Algorithmic Overview

The main steps of the BPS Stack Processor workflow are depicted in the following figure and described in detail in the following sub-sections.

(fig:atbd-l1-stack-3)=
::::{div} atbd-mermaid-figure atbd-drawn-figure

<div class="atbd-figure-content">

```{mermaid}
%%{init: {"flowchart": {"useMaxWidth": false}, "class": {"useMaxWidth": false}, "theme": "default", "layout": "elk", "elk": {"nodePlacementStrategy": "NETWORK_SIMPLEX"}}}%%
flowchart LR
  accTitle: Stack Processor detailed workflow. The AUX_PPS is omitted from the figure since it is an input to each step of the Stack Processor.
  accDescr: The L1a Stack enters the Pre-Processing block, which contains cross-polarization combination, estimation of RFI degradation indices, reference image selection and geometrical quantity's upsampling. Its output goes to the Coregistration block, made of shift estimation from geometry, residual shift estimation from data, image shifting and LUT shifting. Its output goes to the Calibration block, made of azimuth spectral filtering (AZF), background ionosphere removal (IOB), multi-squint calibration (MSC) and SKP phase calibration (SKP). The FNF Mask feeds the SKP phase calibration. The Calibration block outputs the L1c Stack.

  %% Layout: left to right as the original. The ELK NETWORK_SIMPLEX placement keeps
  %% the FNF Mask arrow out of the Coregistration and Calibration frames. The steps of
  %% Pre-Processing and Coregistration are chained by invisible links (~~~) to keep their
  %% order; those of Calibration are not, because that frame has an incoming link on a
  %% step (FNF) and ELK then lays its chain out horizontally.

  %% Colours of the original figure. Each class sets its text colour too, so it
  %% stays readable with the light and the dark site themes.
  %%   product: data product (cylinder, white)
  %%   step:    processing step inside a block (light grey box)
  classDef product fill:#ffffff,stroke:#000000,color:#000000
  classDef step fill:#f2f2f2,stroke:#7f7f7f,color:#000000

  l1a[("L1a Stack")]:::product
  fnf[("FNF Mask")]:::product
  l1c[("L1c Stack")]:::product

  subgraph pre["<b>Pre-Processing</b>"]
    direction TB
    xpol["Cross-Polarization Combination"]:::step
    rfi["Estimation of RFI Degradation Indices"]:::step
    refsel["Reference Image Selection"]:::step
    upsamp["Geometrical Quantity's Upsampling"]:::step
    xpol ~~~ rfi ~~~ refsel ~~~ upsamp
  end

  subgraph coreg["<b>Coregistration</b>"]
    direction TB
    shiftgeo["Shift Estimation from Geometry"]:::step
    shiftres["Residual Shift Estimation from Data"]:::step
    imgshift["Image Shifting"]:::step
    lutshift["LUT shifting"]:::step
    shiftgeo ~~~ shiftres ~~~ imgshift ~~~ lutshift
  end

  subgraph cal["<b>Calibration</b>"]
    direction TB
    azf["Azimuth Spectral Filtering (AZF)"]:::step
    iob["Background Ionosphere Removal (IOB)"]:::step
    msc["Multi-Squint Calibration (MSC)"]:::step
    skp["SKP Phase Calibration (SKP)"]:::step
  end

  style pre fill:#ffffff,stroke:#000000,color:#000000
  style coreg fill:#ffffff,stroke:#000000,color:#000000
  style cal fill:#ffffff,stroke:#000000,color:#000000

  l1a --> pre
  pre --> coreg
  coreg --> cal
  fnf --> skp
  cal --> l1c
```

<!-- source image: figure_3_section_2.3 sha256 eae4c59c14c2446fd0742a432aac4c5d69c4c74c23f9d211a133091c3137cf0a -->

</div>

<p class="atbd-figure-caption"><span class="caption-number">Fig. 3</span><span class="caption-text"> — Stack Processor detailed workflow. The AUX_PPS is omitted from the figure since it is an input to each step of the Stack Processor.</span></p>

::::

(sec:atbd-l1-stack-2-3-1)=
#### 2.3.1 Pre-processing

The *Pre-processing* block prepares and packs the data needed by the next blocks (Stacking and Calibration). The main steps of the Pre-processor are the following:

- Computation of RFI coherence degradation: Computation of the RFI masking impact on the InSAR coherence.
- Polarization combination (if enabled): The cross-polarizations of the L1a product are possibly averaged (*cross-pol merging*) or selectively dropped (*i.e.* keep only HV or VH) in a single image, the associated LUTs are respectively merged or dropped.
- Coregistration primary image selection: Selection of the most appropriate image for being utilized as reference image for coregistration . For this, the Pre-processor will employ quality indicators of the L1 stack and/or coverage information.
- Upsampling of geometric quantities stored in the L1a input products’s LUTs to match the resolution of the SCS.

(sec:atbd-l1-stack-2-3-2)=
#### 2.3.2 Stacking

The main steps of the *Stacking* block are:

- Geometry-based estimation of coregistration shifts.
- Residual offset estimation: Further estimation of residual coregistration shifts that may be due to imprecise information on the satellite’s orbit, as well as ionospheric artefacts that affect the input data;
- Image and LUT interpolation: Coregistration shifts are applied to input data and LUTs to align them to the primary image grid.

(sec:atbd-l1-stack-2-3-3)=
#### 2.3.3 Calibration

The *Calibration* block consists of the following steps:

- Azimuth spectral filtering: Apply a common band filtering in azimuth direction to crop the data spectrum to the azimuth common bandwidth.
- Slow-varying ionosphere estimation: A Multi-baseline split-spectrum-based method that estimates and removes from data low-pass ionospheric artefacts
- Baseline and residual fast-varying ionosphere correction: A Multi-Squint-based (MS) approach that estimates and corrects baseline errors and residual high-pass ionospheric artefacts from the image.
- Phase calibration by SKP: A module for estimating and possibly removing ground steering phase-screens required to generate tomography-ready stacks {rd}`3`. By default, the SKP phases will be estimated but not applied. The SKP phase removal can be activated by configuration.
