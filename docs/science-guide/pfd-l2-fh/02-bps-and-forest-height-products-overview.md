<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:pfd-fh-2)=
# 2. BPS and Forest Height Products Overview

BIOMASS is a fully polarimetric SAR mission with its radar instrument relying on a single imaging mode, being Stripmap (SM), declined into 3 different swaths (S1, S2, S3) to achieve the global coverage.

The Forest Height product family is composed by:

- Level-2a (L2a): product composed by two stack-based images representing top canopy height and its quality layer, obtained from L1c processing and auxiliary data and with a different resolution and projection (latitude-longitude) with respect to the L1c source data. Same algorithms are envisaged to generate the same product in INT and TOM phases;
- Level-2b (L2b): top canopy height and quality layer images, on map and tile based, derived from merging of L2a in a given Global cycle, on the tile;

The Forest Height product family is generated during routine phase by the BIOMASS Processors (i.e., L2a Processor for L2a products, L2b Processor for L2b products running in the operational processing environment. The L2b FH processing has optional internal dependency on FD L2b .L2 products are internal products generated only for the BIOMASS scientific team.

The L2b BIOMASS products are geocoded to a specific tile based geographic DGG {rd}`9`, while the L2a products are geocoded in a latitude longitude projection with the same origin and sampling of the DGG, but with an extension of the current L2a product ground coverage.

```{note}
Provided that FD algorithm proper validation will only be possible with BIOMASS in orbit, the strategy proposed to deal with FH/AGB dependency on FD is to initially input to FH/AGB a Forest/Non-Forest mask generated outside of BPS {ad}`6` {ad}`7` {ad}`8` {ad}`9`. After testing FD performance with BIOMASS in orbit, FD dependency can be switched on again, provided it has a satisfactory performance.
```
