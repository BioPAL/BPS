<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:pfd-l1-2)=
# 2. BPS and L1 Products Overview

BIOMASS is a fully polarimetric SAR mission with its radar instrument relying on a single imaging mode, being Stripmap (SM), declined into 3 different swaths (S1, S2, S3) to achieve the global coverage.

The Level-1 product family is composed by:

- Level-1a (L1a): Single-look Complex Slant (SCS) data, consisting of SAR focused complex data lying in the Zero-Doppler azimuth plane and in slant range geometric projection with all the necessary radiometric and geometric corrections;
- Level-1b (L1b): Detected Ground Multi-look (DGM) data, consisting of SAR focused detected data lying in the Zero-Doppler azimuth plane and in ground range geometric projection with all the necessary radiometric and geometric corrections;
- Level-1c (L1c): stack interferometric product (STA), consisting of co-registered and phase calibrated SCS products based for both TOM or INT sequences from the same swath and major cycle acquisitions.

The L1 product family is generated during routine phase by the BIOMASS Processors (i.e., L1 Processor for L1a and L1b products, STA Processor for L1c product) running in the operational processing environment which is part of the PDGS. For specific calibration purposes and for products acquired over transponder (PARC), it is possible to manually activate the generation of four additional L1a (SCS) products, indicated with the acronym SCS-CAL, each one keeping into account and compensating the delay introduced by the transponder for one of its four scattering responses (i.e., Gt1, Gt2, X and Y).

These products serve the BIOMASS primary and secondary objectives. The BIOMASS products and associated metadata definition is such that it allows the generation of higher-level products for known and unknown applications but also to support the verification and monitoring activities during the commissioning and routine phases.

All the L1 BIOMASS products are belonging to the following product category:

- Level-1 Standard products (L1S), that is the “complete” focused data. In other words, it contains all the data sets being measurement, annotation and representation.

while the L1a and L1c (i.e., all except L1b) belong also to another product category, being:

- Level-1 Monitoring product (L1M), that is the “light” product for monitoring purpose, that is derived from the L1S by removing the images component of the measurement data set.

BIOMASS acquires data-takes as long segment of several minutes. For practical reasons related to data volume all the L1 products are chopped to generate subsets of prefixed size, defined as frames, and lying on the L1 framing grid. Such chopping is the default production rule for the generated BIOMASS L1 products. However, there will be cases where the framing is not required and the L1 might be floating (i.e., not aligned to the L1 framing grid) and the framing is disabled, generating products long as the whole Data Take. The proposed L1 organisation is valid independent on whether or not the products’ chopping into frames is applied or not.

In {sec}`3` an overview of L1a/b products format is provided, while its detailed description is included in {sec}`4`. The same will be done for L1c products respectively in {sec}`5` and 6.
