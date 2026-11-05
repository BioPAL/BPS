<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:atbd-l1-sar-9)=
# 9. Appendix: L1 Framing Tool

The L1 processing will be done starting always from a single L0 slice. This slice can be as long as the nominal one (i.e., 107.01580774s), shorter (partial slice) or longer (merged slice), up to the entire datatake. The L1 Processor will be able to manage all these cases, even processing just a sub-portion of or the whole input L0 slice.

In output, one L1 frame will be generated. The framing parameters (start and stop times) will be computed by a dedicated pre-processing tool and provided in input to the L1 Processor (via Job Order).

This tool computes the framing parameters basing on a regular angular grid. This grid is defined “cutting” the orbit in equal portions of the angle given by ANX position / Earth centre / satellite position (OPS angle). This approach reflects the one used for commanding the spacecraft. It allows to be robust against orbital period changes due to various “non-idealities” (orbit variations, manoeuvres, …) and to have a single grid for all the 3 BIOMASS swaths and for the 2 mission phases (TOM and INT). On the other hand, this will imply that frame length will be no more fixed but depending on position. In general, it is noted that frames having a non-nominal length (i.e., 21.00316155s, overlap 2s) for the mentioned reasons are considered accepted.
