<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:atbd-l1-sar-8)=
# 8. Open Points

The following table summarizes currently still-open points impacting algorithms definition and their expected date of closure (if already foreseen).

(tab:atbd-l1-sar-3)=
:::{div} atbd-table

<p class="atbd-table-caption"><span class="caption-number">Table 3</span><span class="caption-text"> — List of open points.</span></p>

| Impacted processing step | Description | Plan for closure | Risk assessment |
|---|---|---|---|
| Internal Calibration Computation: Instrument Replica | In {ad}`3` it is not mentioned whether all PGP products within the data take should be used to derive the replica or whether the replica should be generated from the PGP of each calibration sequence and then be applied and updated as a function of azimuth time by the processor. Therefore, in the following the replica will be obtained by averaging all PGPs within the data take corresponding to the same chirp parameters, i.e., in the case of a PRF change within a data take two different replicas will be derived. | Update issue of {ad}`3` | Low |
| Internal Calibration Computation: Instrument Internal Channel Imbalance | Since nothing is mentioned in {ad}`3`, it will be assumed that one single channel imbalance value per data take should be derived. Consequently, the different calibration pulses will be averaged to improve the estimation. It is also assumed that the channel imbalance is the same independent of the pulse type used. | Update issue of {ad}`3` | Low |
| Internal Calibration Computation and Correction | {ad}`4` is applicable for internal calibration algorithms. Minor open point: BPS normalizes the replica (as requested by ESA) instead of computing a scaling factor (as described by ADS). In addition, the estimation/compensation of drifts is not described in the technical memorandum. If the replicas are used without energy normalization and the drifts are also applied, the channel imbalance will be corrected in practice twice, which should be avoided. The current implementation makes that. | Action opened during CDR1 | Low |

:::
