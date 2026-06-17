<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Applicable documents

The table below lists every official document applicable to **BIOMASS BPS v4.4.4**.
Each entry links to the authoritative PDF hosted on the ESA dissemination portal
[biomass-disc.info](https://www.biomass-disc.info/release_note).

When a new BPS version is released, this page is updated as the single source of
truth. Per-processor pages in the [Science Guide](../science-guide/index.md)
and the [User Guide](../user-guide/index.md) link back here.

## Project-level

| Document | Reference | Version | Date | Download |
|---|---|---|---|---|
| BIOMASS Processing Suite Release Note | `BIO-BPS-RN-ARE-010556` | 4.4.4 | 2026-05-15 | [PDF](https://www.biomass-disc.info/api/user-manager/v1/files/media/share/docs_BPS_v4_4_4/BPS_RN_v4_4_4.pdf) |
| BIOMASS Processing Suite Software User Manual (SUM) | `BIO-BPS-SUM-ARE-010479` | 4.4.1 | 2025-03-13 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_SUM_v4_4_1.pdf) |

## Level 1: SAR and Stack

| Document | Reference | Version | Date | Download |
|---|---|---|---|---|
| L1 a/b/c Products Format Specification | `BIO-BPS-L1-PFD-ARE-010076` | 1.6.1 | 2026-04-02 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_L1_PFD_v1_6_1.pdf) |
| L1 SAR Product ATBD | `BIO-BPS-L1-SAR-ATBD-ARE-010165` | 1.2.4 | 2026-03-27 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_L1_SAR_ATBD_v1_2_4.pdf) |
| L1c Stack Product ATBD | `BIO-BPS-L1-STACK-ATBD-ARE-010166` | 1.4.0 | 2026-04-02 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_L1_STACK_ATBD_v1_4_0.pdf) |

## Level 2b: Above-Ground Biomass (AGB)

| Document | Reference | Version | Date | Download |
|---|---|---|---|---|
| AGB Products Format Specification | `BIO-BPS-AGB-PFD-ARE-010257` | 3.4.0 | 2026-03-13 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_AGB_PFD_v3_4_0.pdf) |
| AGB Product ATBD | `BIO-BPS-AGB-ATBD-ARE-024912` | 3.1.4 | 2026-04-02 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_AGB_ATBD_v3_1_4.pdf) |

## Level 2b: Forest Height (FH)

| Document | Reference | Version | Date | Download |
|---|---|---|---|---|
| FH Products Format Specification | `BIO-BPS-FH-PFD-ARE-010256` | 3.4.0 | 2026-03-13 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_FH_PFD_v3_4_0.pdf) |
| FH Product ATBD | `BIO-BPS-FH-ATBD-ARE-10343` | 2.2.0 | 2026-03-13 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_FH_ATBD_v2_2_0.pdf) |

## Level 2b: Forest Disturbance (FD)

| Document | Reference | Version | Date | Download |
|---|---|---|---|---|
| FD Products Format Specification | `BIO-BPS-FD-PFD-ARE-010258` | 3.4.0 | 2026-03-13 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_FD_PFD_v3_4_0.pdf) |
| FD Product ATBD | `BIO-BPS-FD-ATBD-ARE-10344` | 2.1.8 | 2025-04-30 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_FD_ATBD_v2.1.8.pdf) |

## Interfaces and auxiliary

| Document | Reference | Version | Date | Download |
|---|---|---|---|---|
| Processing Interface Control Document (ICD) | `BIO-BPS-ICD-ARE-010113` | 3.2.3 | 2025-09-29 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_ICD_v3_2_3.pdf) |
| Processing Input & Output Data Definition (IODD) | `BIO-BPS-IODD-ARE-010112` | 3.1.2 | 2025-09-29 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_IODD_v3_1_2.pdf) |
| BPS Auxiliary Products Format | `BIO-BPS-AUX-FMT-ARE-010163` | 3.6.1 | 2026-04-02 | [PDF](https://www.biomass-disc.info/assets/documents/BPS_v4.4.2/BPS_AUX_FMT_v3_6_1.pdf) |

---

## Notes

- The **Release Note v4.4.4** lives on a dedicated path (`docs_BPS_v4_4_4/`).
  The 13 other documents are hosted under `/assets/documents/BPS_v4.4.2/`,
  the directory used as the active set during the 4.4.2 → 4.4.4 transition.
  They will be relocated to a `BPS_v4_4_4/` directory in a future portal
  refresh; this table will be updated accordingly.
- **Markdown conversion** of the ATBDs and SUM is tracked progressively in
  separate `[docs]` issues on the [issue tracker](https://github.com/BioPAL/BPS/issues).
  Until each conversion is merged, the PDF linked here remains the authoritative
  reference.
