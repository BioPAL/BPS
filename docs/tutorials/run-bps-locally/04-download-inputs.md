<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# 4. Download raw input products

## Find a RAW_0S product ID

Go to the [ESA MAAP Explorer](https://explorer.maap.eo.esa.int/), search
for a Biomass Level 0 product over your area of interest, and copy the
product ID. Example:

```
BIO_S1_RAW__0S_20251121T095253_20251121T095450_T_G01_M01_C01_T006_F062_01_DNG988
```

The product ID encodes useful information:

| Field | Example | Meaning |
|---|---|---|
| Swath | `S1` | Swath identifier |
| Start time | `20251121T095253` | Acquisition start (UTC) |
| Stop time | `20251121T095450` | Acquisition stop (UTC) |
| Global cycle | `G01` | Global coverage ID |
| Major cycle | `M01` | Major cycle ID |
| Repeat cycle | `C01` | Repeat cycle (1 to 7) |
| Track | `T006` | Track number |
| Frame | `F062` | Frame number |

## Download the raw products

You need 5 product types for a single acquisition: `RAW_0S`, `RAW_0M`,
`AUX_ORB`, `AUX_ATT`, `AUX_TEC`. For the full Tomographic chain (STA +
L2A), you need **7 repeat passes** of the same track and frame, i.e. 35
products.

::::{tab-set}

:::{tab-item} Manual download from MAAP Explorer
For a one-off run, the simplest path is to download each product manually
from the MAAP Explorer search results. For the full Tomographic chain,
search by `track`/`frame` and download all 7 cycles (C01 to C07).

Extract everything into a single `Inputs/` folder:

```bash
mkdir -p ~/bps-work/data/run_001/Inputs
mv ~/Downloads/BIO_*.zip ~/bps-work/data/run_001/Inputs/
cd ~/bps-work/data/run_001/Inputs
for f in *.zip; do unzip -o "$f" && rm "$f"; done
```
:::

:::{tab-item} Scripted download
[`scripts/BPS_inputs_download.py`](https://github.com/BioPAL/BPS/blob/main/docs/tutorials/run-bps-locally/scripts/BPS_inputs_download.py)
automates retrieval from a single `RAW_0S` product ID, using the ESA MAAP
catalogue API (requires a free MAAP account).

Run from
[`notebooks/1_BPS_downloads_raws_for_processing.ipynb`](https://github.com/BioPAL/BPS/blob/main/docs/tutorials/run-bps-locally/notebooks/1_BPS_downloads_raws_for_processing.ipynb).
Edit cell 0 to set:

```python
output_path = '/home/yourname/bps-work/data/run_001/'
```

Choose between single repeat cycle (5 products) or full major cycle (35
products, required for STA + L2A) via the matching helper function inside
the notebook.
:::

::::

## Expected layout after download

For a full major cycle, the `Inputs/` folder contains 7 copies of each
product type, all sharing the same track and frame:

```
~/bps-work/data/run_001/Inputs/
├── BIO_AUX_ATT____...  (7 files, one per repeat cycle)
├── BIO_AUX_ORB____...  (7 files)
├── BIO_AUX_TEC____...  (7 files, one per acquisition day)
├── BIO_S1_RAW__0M_..._C01_...  ... _C07_... (7 files)
└── BIO_S1_RAW__0S_..._C01_T006_F062_... ... _C07_T006_F062_... (7 files)
```
