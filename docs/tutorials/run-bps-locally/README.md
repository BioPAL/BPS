<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Run BPS locally — example scripts and notebooks

This folder bundles everything needed to run the full BPS processing chain
(L1F → L1 → STA → L2A) on a local workstation. It is the companion to the
[Run BPS on your own computer](../run-bps-locally.md)
tutorial.

## Layout

```
docs/tutorials/run-bps-locally/
├── README.md                                   this file
├── scripts/
│   ├── JOBuilder.py                            JobOrder XML generator
│   ├── BPS_inputs_download.py                  ESA MAAP-backed raw downloader
│   ├── BiomassProduct.py                       Biomass product metadata parser
│   ├── biofetch.yml                            conda env spec for JOBuilder
│   └── config.ini                              local paths (edit before use)
├── notebooks/
│   ├── 0_BPS_installation.ipynb                step 0: install BPS
│   ├── 1_BPS_downloads_raws_for_processing.ipynb   step 1: download raws
│   └── 2_BPS_Run.ipynb                         step 2: run the chain
└── CONFIGURATION_FILE/
    ├── set_environment.bash                    conda activation + LD_LIBRARY_PATH
    └── JO_TEMPLATE/                            JobOrder XML templates (7 files)
```

## Before first use

Open [`scripts/config.ini`](scripts/config.ini) and
[`CONFIGURATION_FILE/set_environment.bash`](CONFIGURATION_FILE/set_environment.bash)
and replace every `<BPS_ROOT>` placeholder with the absolute path to your
local working directory (e.g. `/home/yourname/bps-work`).

Then follow the
[Run BPS on your own computer](../run-bps-locally.md)
tutorial step by step.

## Attribution

This bundle is adapted from
[BioPAL/MAAP_BPS_scripts](https://github.com/BioPAL/MAAP_BPS_scripts), which
targets the ESA MAAP JupyterLab environment. The original work is by Aresys
S.r.l. and the BioPAL team. Adaptations made here:

- Hardcoded `/home/jovyan/` paths replaced with a `<BPS_ROOT>` placeholder.
- `set_environment.bash` falls back to `$HOME/miniforge3` and
  `$HOME/miniconda3` if the system conda is not at `/opt/anaconda3`.

For the original, unmodified MAAP-targeted version, see the upstream
repository.
