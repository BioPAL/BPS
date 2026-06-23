<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# 7. Reference and troubleshooting

## Supporting scripts

### `JOBuilder.py`

```bash
python scripts/JOBuilder.py <processing_type> <processor_version> <input_folder> [mission_phase]
```

| Argument | Description | Example |
|---|---|---|
| `processing_type` | `L1F`, `L1`, `L1_chain`, `STA`, `STA_chain`, `L2A`, `L2A_chain` | `L1F` |
| `processor_version` | Format `XX.XX` | `04.43` |
| `input_folder` | Root processing folder | `~/bps-work/data/run_001` |
| `mission_phase` | Required only for STA/STA_chain | `TOMOGRAPHIC` |

`JOBuilder.py` reads `config.ini` from the **same directory**.

### `config.ini`

After the `sed` substitution from
[step 1](01-prerequisites.md#replace-the-bps_root-placeholders),
`config.ini` already points at your local paths. The file is reproduced
here for reference; see
[`scripts/config.ini`](https://github.com/BioPAL/BPS/blob/main/docs/tutorials/run-bps-locally/scripts/config.ini)
for the live version.

```ini
[TEMPLATE_JO]
L1F = ~/bps-work/BPS/docs/tutorials/run-bps-locally/CONFIGURATION_FILE/JO_TEMPLATE/BIO_L1F_P_TEMPLATE_JobOrder.xml
L1  = ~/bps-work/BPS/docs/tutorials/run-bps-locally/CONFIGURATION_FILE/JO_TEMPLATE/BIO_L1_P_TEMPLATE_JobOrder.xml
STA = ~/bps-work/BPS/docs/tutorials/run-bps-locally/CONFIGURATION_FILE/JO_TEMPLATE/BIO_STA_P_TEMPLATE_JobOrder.xml
L2A = ~/bps-work/BPS/docs/tutorials/run-bps-locally/CONFIGURATION_FILE/JO_TEMPLATE/BIO_L2A_P_ALL_TEMPLATE_JobOrder.xml

[AUX_STATIC]
DEM = ~/bps-work/bps/internal_resources/DEM
FNF = ~/bps-work/bps/internal_resources/FNF
GMF = ~/bps-work/bps/internal_resources/GMF
IRI = ~/bps-work/bps/internal_resources/IRI

[AUX]
AUX_DEFAULT_DIR = ~/bps-work/BPS/docs/tutorials/run-bps-locally/CONFIGURATION_FILE/AUX_443
AUX_USER_DIR    = ~/bps-work/BPS/docs/tutorials/run-bps-locally/CONFIGURATION_FILE/AUX_USER
```

The `AUX_STATIC` block points at static auxiliary data (DEM, FNF, GMF,
IRI) installed with the bundle under
`<conda-env>/share/bps/internal_resources/`. Either symlink that folder or
update the paths to match your install.

### `set_environment.bash`

Activates the BPS conda environment and sets `LD_LIBRARY_PATH` for the
native L1 framing library. Sourced automatically by the `bps443` alias
added during install. When installing a new version, update the top of the
file:

```bash
BPS_DOCKER_TAG="04.43"
BPS_PYTHON_ENV=BPS_443
```

### `footprint_generic_start_stop.py`

```bash
python l0_footprint_generator/footprint_generic_start_stop.py \
  --raw_0s   /path/to/Inputs/BIO_S1_RAW__0S_... \
  --aux_orb  /path/to/Inputs/BIO_AUX_ORB___... \
  --start_time 2025-11-21T09:52:53.000000 \
  --stop_time  2025-11-21T09:54:50.000000
```

Prints the four corner coordinates (lat/lon) of the footprint.

(troubleshooting)=
## Troubleshooting

:::{dropdown} <i class="doc-icon fa-solid fa-memory fa-fw" aria-hidden="true"></i> The conda solver crashes during installation
Your system RAM is below 20 GB. Free up memory (close other applications)
or add RAM. On a workstation with 16 GB or less, install one processor at
a time instead of the whole bundle, or use the cloud-based MAAP setup.
:::

:::{dropdown} <i class="doc-icon fa-solid fa-circle-xmark fa-fw" aria-hidden="true"></i> No L1F JobOrder files found
`JOBuilder.py` failed silently. Check `processing_L1F.log` in your
`INPUT_FOLDER`. The most common causes are an incorrect template path in
`config.ini` or a missing `RAW__0S` file in `Inputs/`.
:::

:::{dropdown} <i class="doc-icon fa-solid fa-triangle-exclamation fa-fw" aria-hidden="true"></i> AUX_INS___ not found
The `AUX_INS` file is missing from `Inputs/`, `AUX_USER_DIR` and
`AUX_DEFAULT_DIR`. Verify that `config.ini` points to the correct AUX
version folder for your BPS version.
:::

:::{dropdown} <i class="doc-icon fa-regular fa-clock fa-fw" aria-hidden="true"></i> No packets found in the specified time interval (footprint script)
The start/stop times do not overlap with the packets in the `RAW_0S` file.
Use the format `YYYY-MM-DDTHH:MM:SS.000000` and stay within the
acquisition window.
:::

:::{dropdown} <i class="doc-icon fa-solid fa-layer-group fa-fw" aria-hidden="true"></i> STA produces an error about insufficient products
You downloaded only a single repeat cycle. STA needs multiple acquisitions
over the same frame. Re-download a full major cycle (7 repeat passes).
:::

:::{dropdown} <i class="doc-icon fa-solid fa-puzzle-piece fa-fw" aria-hidden="true"></i> libl1framing.so: cannot open shared object file
The L1 framing native library is missing from `LD_LIBRARY_PATH`. Make
sure you sourced `set_environment.bash` (or ran the `bps443` alias)
before launching the processor. Source-install users: confirm
`bps-l1_binaries` was installed in
[step 3.3](03-install-from-source.md#33-install-the-native-binary-packages-from-the-bundle).
:::

:::{dropdown} <i class="doc-icon fa-solid fa-server fa-fw" aria-hidden="true"></i> bps_l1_processor crashes with "shared memory" error on Linux
`/dev/shm` is too small. Check its size with `df -h /dev/shm` and remount
larger if needed: `sudo mount -o remount,size=12G /dev/shm`.
:::

:::{dropdown} <i class="doc-icon fa-brands fa-python fa-fw" aria-hidden="true"></i> Source-install: ModuleNotFoundError: No module named 'arepytools'
Aresys-provided Python deps are pulled from PyPI when you run
`pip install -e .`. If a package is private to Aresys, install the bundle
version of the package via conda from the bundle channel:

```bash
conda install -c "file://$HOME/bps-work/SW/BPS_V443/bundle/bps_conda_channel" arepytools
```
:::

## Next steps

- Browse the [Science Guide](../../science-guide/index.md) for the
  algorithmic details of each processor (L1 SAR, AGB, FH, FD).
- See the [User Guide](../../user-guide/index.md) for end-user reference
  documentation.
- Source-install users: see the
  [Contributing guide](../../contributing/index.md) to open a pull request
  against any Python processor.
- Report a bug or suggest an improvement to the example scripts in the
  [BPS issue tracker](https://github.com/BioPAL/BPS/issues).
