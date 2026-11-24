<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# 6. Run the processing chain

You can drive the chain either through the run notebook or directly from a
shell. The notebook is the recommended way for a first run; the shell
approach is better suited to batch processing once you understand the
chain.

## Option A: from the notebook

Open
[`notebooks/2_BPS_Run.ipynb`](https://github.com/BioPAL/BPS/blob/main/docs/tutorials/run-bps-locally/notebooks/2_BPS_Run.ipynb)
and edit cell 0 with at minimum:

```python
INPUT_FOLDER      = "/home/yourname/bps-work/data/run_001"
PROCESSOR_VERSION = "04.43"
MISSION_PHASE     = "TOMOGRAPHIC"   # or "INTERFEROMETRIC"
```

The notebook executes the six cells described below (`6.1` → `6.6`), each
of which can also be run on its own from a shell.

## Option B: from a shell

```bash
# 1. Activate the BPS environment
bps443                                   # or:  conda activate bps-dev  (source path)

cd ~/bps-work/BPS/docs/tutorials/run-bps-locally

# 2. Generate JobOrder XMLs for L1F
python scripts/JOBuilder.py L1F 04.43 ~/bps-work/data/run_001

# 3. Run framing on each JobOrder
for jo in ~/bps-work/data/run_001/JobOrder_L1F_*.xml; do
  bps_l1_framing_processor "$jo"
done

# 4. Generate and run L1 (chain mode, picks up EOF frames from L1F output)
python scripts/JOBuilder.py L1_chain 04.43 ~/bps-work/data/run_001
for jo in ~/bps-work/data/run_001/JobOrder_L1_*.xml; do
  bps_l1_processor "$jo"
done

# 5. Generate and run STA (Tomographic phase)
python scripts/JOBuilder.py STA_chain 04.43 ~/bps-work/data/run_001 TOMOGRAPHIC
for jo in ~/bps-work/data/run_001/JobOrder_STA_*.xml; do
  bps_stack_processor "$jo"
done

# 6. Generate and run L2A
python scripts/JOBuilder.py L2A_chain 04.43 ~/bps-work/data/run_001
for jo in ~/bps-work/data/run_001/JobOrder_L2A_*.xml; do
  bps_l2a_processor "$jo"
done
```

## What each step does

### 6.1 Check environment
Verifies that the `biofetch` conda environment exists, creating it from
[`scripts/biofetch.yml`](https://github.com/BioPAL/BPS/blob/main/docs/tutorials/run-bps-locally/scripts/biofetch.yml)
if missing. `biofetch` is the lightweight env used by `JOBuilder.py`; it
does **not** require the full BPS install.

### 6.2 Check input folder
Sanity-check that `INPUT_FOLDER/Inputs/` exists and is not empty. If you
see `⚠️ Inputs/ not found`, go back to
[step 4](04-download-inputs.md).

### 6.3 L1F chain (framing)
Reads the raw ISP packets (`RAW_0S`), determines frame boundaries, and
produces framed L1 virtual products (EOF files, one per frame). After L1F
completes, inspect the generated `frames_*` folder and choose which frames
to process downstream:

```python
SELECTED_FRAMES = []              # empty = process ALL frames
SELECTED_FRAMES = ["306", "307"]  # specific frame IDs
```

### 6.4 L1 chain (SLC focusing)
Focuses each selected frame into a Single Look Complex (SLC) product
(`SCS__1S`). Additional AUX files (`AUX_INS`, `AUX_PP1`) are resolved by
`JOBuilder.py` in this priority order:

1. `INPUT_FOLDER/Inputs/` (if placed manually)
2. `AUX_USER_DIR` (from `config.ini`)
3. `AUX_DEFAULT_DIR` (from `config.ini`)

Output goes to `INPUT_FOLDER/OUTPUT_L1/`.

### 6.5 STA chain (stack processor)
Stacks multiple SLC products from different repeat cycles over the same
frame and produces coherence/interferometric stack products (`STA__1S`).

:::{warning}
STA requires **multiple acquisitions** over the same frame. If you only
downloaded a single repeat cycle, STA will either fail or produce a
degenerate result. Use a full major cycle (7 repeat passes) for the
complete chain.
:::

The `MISSION_PHASE` variable selects `TOMOGRAPHIC` or `INTERFEROMETRIC`.
Output goes to `INPUT_FOLDER/OUTPUT_STA_<frame>_V<version>/`.

### 6.6 L2A chain (biomass estimation)
Reads the STA stack products and estimates L2A products per frame.
Requires the `AUX_PP2_2A_*` file in the AUX directory configured in
`config.ini`. Output goes to
`INPUT_FOLDER/OUTPUT_L2A_<frame>_V<version>/`.
