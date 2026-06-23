#!/bin/bash (source)
# SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
# SPDX-License-Identifier: Apache-2.0
#
# Sources the BPS conda environment and sets LD_LIBRARY_PATH for the L1
# framing native library. Replace <BPS_ROOT> with the absolute path to your
# local bps-work directory before sourcing.

set -uo pipefail

BPS_DOCKER_TAG="04.43"
BPS_PYTHON_ENV=BPS_443

BPS_TEST_PLAN_PATH=<BPS_ROOT>/SW/BPS_V443

BPS_BUNDLE_DIR=${BPS_TEST_PLAN_PATH}/bundle
BPS_CONDA_CHANNEL=${BPS_BUNDLE_DIR}/bps_conda_channel

BPS_TDS_PATH=<BPS_ROOT>/BPS/BPS_V443
BPS_TEST_CASES_PATH=${BPS_TEST_PLAN_PATH}/test_cases
BPS_CONF_FILES_PATH=${BPS_TEST_PLAN_PATH}/configuration_files
BPS_SCRIPTS_PATH=${BPS_TEST_PLAN_PATH}/tools/scripts

# Source conda from the first location that exists.
if [ -f /opt/anaconda3/etc/profile.d/conda.sh ]; then
    source /opt/anaconda3/etc/profile.d/conda.sh
elif [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f /opt/miniconda3/etc/profile.d/conda.sh ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

set -e
conda activate ${BPS_PYTHON_ENV}
set +e

set +u
BPS_L1F_LIBRARIES_PATH=${BPS_BUNDLE_DIR}/l1_framing_processor/lib
export LD_LIBRARY_PATH=${BPS_L1F_LIBRARIES_PATH}:${LD_LIBRARY_PATH:-}
set -u

# Override to use a locally-built BPS executable set instead of the bundle binaries.
USE_LOCAL_EXEC=false
if [ "${USE_LOCAL_EXEC}" = true ]; then
    USE_LOCAL_BIN=/path/to/executables/bin
    USE_LOCAL_LIB=/path/to/executables/lib
    export PATH="${USE_LOCAL_BIN}:${PATH}"
    export LD_LIBRARY_PATH="${USE_LOCAL_LIB}:${LD_LIBRARY_PATH:-}"
fi

# Per-processor memory and CPU limits (used when running through cgroups).
BPS_L1F_MEMORY_LIMIT="2048m"
BPS_L1_MEMORY_LIMIT="25600m"
BPS_STA_MEMORY_LIMIT="65536m"
BPS_L2A_MEMORY_LIMIT="30720m"
BPS_L2B_FH_MEMORY_LIMIT="9216m"
BPS_L2B_FD_MEMORY_LIMIT="9216m"
BPS_L2B_AGB_MEMORY_LIMIT="65536m"

BPS_L1_TMPFS_MEMORY_SIZE="15360m"

BPS_L1F_CPU_LIMIT="1000000"
BPS_L1_CPU_LIMIT="6000000"
BPS_STA_CPU_LIMIT="7000000"
BPS_L2A_CPU_LIMIT="8000000"
BPS_L2B_FH_CPU_LIMIT="4000000"
BPS_L2B_FD_CPU_LIMIT="4000000"
BPS_L2B_AGB_CPU_LIMIT="4000000"
