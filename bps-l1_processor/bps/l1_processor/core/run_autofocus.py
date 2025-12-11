# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Run custom autofocus l1 chain
-----------------------------
"""

from pathlib import Path

from arepyextras.runner import Environment
from bps.common.runner_helper import run_application
from bps.l1_core_processor.interface import (
    L1CoreProcessorInterface,
    write_l1_coreproc_options_file,
)
from bps.l1_core_processor.processing_options import (
    BPSL1CoreProcessorProductId,
    BPSL1CoreProcessorStep,
)
from bps.l1_processor.autofocus.interface import (
    AutofocusConf,
    AutofocusInputs,
    AutofocusOutputs,
    run_autofocus,
)
from bps.l1_processor.core.interface import L1CoreProcessorInterfaceFiles
from bps.l1_processor.settings.l1_binaries import BPS_L1COREPROC_EXE_NAME


def run_with_autofocus(
    env: Environment,
    interface: L1CoreProcessorInterface,
    files: L1CoreProcessorInterfaceFiles,
):
    """Run the focusing chain twice to add an intermediate autofocus step"""
    core_proc_intermediate_dir = files.input_file.parent.joinpath("l1_core_processor_output", "Intermediate")

    original_steps = interface.options.steps.copy()

    interface.options.interface_settings.remove_intermediate_products = False
    interface.options.output_products[BPSL1CoreProcessorProductId.RANGE_COMPENSATOR_CORRECTED_FACTORS] = str(
        core_proc_intermediate_dir.joinpath("iRGC_corr_factors")
    )

    # Switch off steps after autofocus
    for step in (
        BPSL1CoreProcessorStep.MULTI_LOOKER,
        BPSL1CoreProcessorStep.NESZ_MAP_GENERATOR,
        BPSL1CoreProcessorStep.DENOISER,
        BPSL1CoreProcessorStep.SLANT2_GROUND,
    ):
        interface.options.steps[step] = False
    write_l1_coreproc_options_file(interface.options, files.options_file)

    run_application(
        env,
        BPS_L1COREPROC_EXE_NAME,
        str(files.input_file.absolute()),
        1,
    )

    output_products = interface.options.output_products

    slc_iono_corrected = output_products.get(BPSL1CoreProcessorProductId.POLARIMETRIC_COMPENSATOR)
    assert slc_iono_corrected is not None

    phase_screen_bb = output_products.get(BPSL1CoreProcessorProductId.POLARIMETRIC_COMPENSATOR_PHASE_SCREEN_BB)
    assert phase_screen_bb is not None

    fr_plane = output_products.get(BPSL1CoreProcessorProductId.POLARIMETRIC_COMPENSATOR_FR_PLANE)
    assert fr_plane is not None

    af_inputs = AutofocusInputs(
        slc_iono_corrected=Path(slc_iono_corrected),
        phase_screen_bb=Path(phase_screen_bb),
        fr_plane=Path(fr_plane),
        geomagnetic_field=core_proc_intermediate_dir.joinpath("iGeomagField"),
        multilooked_rllr_phase=core_proc_intermediate_dir.joinpath("iMLKRLLRPhase"),
        multilooked_coherence=core_proc_intermediate_dir.joinpath("iMLKCoherence"),
    )

    af_outputs = AutofocusOutputs(
        slc_af_corrected=files.input_file.parent.joinpath("l1_core_processor_output", "iSLC_af_corrected"),
        phase_screen_af=files.input_file.parent.joinpath("l1_core_processor_output", "iPhaseScreenAF"),
    )

    af_conf = AutofocusConf()

    run_autofocus(af_inputs, af_outputs, af_conf)

    interface.options.interface_settings.remove_intermediate_products = True

    # Restore original steps
    interface.options.steps = original_steps
    write_l1_coreproc_options_file(interface.options, files.options_file)
    run_application(
        env,
        BPS_L1COREPROC_EXE_NAME,
        str(files.input_file.absolute()),
        1,
    )
