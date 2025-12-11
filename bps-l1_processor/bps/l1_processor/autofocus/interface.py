# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Interface to autofocus module
-----------------------------
"""

from dataclasses import dataclass, field
from pathlib import Path

from bps.l1_processor.autofocus import main_autofocus_bps


def _default_pol_list() -> list[str]:
    return ["H/H", "H/V", "V/H", "V/V"]


@dataclass
class AutofocusBlockConfiguration:
    """Configuration of autofocus blocks"""

    azimuth_block_size: int = 128  # delta_aa: blocksize in azimuth
    range_block_size: int = 128  # delta_rr: block size in range
    azimuth_overlap: int = 2  # olfa: overlapping factor between blocks in azimuth
    range_overlap: int = 2  # olfr: overlapping factor between blocks in range


@dataclass
class AutofocusConf:
    """Autofocus configuratoin"""

    # h_iono: ionospheric height in meters, not estimated
    ionospheric_height: float = 350000.0

    blocks_conf: AutofocusBlockConfiguration = field(default_factory=AutofocusBlockConfiguration)

    num_iterations: int = 3  # Niter: number of iterations

    use_faraday: bool = True
    correct_residual: bool = False

    polarizations: list[str] = field(default_factory=_default_pol_list)  # pols_use


@dataclass
class AutofocusInputs:
    """Input products"""

    slc_iono_corrected: Path
    multilooked_rllr_phase: Path  # folder_bb
    phase_screen_bb: Path  # folder_bb_fullres
    fr_plane: Path  # folder_plane
    geomagnetic_field: Path  # folder_bcos
    multilooked_coherence: Path  # folder_frcoh


@dataclass
class AutofocusOutputs:
    """Output products"""

    slc_af_corrected: Path
    phase_screen_af: Path


def run_autofocus(inputs: AutofocusInputs, outputs: AutofocusOutputs, conf: AutofocusConf):
    """Run autofocus module"""
    main_autofocus_bps.run_autofocus(
        str(inputs.slc_iono_corrected),
        str(inputs.multilooked_rllr_phase),
        str(inputs.phase_screen_bb),
        str(inputs.fr_plane),
        str(inputs.geomagnetic_field),
        str(inputs.multilooked_coherence),
        str(outputs.slc_af_corrected),
        str(outputs.phase_screen_af),
        conf.ionospheric_height,
        conf.blocks_conf.azimuth_block_size,
        conf.blocks_conf.range_block_size,
        conf.blocks_conf.azimuth_overlap,
        conf.blocks_conf.range_overlap,
        conf.num_iterations,
        conf.use_faraday,
        conf.correct_residual,
        conf.polarizations,
    )
