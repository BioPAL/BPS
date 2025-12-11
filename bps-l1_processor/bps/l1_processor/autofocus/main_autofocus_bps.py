# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l., DLR, Deimos Space
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""_summary_"""

from bps.common import bps_logger
from bps.common.decorators import log_elapsed_time
from bps.l1_processor.autofocus.af_lib import ImportFaraday, masterAutofocus


@log_elapsed_time("AutoFocus")
def run_autofocus(
    folder2operate: str,
    folder_bb: str,
    folder_bb_fullres: str,
    folder_plane: str,
    folder_bcos: str,
    folder_frcoh: str,
    folderOutputSLC: str,
    folderPhaseScreen: str,
    h_iono: float,
    delta_aa: int,
    delta_rr: int,
    olfa: int,
    olfr: int,
    Niter: int,
    useFaraday: bool,
    correctResidual: bool,
    pols_use: list,
):
    bps_logger.info(f"Autofocus - Input folder: {folder2operate}")
    bps_logger.info(f"Autofocus - Save output image in: {folderOutputSLC}")
    bps_logger.info(f"Autofocus - Save final phase screen in: {folderPhaseScreen}")
    bps_logger.info(f"Autofocus - h iono: {h_iono} m")
    bps_logger.info(f"Autofocus - Block size is: [{delta_aa}, {delta_rr}] px")
    bps_logger.info(f"Autofocus - Overlapping factor between blocks is: [{olfa}, {olfr}]")
    bps_logger.info(f"Autofocus - Number of iterations: {Niter}")
    bps_logger.info(f"Autofocus - Use FR: {useFaraday}")
    bps_logger.info(f"Autofocus - Correct residuals: {correctResidual}")

    phase_fr = ImportFaraday(
        folder2operate,
        folder_bb,
        folder_bb_fullres,
        folder_plane,
        folder_bcos,
        folder_frcoh,
    )

    master_autofocus = masterAutofocus(
        folder2operate=folder2operate,
        folderPhaseScreen=folderPhaseScreen,
        folderOutputSLC=folderOutputSLC,
        h_iono=h_iono,
        delta_aa=delta_aa,
        delta_rr=delta_rr,
        olfa=olfa,
        olfr=olfr,
        Niter=Niter,
        pols_use=pols_use,
        returnCorrectedSLC=True,
        saveAllScreens=False,
        phase_fr=phase_fr,
        weighted=True,
        constrained=True,
    )

    master_autofocus.run()
