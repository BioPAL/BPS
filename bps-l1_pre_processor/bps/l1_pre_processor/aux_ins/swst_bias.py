# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
SWST bias
---------
"""

from pathlib import Path

from bps.l1_pre_processor.io.parsing import parse_aux_ins


def retrieve_swst_bias(aux_ins_file: Path) -> float:
    """Retrieve SWST bias from aux ins file"""
    aux_ins_parameters = parse_aux_ins(aux_ins_file.read_text(encoding="utf-8"))
    return aux_ins_parameters.calibration_signals_swp
