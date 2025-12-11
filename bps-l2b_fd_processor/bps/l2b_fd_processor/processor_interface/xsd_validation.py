# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD validation utils
--------------------
"""

import importlib.resources
from pathlib import Path

import bps.l2b_fd_processor
from bps.common.io.parsing import validate


def validate_aux_pp2_fd(aux_pp2_fd_xml_file: Path):
    """Validate aux pp2 fd"""
    main_folder = importlib.resources.files(bps.l2b_fd_processor)
    aux_pp2_fd_xsd_file = Path(main_folder).joinpath("xsd", "biomass-xsd", "bio-aux-pp2_fd.xsd")

    validate(xml_file=aux_pp2_fd_xml_file, schema=aux_pp2_fd_xsd_file)
