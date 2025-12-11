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

import bps.l1_processor
from bps.common.io.parsing import validate


def validate_aux_pp1(aux_pp1_xml_file: Path):
    """Validate aux pp1"""
    main_folder = importlib.resources.files(bps.l1_processor)
    aux_pp1_xsd_file = Path(main_folder).joinpath("xsd", "biomass-xsd", "bio-aux-pp1.xsd")

    validate(xml_file=aux_pp1_xml_file, schema=aux_pp1_xsd_file)
