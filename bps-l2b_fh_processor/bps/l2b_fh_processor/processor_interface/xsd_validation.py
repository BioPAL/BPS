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

import bps.l2b_fh_processor
from bps.common.io.parsing import validate


def validate_aux_pp2_fh(aux_pp2_fh_xml_file: Path):
    """Validate aux pp2 fh"""
    main_folder = importlib.resources.files(bps.l2b_fh_processor)
    aux_pp2_fh_xsd_file = Path(main_folder).joinpath("xsd", "biomass-xsd", "bio-aux-pp2_fh.xsd")

    validate(xml_file=aux_pp2_fh_xml_file, schema=aux_pp2_fh_xsd_file)
