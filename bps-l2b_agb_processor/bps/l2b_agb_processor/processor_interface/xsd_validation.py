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

import bps.l2b_agb_processor
from bps.common.io.parsing import validate


def validate_aux_pp2_agb(aux_pp2_ab_xml_file: Path):
    """Validate aux pp2 ab"""
    main_folder = importlib.resources.files(bps.l2b_agb_processor)
    aux_pp2_ab_xsd_file = Path(main_folder).joinpath("xsd", "biomass-xsd", "bio-aux-pp2_ab.xsd")

    validate(xml_file=aux_pp2_ab_xml_file, schema=aux_pp2_ab_xsd_file)
