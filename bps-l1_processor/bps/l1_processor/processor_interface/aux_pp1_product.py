# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PP1 product
---------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bps.common import retrieve_aux_product_data_content


@dataclass
class AuxPP1Product:
    """Auxiliary instrument information"""

    aux_pp1_file: Path
    rfi_activation_mask: Path | None

    @classmethod
    def from_product(cls, aux_pp1_product: Path) -> AuxPP1Product:
        """Retrieve content of aux pp1 product"""
        aux_pp1_raw_content = retrieve_aux_product_data_content(aux_pp1_product)

        aux_pp1_file: Path | None = None
        rfi_activation_mask: Path | None = None

        for file in aux_pp1_raw_content:
            if ".nc" in file.name:
                rfi_activation_mask = file

            elif ".xml" in file.name:
                aux_pp1_file = file
            else:
                raise RuntimeError(f"Unexpected file in {aux_pp1_product}: {file}")

        if aux_pp1_file is None:
            raise RuntimeError(f"Missing xml file in {aux_pp1_product}")

        return cls(aux_pp1_file=aux_pp1_file, rfi_activation_mask=rfi_activation_mask)
