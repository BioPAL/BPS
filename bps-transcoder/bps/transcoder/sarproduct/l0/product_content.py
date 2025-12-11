# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""BPS L0 product content"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class L0ProductContent:
    """Relative paths to product content."""

    mph_file: Path

    rxh: Path
    """RX H file"""
    rxv: Path
    """RX v file"""
    ia_rxh: Path
    """Instrument ancillary H file"""
    ia_rxv: Path
    """Instrument ancillary v file"""
    idx_rxh: Path | None
    """Index file fo RX H file"""
    idx_rxv: Path | None
    """Index file fo RX V file"""

    @classmethod
    def from_name(cls, name: str) -> L0ProductContent:
        """Build all paths regardless of product existance"""

        standard_product = name[13] != "M"

        lower_name = name.lower()
        name_root = lower_name[:-10]

        mph_file = Path(lower_name + ".xml")

        rxh = Path(name_root + "_rxh.dat")
        rxv = Path(name_root + "_rxv.dat")
        ia_rxh = Path(name_root + "_ia_rxh.dat")
        ia_rxv = Path(name_root + "_ia_rxv.dat")

        idx_rxh = idx_rxv = None
        if standard_product:
            idx_rxh = Path(name_root + "_idx_rxh.dat")
            idx_rxv = Path(name_root + "_idx_rxv.dat")

        return L0ProductContent(
            mph_file=mph_file,
            rxh=rxh,
            rxv=rxv,
            ia_rxh=ia_rxh,
            ia_rxv=ia_rxv,
            idx_rxh=idx_rxh,
            idx_rxv=idx_rxv,
        )
