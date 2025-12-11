# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PP2 2B FH
-------------
"""

from dataclasses import dataclass


@dataclass
class IntCompressionType:
    """Compression options for MDS integer images"""

    compression_factor: int


@dataclass
class MDSfloatCompressionType:
    """Compression options for MDS float images"""

    compression_factor: int
    max_z_error: float


@dataclass
class AuxProcessingParametersL2BFH:
    """FH L2b configuration"""

    @dataclass
    class CompressionConf:
        """Compression configuration for all FH L2b MDS products"""

        @dataclass
        class MDS:
            """Compression configuration for FH L2b MDS"""

            fh: MDSfloatCompressionType
            fhquality: MDSfloatCompressionType
            bps_fnf: IntCompressionType
            heatmap: MDSfloatCompressionType
            acquisition_id_image: IntCompressionType

        mds: MDS
        mds_block_size: int

    l2bFHProductDOI: str
    forest_masking_flag: bool
    compression_options: CompressionConf
    minimumL2acoverage: float
    rollOffFactorAzimuth: float
    rollOffFactorRange: float
