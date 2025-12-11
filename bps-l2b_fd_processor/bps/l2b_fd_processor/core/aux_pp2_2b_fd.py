# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PP2 2B FD
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
class AuxProcessingParametersL2BFD:
    """FD L2b configuration"""

    @dataclass
    class CompressionConf:
        """Compression configuration for all FD L2b MDS products"""

        @dataclass
        class MDS:
            """Compression configuration for FD L2b MDS"""

            fd: IntCompressionType
            probability_of_change: MDSfloatCompressionType
            cfm: IntCompressionType
            heatmap: IntCompressionType
            acquisition_id_image: IntCompressionType

        mds: MDS
        mds_block_size: int

    l2bFDProductDOI: str
    minimumL2acoverage: float
    compression_options: CompressionConf
