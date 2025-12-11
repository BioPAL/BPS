# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PP2 2B AGB
--------------
"""

from dataclasses import dataclass

from bps.common.io.common_types import (
    FloatArray,
    IntArray,
    MinMaxType,
    MinMaxTypeWithUnit,
)


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
class AuxProcessingParametersL2BAGB:
    """AGB L2b configuration"""

    @dataclass
    class CompressionConf:
        """Compression configuration for all AGB L2b MDS products"""

        @dataclass
        class MDS:
            """Compression configuration for AGB L2b MDS"""

            AGB: MDSfloatCompressionType
            AGBstandardDeviation: MDSfloatCompressionType
            bps_fnf: IntCompressionType
            heatmap: IntCompressionType
            acquisition_id_image: IntCompressionType

        mds: MDS
        mds_block_size: int

    l2bAGBProductDOI: str
    forest_masking_flag: bool
    minimumL2acoverage: float
    rejected_landcover_classes: IntArray
    backscatterLimits: dict[str, MinMaxType]
    angleLimits: MinMaxTypeWithUnit
    meanAGBLimits: MinMaxTypeWithUnit
    stdAGBLimits: MinMaxTypeWithUnit
    relativeAGBLimits: MinMaxType
    referenceSelection: str
    indexingL: str
    indexingA: str
    indexingN: str
    useConstantN: bool
    valuesConstantN: FloatArray
    regressionSolver: str
    minimumPercentageOfFillableVoids: float
    compression_options: CompressionConf
