# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PP2 2A AGB Translation
--------------------------
"""

from bps.common.io.common_types import MinMaxType, PolarisationType
from bps.l2b_agb_processor.core.aux_pp2_2b_agb import (
    AuxProcessingParametersL2BAGB,
    IntCompressionType,
    MDSfloatCompressionType,
)
from bps.l2b_agb_processor.io import aux_pp2_2b_agb_models


class InvalidAuxPP2_2B_AGB(RuntimeError):
    """Raised when input aux pp2 agb is invalid"""


class InvalidBoolTagContent(RuntimeError):
    """Raised when input bool tag content is different from true or false"""


def str_to_bool(tag: str) -> bool:
    """Safe string to bool tag content conversion"""
    tag = tag.lower()
    if tag == "true":
        return True
    if tag == "false":
        return False
    raise InvalidBoolTagContent(tag)


def translate_model_to_comp_agb_conf(
    conf: aux_pp2_2b_agb_models.CompressionOptionsL2BAb,
) -> AuxProcessingParametersL2BAGB.CompressionConf:
    """Translate AGB Compression configuration section to the corresponding conf"""
    return AuxProcessingParametersL2BAGB.CompressionConf(
        mds=AuxProcessingParametersL2BAGB.CompressionConf.MDS(
            AGB=MDSfloatCompressionType(
                conf.mds.agb.compression_factor,
                conf.mds.agb.max_z_error,
            ),
            AGBstandardDeviation=MDSfloatCompressionType(
                conf.mds.agbstandard_deviation.compression_factor,
                conf.mds.agbstandard_deviation.max_z_error,
            ),
            bps_fnf=IntCompressionType(compression_factor=conf.mds.bps_fnf.compression_factor),
            heatmap=IntCompressionType(
                compression_factor=conf.mds.heat_map.compression_factor,
            ),
            acquisition_id_image=IntCompressionType(
                compression_factor=conf.mds.acquisition_id_image.compression_factor,
            ),
        ),
        mds_block_size=conf.mds_block_size,
    )


def translate_model_to_aux_processing_parameters_l2b_agb(
    model: aux_pp2_2b_agb_models.AuxiliaryL2BAbprocessingParameters,
) -> AuxProcessingParametersL2BAGB:
    """Translate aux pp2 2b fd to the corresponding structure"""
    return AuxProcessingParametersL2BAGB(
        model.l2b_agbproduct_doi,
        forest_masking_flag=str_to_bool(model.forest_masking_flag),
        minimumL2acoverage=model.minimum_l2a_coverage,
        rejected_landcover_classes=[int(value) for value in model.rejected_landcover_classes.value.split()],
        backscatterLimits={
            PolarisationType.HH.value: MinMaxType(
                min=model.backscatter_limits.hh.min, max=model.backscatter_limits.hh.max
            ),
            PolarisationType.HV.value: MinMaxType(
                min=model.backscatter_limits.vh.min, max=model.backscatter_limits.vh.max
            ),
            PolarisationType.VV.value: MinMaxType(
                min=model.backscatter_limits.vv.min, max=model.backscatter_limits.vv.max
            ),
        },
        angleLimits=model.angle_limits,
        meanAGBLimits=model.mean_agblimits,
        stdAGBLimits=model.std_agblimits,
        relativeAGBLimits=model.relative_agblimits,
        referenceSelection=model.reference_selection.value,
        indexingL=model.indexing_l.value,
        indexingA=model.indexing_a.value,
        indexingN=model.indexing_n.value,
        useConstantN=str_to_bool(model.use_constant_n),
        valuesConstantN=[float(value) for value in model.values_constant_n.value.split()],
        regressionSolver=model.regression_solver,
        minimumPercentageOfFillableVoids=model.minimum_percentage_of_fillable_voids,
        compression_options=translate_model_to_comp_agb_conf(model.compression_options),
    )
