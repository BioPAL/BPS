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

    assert conf.mds is not None
    assert conf.mds.agb is not None
    assert conf.mds.agb.compression_factor is not None
    assert conf.mds.agb.max_z_error is not None
    assert conf.mds.agbstandard_deviation is not None
    assert conf.mds.agbstandard_deviation.compression_factor is not None
    assert conf.mds.agbstandard_deviation.max_z_error is not None
    assert conf.mds.bps_fnf is not None
    assert conf.mds.bps_fnf.compression_factor is not None
    assert conf.mds.heat_map is not None
    assert conf.mds.heat_map.compression_factor is not None
    assert conf.mds.acquisition_id_image is not None
    assert conf.mds.acquisition_id_image.compression_factor is not None
    assert conf.mds_block_size is not None

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

    assert model.l2b_agbproduct_doi is not None
    assert model.minimum_l2a_coverage is not None
    assert model.forest_masking_flag is not None
    assert model.rejected_landcover_classes is not None
    assert model.backscatter_limits is not None
    assert model.backscatter_limits.hh is not None
    assert model.backscatter_limits.hh.min is not None
    assert model.backscatter_limits.hh.max is not None
    assert model.backscatter_limits.vh is not None
    assert model.backscatter_limits.vh.min is not None
    assert model.backscatter_limits.vh.max is not None
    assert model.backscatter_limits.vv is not None
    assert model.backscatter_limits.vv.min is not None
    assert model.backscatter_limits.vv.max is not None
    assert model.angle_limits is not None
    assert model.angle_limits.min is not None
    assert model.angle_limits.max is not None
    assert model.mean_agblimits is not None
    assert model.mean_agblimits.min is not None
    assert model.mean_agblimits.max is not None
    assert model.std_agblimits is not None
    assert model.std_agblimits.min is not None
    assert model.std_agblimits.max is not None
    assert model.relative_agblimits is not None
    assert model.relative_agblimits.min is not None
    assert model.relative_agblimits.max is not None
    assert model.reference_selection is not None
    assert model.indexing_l is not None
    assert model.indexing_a is not None
    assert model.indexing_n is not None
    assert model.use_constant_n is not None
    assert model.values_constant_n is not None
    assert model.regression_solver is not None
    assert model.minimum_percentage_of_fillable_voids is not None
    assert model.compression_options is not None

    return AuxProcessingParametersL2BAGB(
        model.l2b_agbproduct_doi,
        forest_masking_flag=str_to_bool(model.forest_masking_flag),
        minimumL2acoverage=model.minimum_l2a_coverage,
        rejected_landcover_classes=[int(value) for value in model.rejected_landcover_classes.value.split()],
        backscatterLimits={
            PolarisationType.HH.value: MinMaxType(model.backscatter_limits.hh.min, model.backscatter_limits.hh.max),
            PolarisationType.HV.value: MinMaxType(model.backscatter_limits.vh.min, model.backscatter_limits.vh.max),
            PolarisationType.VV.value: MinMaxType(model.backscatter_limits.vv.min, model.backscatter_limits.vv.max),
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
