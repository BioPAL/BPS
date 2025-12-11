# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PP2 2A FH Translation
-------------------------
"""

from bps.l2b_fh_processor.core.aux_pp2_2b_fh import (
    AuxProcessingParametersL2BFH,
    IntCompressionType,
    MDSfloatCompressionType,
)
from bps.l2b_fh_processor.io import aux_pp2_2b_fh_models


class InvalidAuxPP2_2B_FH(RuntimeError):
    """Raised when input aux pp2 fh is invalid"""


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


def translate_model_to_comp_fh_conf(
    conf: aux_pp2_2b_fh_models.CompressionOptionsL2BFh,
) -> AuxProcessingParametersL2BFH.CompressionConf:
    """Translate FH Compression configuration section to the corresponding conf"""

    assert conf.mds is not None
    assert conf.mds.fh is not None
    assert conf.mds.fh.compression_factor is not None
    assert conf.mds.fh.max_z_error is not None
    assert conf.mds.quality is not None
    assert conf.mds.quality.compression_factor is not None
    assert conf.mds.quality.max_z_error is not None
    assert conf.mds.bps_fnf is not None
    assert conf.mds.bps_fnf.compression_factor is not None
    assert conf.mds.heat_map is not None
    assert conf.mds.heat_map.compression_factor is not None
    assert conf.mds.heat_map.max_z_error is not None
    assert conf.mds.acquisition_id_image is not None
    assert conf.mds.acquisition_id_image.compression_factor is not None
    assert conf.mds_block_size is not None

    return AuxProcessingParametersL2BFH.CompressionConf(
        mds=AuxProcessingParametersL2BFH.CompressionConf.MDS(
            fh=MDSfloatCompressionType(
                compression_factor=conf.mds.fh.compression_factor,
                max_z_error=conf.mds.fh.max_z_error,
            ),
            fhquality=MDSfloatCompressionType(
                conf.mds.quality.compression_factor,
                conf.mds.quality.max_z_error,
            ),
            bps_fnf=IntCompressionType(
                compression_factor=conf.mds.bps_fnf.compression_factor,
            ),
            heatmap=MDSfloatCompressionType(
                compression_factor=conf.mds.heat_map.compression_factor,
                max_z_error=conf.mds.heat_map.max_z_error,
            ),
            acquisition_id_image=IntCompressionType(
                compression_factor=conf.mds.acquisition_id_image.compression_factor,
            ),
        ),
        mds_block_size=conf.mds_block_size,
    )


def translate_model_to_aux_processing_parameters_l2b_fh(
    model: aux_pp2_2b_fh_models.AuxiliaryL2BFhprocessingParameters,
) -> AuxProcessingParametersL2BFH:
    """Translate aux pp2 2b fh to the corresponding structure"""

    assert model.l2b_fhproduct_doi is not None
    assert model.compression_options is not None
    assert model.minimum_l2a_coverage is not None
    assert model.forest_masking_flag is not None
    assert model.roll_off_factor_azimuth is not None
    assert model.roll_off_factor_range is not None

    return AuxProcessingParametersL2BFH(
        model.l2b_fhproduct_doi,
        forest_masking_flag=str_to_bool(model.forest_masking_flag),
        minimumL2acoverage=model.minimum_l2a_coverage,
        compression_options=translate_model_to_comp_fh_conf(model.compression_options),
        rollOffFactorAzimuth=model.roll_off_factor_azimuth,
        rollOffFactorRange=model.roll_off_factor_range,
    )
