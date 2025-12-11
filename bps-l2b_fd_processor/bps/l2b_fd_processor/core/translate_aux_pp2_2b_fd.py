# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PP2 2A FD Translation
-------------------------
"""

from bps.l2b_fd_processor.core.aux_pp2_2b_fd import (
    AuxProcessingParametersL2BFD,
    IntCompressionType,
    MDSfloatCompressionType,
)
from bps.l2b_fd_processor.io import aux_pp2_2b_fd_models


class InvalidAuxPP2_2B_FD(RuntimeError):
    """Raised when input aux pp2 fd is invalid"""


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


def translate_model_to_comp_fd_conf(
    conf: aux_pp2_2b_fd_models.CompressionOptionsL2BFd,
) -> AuxProcessingParametersL2BFD.CompressionConf:
    """Translate FD Compression configuration section to the corresponding conf"""

    assert conf.mds is not None
    assert conf.mds.fd is not None
    assert conf.mds.fd.compression_factor is not None
    assert conf.mds.probability_of_change is not None
    assert conf.mds.probability_of_change.compression_factor is not None
    assert conf.mds.probability_of_change.max_z_error is not None
    assert conf.mds.cfm is not None
    assert conf.mds.cfm.compression_factor is not None
    assert conf.mds.heat_map is not None
    assert conf.mds.heat_map.compression_factor is not None
    assert conf.mds.acquisition_id_image is not None
    assert conf.mds.acquisition_id_image.compression_factor is not None
    assert conf.mds_block_size is not None

    return AuxProcessingParametersL2BFD.CompressionConf(
        mds=AuxProcessingParametersL2BFD.CompressionConf.MDS(
            fd=IntCompressionType(
                compression_factor=conf.mds.fd.compression_factor,
            ),
            probability_of_change=MDSfloatCompressionType(
                conf.mds.probability_of_change.compression_factor,
                conf.mds.probability_of_change.max_z_error,
            ),
            cfm=IntCompressionType(
                compression_factor=conf.mds.cfm.compression_factor,
            ),
            heatmap=IntCompressionType(
                compression_factor=conf.mds.heat_map.compression_factor,
            ),
            acquisition_id_image=IntCompressionType(
                compression_factor=conf.mds.acquisition_id_image.compression_factor,
            ),
        ),
        mds_block_size=conf.mds_block_size,
    )


def translate_model_to_aux_processing_parameters_l2b_fd(
    model: aux_pp2_2b_fd_models.AuxiliaryL2BFdprocessingParameters,
) -> AuxProcessingParametersL2BFD:
    """Translate aux pp2 2b fd to the corresponding structure"""

    assert model.l2b_fdproduct_doi is not None
    assert model.compression_options is not None
    assert model.minimum_l2a_coverage is not None

    return AuxProcessingParametersL2BFD(
        model.l2b_fdproduct_doi,
        minimumL2acoverage=model.minimum_l2a_coverage,
        compression_options=translate_model_to_comp_fd_conf(model.compression_options),
    )
