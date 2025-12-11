# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
AUX PP1 Translation
-------------------
"""

import numpy as np
from bps.l2a_processor.core.aux_pp2_2a import (
    ADSfloatCompressionType,
    AuxProcessingParametersL2A,
    GeneralConf,
    GroundCancellationConfAGB,
    GroundCancellationConfFD,
    IntCompressionType,
    L2aAGBConf,
    L2aFDConf,
    L2aFHConf,
    L2aTFHConf,
    MDSfloatCompressionType,
    MinMaxNumType,
    MinMaxType,
    OperationalModeType,
)
from bps.l2a_processor.io import aux_pp2_2a_models
from bps.transcoder.io import common_annotation_models_l2


class InvalidAuxPP2_2A(RuntimeError):
    """Raised when input aux pp2 2a is invalid"""


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


def translate_model_to_general_conf(
    conf: aux_pp2_2a_models.GeneralType,
) -> GeneralConf:
    """Translate general configuration section to the corresponding conf"""
    assert conf.apply_calibration_screen is not None
    assert conf.forest_coverage_threshold is not None
    assert conf.forest_mask_interpolation_threshold is not None
    assert conf.subsetting_rule is not None
    assert conf.subsetting_rule.value is not None
    return GeneralConf(
        apply_calibration_screen=conf.apply_calibration_screen,
        forest_coverage_threshold=conf.forest_coverage_threshold,
        forest_mask_interpolation_threshold=conf.forest_mask_interpolation_threshold,
        subsetting_rule=GeneralConf.SubsettingRules(conf.subsetting_rule.value),
    )


def translate_model_to_agb_conf(
    conf: aux_pp2_2a_models.Agbtype,
) -> L2aAGBConf:
    """Translate AGB configuration section to the corresponding conf"""
    assert conf.l2a_agbproduct_doi is not None
    assert conf.product_id is not None
    assert conf.enable_product_flag is not None
    assert conf.ground_cancellation is not None
    assert conf.product_resolution is not None
    assert conf.product_resolution.value is not None
    assert conf.upsampling_factor is not None
    assert conf.compression_options is not None

    return L2aAGBConf(
        l2aAGBProductDOI=conf.l2a_agbproduct_doi,
        product_id=conf.product_id,
        enable_product_flag=str_to_bool(conf.enable_product_flag),
        ground_cancellaton=translate_model_to_gn_conf(conf.ground_cancellation, "AGB"),
        product_resolution=conf.product_resolution.value,
        upsampling_factor=conf.upsampling_factor,
        compression_options=translate_model_to_comp_agb_conf(conf.compression_options),
    )


def translate_model_to_fd_conf(
    conf: aux_pp2_2a_models.Fdtype,
) -> L2aFDConf:
    """Translate FD configuration section to the corresponding conf"""
    assert conf.l2a_fdproduct_doi is not None
    assert conf.product_id is not None
    assert conf.enable_product_flag is not None
    assert conf.ground_cancellation is not None
    assert conf.significance_level is not None
    assert conf.product_resolution is not None
    assert conf.product_resolution.value is not None
    assert conf.upsampling_factor is not None
    assert conf.compression_options is not None

    return L2aFDConf(
        l2aFDProductDOI=conf.l2a_fdproduct_doi,
        product_id=conf.product_id,
        enable_product_flag=str_to_bool(conf.enable_product_flag),
        ground_cancellaton=translate_model_to_gn_conf(conf.ground_cancellation, "FD"),
        significance_level=conf.significance_level,
        product_resolution=conf.product_resolution.value,
        numerical_determinant_limit=(
            conf.numerical_determinant_limit  # Optional
            if conf.numerical_determinant_limit is not None
            else 1.0e-12
        ),
        upsampling_factor=conf.upsampling_factor,
        compression_options=translate_model_to_comp_fd_conf(conf.compression_options),
    )


def translate_model_to_fh_conf(
    conf: aux_pp2_2a_models.Fhtype,
) -> L2aFHConf:
    """Translate FH configuration section to the corresponding conf"""
    assert conf.l2a_fhproduct_doi is not None
    assert conf.product_id is not None
    assert conf.enable_product_flag is not None
    assert conf.vertical_reflectivity_option is not None
    assert conf.vertical_reflectivity_option.value is not None
    assert conf.vertical_reflectivity_default_profile is not None
    assert conf.model_inversion is not None
    assert conf.model_inversion.value is not None
    assert conf.spectral_decorrelation_compensation_flag is not None
    assert conf.correct_terrain_slopes_flag is not None
    assert conf.snrdecorrelation_compensation_flag is not None
    assert conf.normalised_height_estimation_range is not None
    assert conf.normalised_height_estimation_range.min is not None
    assert conf.normalised_height_estimation_range.max is not None
    assert conf.normalised_wavenumber_estimation_range is not None
    assert conf.normalised_wavenumber_estimation_range.min is not None
    assert conf.normalised_wavenumber_estimation_range.max is not None
    assert conf.normalised_wavenumber_estimation_range.num is not None
    assert conf.ground_to_volume_ratio_range is not None
    assert conf.ground_to_volume_ratio_range.min is not None
    assert conf.ground_to_volume_ratio_range.max is not None
    assert conf.ground_to_volume_ratio_range.num is not None
    assert conf.temporal_decorrelation_estimation_range is not None
    assert conf.temporal_decorrelation_estimation_range.min is not None
    assert conf.temporal_decorrelation_estimation_range.max is not None
    assert conf.temporal_decorrelation_estimation_range.num is not None
    assert conf.temporal_decorrelation_ground_to_volume_ratio is not None
    assert conf.residual_decorrelation is not None
    assert conf.product_resolution is not None
    assert conf.product_resolution.value is not None
    assert conf.uncertainty_validvalues_limits is not None
    assert conf.uncertainty_validvalues_limits.min is not None
    assert conf.uncertainty_validvalues_limits.min.value is not None
    assert conf.uncertainty_validvalues_limits.max is not None
    assert conf.uncertainty_validvalues_limits.max.value is not None
    assert conf.vertical_wavenumber_validvalues_limits is not None
    assert conf.vertical_wavenumber_validvalues_limits.min is not None
    assert conf.vertical_wavenumber_validvalues_limits.min.value is not None
    assert conf.vertical_wavenumber_validvalues_limits.max is not None
    assert conf.vertical_wavenumber_validvalues_limits.max.value is not None
    assert conf.lower_height_limit is not None
    assert conf.lower_height_limit.value is not None
    assert conf.upsampling_factor is not None
    assert conf.compression_options is not None

    return L2aFHConf(
        l2aFHProductDOI=conf.l2a_fhproduct_doi,
        product_id=conf.product_id,
        enable_product_flag=str_to_bool(conf.enable_product_flag),
        vertical_reflectivity_option=L2aFHConf.verticalReflectivityOptions(conf.vertical_reflectivity_option.value),
        vertical_reflectivity_default_profile=np.array([val for val in conf.vertical_reflectivity_default_profile.val]),
        model_inversion=L2aFHConf.ModelInversionOptions(conf.model_inversion.value),
        spectral_decorrelation_compensation_flag=str_to_bool(conf.spectral_decorrelation_compensation_flag),
        correct_terrain_slopes_flag=str_to_bool(conf.correct_terrain_slopes_flag),
        snr_decorrelation_compensation_flag=str_to_bool(conf.snrdecorrelation_compensation_flag),
        normalised_height_estimation_range=MinMaxType(
            min=conf.normalised_height_estimation_range.min,
            max=conf.normalised_height_estimation_range.max,
        ),
        normalised_wavenumber_estimation_range=MinMaxNumType(
            min=conf.normalised_wavenumber_estimation_range.min,
            max=conf.normalised_wavenumber_estimation_range.max,
            num=conf.normalised_wavenumber_estimation_range.num,
        ),
        ground_to_volume_ratio_range=MinMaxNumType(
            min=conf.ground_to_volume_ratio_range.min,
            max=conf.ground_to_volume_ratio_range.max,
            num=conf.ground_to_volume_ratio_range.num,
        ),
        temporal_decorrelation_estimation_range=MinMaxNumType(
            min=conf.temporal_decorrelation_estimation_range.min,
            max=conf.temporal_decorrelation_estimation_range.max,
            num=conf.temporal_decorrelation_estimation_range.num,
        ),
        temporal_decorrelation_ground_to_volume_ratio=conf.temporal_decorrelation_ground_to_volume_ratio,
        residual_decorrelation=conf.residual_decorrelation,
        product_resolution=conf.product_resolution.value,
        uncertainty_valid_values_limits=MinMaxType(
            min=conf.uncertainty_validvalues_limits.min.value,
            max=conf.uncertainty_validvalues_limits.max.value,
        ),
        vertical_wavenumber_valid_values_limits=MinMaxType(
            min=conf.vertical_wavenumber_validvalues_limits.min.value,
            max=conf.vertical_wavenumber_validvalues_limits.max.value,
        ),
        lower_height_limit=conf.lower_height_limit.value,
        upsampling_factor=conf.upsampling_factor,
        compression_options=translate_model_to_comp_fh_conf(conf.compression_options),
    )


def translate_model_to_gn_conf(
    conf: aux_pp2_2a_models.GroundCancellationTypeAgb,
    gn_type,
) -> GroundCancellationConfAGB | GroundCancellationConfFD:
    """Translate AGB or FD Ground Cancellation configuration section to the corresponding conf"""
    if gn_type == "AGB":
        assert conf.compute_gnpower_flag is not None
        assert conf.radiometric_calibration_flag is not None
    assert conf.emphasized_forest_height is not None
    assert conf.emphasized_forest_height.value is not None
    assert conf.operational_mode is not None
    assert conf.operational_mode.value is not None
    if conf.images_pair_selection is None:
        images_pair_selection = aux_pp2_2a_models.AuxppacquisitionListType(
            acquisition_folder_name=[
                aux_pp2_2a_models.AuxppacquisitionListType.AcquisitionFolderName(""),
                aux_pp2_2a_models.AuxppacquisitionListType.AcquisitionFolderName(""),
            ],
            count=2,
        )
    else:
        list_temp = []
        for acq_folder_name in conf.images_pair_selection.acquisition_folder_name:
            list_temp.append(
                aux_pp2_2a_models.AuxppacquisitionListType.AcquisitionFolderName(
                    acq_folder_name.value,
                    acq_folder_name.reference_image.lower(),
                    (
                        float(acq_folder_name.average_wavenumber)
                        if acq_folder_name.average_wavenumber is not None
                        else None
                    ),
                ),
            )
        assert len(list_temp) == 2
        images_pair_selection = common_annotation_models_l2.AcquisitionListType(
            acquisition=list_temp,
            count=len(list_temp),
        )
    if conf.disable_ground_cancellation_flag is None:
        disable_ground_cancellation_flag = False
    else:
        disable_ground_cancellation_flag = str_to_bool(conf.disable_ground_cancellation_flag)

    if gn_type == "AGB":
        assert conf.compute_gnpower_flag is not None
        assert conf.radiometric_calibration_flag is not None
        return GroundCancellationConfAGB(
            compute_gn_power_flag=str_to_bool(conf.compute_gnpower_flag),
            radiometric_calibration_flag=str_to_bool(conf.radiometric_calibration_flag),
            emphasized_forest_height=conf.emphasized_forest_height.value,
            operational_mode=OperationalModeType(conf.operational_mode.value),
            images_pair_selection=images_pair_selection,
            disable_ground_cancellation_flag=disable_ground_cancellation_flag,
        )
    if gn_type == "FD":
        return GroundCancellationConfFD(
            emphasized_forest_height=conf.emphasized_forest_height.value,
            operational_mode=OperationalModeType(conf.operational_mode.value),
            images_pair_selection=images_pair_selection,
            disable_ground_cancellation_flag=disable_ground_cancellation_flag,
        )


def translate_model_to_tfh_conf(
    conf: aux_pp2_2a_models.Tfhtype,
) -> L2aTFHConf:
    """Translate TOMO FH configuration section to the corresponding conf"""
    assert conf.l2a_tfhproduct_doi is not None
    assert conf.product_id is not None
    assert conf.enable_product_flag is not None
    assert conf.enable_super_resolution is not None
    assert conf.product_resolution is not None
    assert conf.product_resolution.value is not None
    assert conf.regularization_noise_factor is not None
    assert conf.power_threshold is not None
    assert conf.median_factor is not None
    assert conf.estimation_valid_values_limits is not None
    assert conf.estimation_valid_values_limits.min is not None
    assert conf.estimation_valid_values_limits.min.value is not None
    assert conf.estimation_valid_values_limits.max is not None
    assert conf.estimation_valid_values_limits.max.value is not None
    assert conf.vertical_range is not None
    assert conf.vertical_range.min is not None
    assert conf.vertical_range.min.value is not None
    assert conf.vertical_range.max is not None
    assert conf.vertical_range.max.value is not None
    assert conf.vertical_range.sampling is not None
    assert conf.compression_options is not None

    return L2aTFHConf(
        l2aTOMOFHProductDOI=conf.l2a_tfhproduct_doi,
        product_id=conf.product_id,
        enable_product_flag=str_to_bool(conf.enable_product_flag),
        enable_super_resolution=str_to_bool(conf.enable_super_resolution),
        product_resolution=conf.product_resolution.value,
        regularization_noise_factor=conf.regularization_noise_factor,
        power_threshold=conf.power_threshold,
        median_factor=conf.median_factor,
        estimation_valid_values_limits=conf.estimation_valid_values_limits,
        vertical_range=conf.vertical_range,
        compression_options=translate_model_to_comp_tfh_conf(conf.compression_options),
    )


def translate_model_to_comp_agb_conf(
    conf: aux_pp2_2a_models.CompressionOptionsL2AAgb,
) -> L2aAGBConf.CompressionConf:
    """Translate AGB Compression configuration section to the corresponding conf"""

    assert conf.mds is not None
    assert conf.mds.gn is not None
    assert conf.mds.gn.compression_factor is not None
    assert conf.mds.gn.max_z_error is not None
    assert conf.ads is not None
    assert conf.ads.fnf is not None
    assert conf.ads.fnf.compression_factor is not None
    assert conf.ads.local_incidence_angle is not None
    assert conf.ads.local_incidence_angle.compression_factor is not None
    assert conf.ads.local_incidence_angle.least_significant_digit is not None
    assert conf.mds_block_size is not None
    assert conf.ads_block_size is not None

    return L2aAGBConf.CompressionConf(
        mds=L2aAGBConf.CompressionConf.MDS(
            gn=MDSfloatCompressionType(
                compression_factor=conf.mds.gn.compression_factor,
                max_z_error=conf.mds.gn.max_z_error,
            ),
        ),
        ads=L2aAGBConf.CompressionConf.ADS(
            fnf=IntCompressionType(compression_factor=conf.ads.fnf.compression_factor),
            incidence_angle=ADSfloatCompressionType(
                compression_factor=conf.ads.local_incidence_angle.compression_factor,
                least_significant_digit=conf.ads.local_incidence_angle.least_significant_digit,
            ),
        ),
        mds_block_size=conf.mds_block_size,
        ads_block_size=conf.ads_block_size,
    )


def translate_model_to_comp_fd_conf(
    conf: aux_pp2_2a_models.CompressionOptionsL2AFd,
) -> L2aFDConf.CompressionConf:
    """Translate FD Compression configuration section to the corresponding conf"""

    assert conf.mds is not None
    assert conf.mds.fd is not None
    assert conf.mds.fd.compression_factor is not None
    assert conf.mds.fd.compression_factor is not None
    assert conf.mds.probability_of_change is not None
    assert conf.mds.probability_of_change.compression_factor is not None
    assert conf.mds.probability_of_change.max_z_error is not None
    assert conf.mds.cfm is not None
    assert conf.mds.cfm.compression_factor is not None
    assert conf.ads is not None
    assert conf.ads.fnf is not None
    assert conf.ads.fnf.compression_factor is not None
    assert conf.ads.acm is not None
    assert conf.ads.acm.compression_factor is not None
    assert conf.ads.acm.least_significant_digit is not None
    assert conf.mds_block_size is not None
    assert conf.ads_block_size is not None

    return L2aFDConf.CompressionConf(
        mds=L2aFDConf.CompressionConf.MDS(
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
        ),
        ads=L2aFDConf.CompressionConf.ADS(
            fnf=IntCompressionType(compression_factor=conf.ads.fnf.compression_factor),
            acm=ADSfloatCompressionType(
                compression_factor=conf.ads.acm.compression_factor,
                least_significant_digit=conf.ads.acm.least_significant_digit,
            ),
            number_of_averages=IntCompressionType(compression_factor=conf.ads.acm.compression_factor),
        ),
        mds_block_size=conf.mds_block_size,
        ads_block_size=conf.ads_block_size,
    )


def translate_model_to_comp_fh_conf(
    conf: aux_pp2_2a_models.CompressionOptionsL2AFh,
) -> L2aFHConf.CompressionConf:
    """Translate FH Compression configuration section to the corresponding conf"""

    assert conf.mds is not None
    assert conf.mds.fh is not None
    assert conf.mds.fh.compression_factor is not None
    assert conf.mds.fh.max_z_error is not None
    assert conf.mds.quality is not None
    assert conf.mds.quality.compression_factor is not None
    assert conf.mds.quality.max_z_error is not None
    assert conf.ads is not None
    assert conf.ads.fnf is not None
    assert conf.ads.fnf.compression_factor is not None
    assert conf.mds_block_size is not None
    assert conf.ads_block_size is not None

    return L2aFHConf.CompressionConf(
        mds=L2aFHConf.CompressionConf.MDS(
            fh=MDSfloatCompressionType(
                compression_factor=conf.mds.fh.compression_factor,
                max_z_error=conf.mds.fh.max_z_error,
            ),
            quality=MDSfloatCompressionType(
                compression_factor=conf.mds.quality.compression_factor,
                max_z_error=conf.mds.quality.max_z_error,
            ),
        ),
        ads=L2aFHConf.CompressionConf.ADS(
            fnf=IntCompressionType(compression_factor=conf.ads.fnf.compression_factor),
        ),
        mds_block_size=conf.mds_block_size,
        ads_block_size=conf.ads_block_size,
    )


def translate_model_to_comp_tfh_conf(
    conf: aux_pp2_2a_models.CompressionOptionsL2ATfh,
) -> L2aTFHConf.CompressionConf:
    """Translate TOMO FH Compression configuration section to the corresponding conf"""

    assert conf.mds is not None
    assert conf.mds.tfh is not None
    assert conf.mds.tfh.compression_factor is not None
    assert conf.mds.tfh.max_z_error is not None
    assert conf.mds.quality is not None
    assert conf.mds.quality.compression_factor is not None
    assert conf.mds.quality.max_z_error is not None
    assert conf.ads is not None
    assert conf.ads.fnf is not None
    assert conf.ads.fnf.compression_factor is not None
    assert conf.mds_block_size is not None
    assert conf.ads_block_size is not None

    return L2aTFHConf.CompressionConf(
        mds=L2aTFHConf.CompressionConf.MDS(
            tfh=MDSfloatCompressionType(
                compression_factor=conf.mds.tfh.compression_factor,
                max_z_error=conf.mds.tfh.max_z_error,
            ),
            quality=MDSfloatCompressionType(
                compression_factor=conf.mds.quality.compression_factor,
                max_z_error=conf.mds.quality.max_z_error,
            ),
        ),
        ads=L2aTFHConf.CompressionConf.ADS(
            fnf=IntCompressionType(compression_factor=conf.ads.fnf.compression_factor),
        ),
        mds_block_size=conf.mds_block_size,
        ads_block_size=conf.ads_block_size,
    )


def translate_model_to_aux_processing_parameters_l2a(
    model: aux_pp2_2a_models.AuxiliaryL2AProcessingParameters,
) -> AuxProcessingParametersL2A:
    """Translate aux pp2 2a to the corresponding structure"""

    general_params = model.general
    agb_params = model.agb
    fd_params = model.fd
    fh_params = model.fh
    tfh_params = model.tfh

    assert general_params is not None
    assert agb_params is not None
    assert fd_params is not None
    assert fh_params is not None
    assert tfh_params is not None

    assert general_params.apply_calibration_screen is not None
    assert general_params.forest_coverage_threshold is not None
    assert general_params.forest_mask_interpolation_threshold is not None
    assert general_params.subsetting_rule is not None

    return AuxProcessingParametersL2A(
        general=translate_model_to_general_conf(general_params),
        agb=translate_model_to_agb_conf(agb_params),
        fd=translate_model_to_fd_conf(fd_params),
        fh=translate_model_to_fh_conf(fh_params),
        tfh=translate_model_to_tfh_conf(tfh_params),
    )
