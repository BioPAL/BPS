# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L2A variant-specific annotation translation
-------------------------------------------
"""

from pathlib import Path

from bps.common import bps_logger
from bps.common.io import common
from bps.common.io.parsing import serialize
from bps.transcoder.io import (
    main_annotation_l2a_fh,
    reader_core,
    translate_common,
    translate_common_annotation_l2l3,
)
from bps.transcoder.io.v445 import main_annotation_models_l2a_fh as v445
from bps.transcoder.io.v450 import main_annotation_models_l2a_fh as v450

CompressionOptionsL2A = v450.CompressionOptionsL2A | v445.CompressionOptionsL2A
LATEST_VERSION = v450


def translate_compression_options_l2a_fh(
    options: CompressionOptionsL2A,
) -> main_annotation_l2a_fh.CompressionOptionsL2AFH:
    if isinstance(options, v450.CompressionOptionsL2A):
        geometric_lut_options = main_annotation_l2a_fh.CompressionFactorLeastSignificantDigitType(
            options.ads.geometric_luts.compression_factor,
            options.ads.geometric_luts.least_significant_digit,
        )
    else:
        assert isinstance(options, v445.CompressionOptionsL2A)
        bps_logger.warning("Geometric LUT options are not available for older versions; using default values.")
        geometric_lut_options = main_annotation_l2a_fh.CompressionFactorLeastSignificantDigitType(
            compression_factor=5, least_significant_digit=5
        )

    return main_annotation_l2a_fh.CompressionOptionsL2AFH(
        mds=main_annotation_l2a_fh.CompressionOptionsL2AFH.Mds(
            fh=main_annotation_l2a_fh.CompressionFactorMaxZErrorType(
                options.mds.fh.compression_factor, options.mds.fh.max_z_error
            ),
            quality=main_annotation_l2a_fh.CompressionFactorMaxZErrorType(
                options.mds.quality.compression_factor, options.mds.quality.max_z_error
            ),
        ),
        ads=main_annotation_l2a_fh.CompressionOptionsL2AFH.Ads(
            fnf=main_annotation_l2a_fh.CompressionFactorType(options.ads.fnf.compression_factor),
            geometric_luts=geometric_lut_options,
        ),
        mds_block_size=options.mds_block_size,
        ads_block_size=options.ads_block_size,
    )


def translate_compression_options_l2a_fh_to_model(
    options: main_annotation_l2a_fh.CompressionOptionsL2AFH,
) -> LATEST_VERSION.CompressionOptionsL2A:
    model = LATEST_VERSION.CompressionOptionsL2A
    return model(
        mds=model.Mds(
            fh=model.Mds.Fh(
                compression_factor=options.mds.fh.compression_factor,
                max_z_error=options.mds.fh.max_z_error,
            ),
            quality=model.Mds.Quality(
                compression_factor=options.mds.quality.compression_factor,
                max_z_error=options.mds.quality.max_z_error,
            ),
        ),
        ads=model.Ads(
            fnf=model.Ads.Fnf(compression_factor=options.ads.fnf.compression_factor),
            geometric_luts=model.Ads.GeometricLuts(
                compression_factor=options.ads.geometric_luts.compression_factor,
                least_significant_digit=options.ads.geometric_luts.least_significant_digit,
            ),
        ),
        mds_block_size=options.mds_block_size,
        ads_block_size=options.ads_block_size,
    )


VerticalReflectivityProfileType = v450.VerticalReflectivityProfileType | v445.VerticalReflectivityProfileType
ProcessingParametersL2AType = v450.ProcessingParametersL2AType | v445.ProcessingParametersL2AType


def translate_vertical_reflectivity_profile(
    profile: VerticalReflectivityProfileType,
) -> list[float]:
    return profile.val


def translate_vertical_reflectivity_profile_to_model(
    profile: list[float],
) -> LATEST_VERSION.VerticalReflectivityProfileType:
    return LATEST_VERSION.VerticalReflectivityProfileType(
        val=profile,
        count=len(profile),
    )


def translate_processing_parameters_l2a_fh(
    params: ProcessingParametersL2AType,
) -> main_annotation_l2a_fh.BIOMASSL2aMainADSProcessingParametersFH:
    """Translate FH L2A processing parameters from their model."""
    return main_annotation_l2a_fh.BIOMASSL2aMainADSProcessingParametersFH(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime(params.product_generation_time),
        general_configuration=translate_common_annotation_l2l3.translate_general_configuration(
            params.general_configuration_parameters
        ),
        compression_options=translate_compression_options_l2a_fh(params.compression_options),
        vertical_reflectivity_option=params.vertical_reflectivity_option.value,
        model_inversion=params.model_inversion.value,
        spectral_decorrelation_compensation_flag=translate_common.translate_bool(
            params.spectral_decorrelation_compensation_flag
        ),
        snr_decorrelation_compensation_flag=translate_common.translate_bool(params.snrdecorrelation_compensation),
        correct_terrain_slopes_flag=translate_common.translate_bool(params.correct_terrain_slopes_flag),
        normalised_height_estimation_range=translate_common.translate_min_max_type(
            params.normalised_height_estimation_range
        ),
        normalised_wavenumber_estimation_range=translate_common_annotation_l2l3.translate_min_max_num_type(
            params.normalised_wavenumber_estimation_range
        ),
        ground_to_volume_ratio_range=translate_common_annotation_l2l3.translate_min_max_num_type(
            params.ground_to_volume_ratio_range
        ),
        temporal_decorrelation_estimation_range=translate_common_annotation_l2l3.translate_min_max_num_type(
            params.temporal_decorrelation_estimation_range
        ),
        temporal_decorrelation_ground_to_volume_ratio=params.temporal_decorrelation_ground_to_volume_ratio,
        residual_decorrelation=params.residual_decorrelation,
        product_resolution=translate_common.translate_float_with_unit(params.product_resolution),
        uncertainty_valid_values_limits=translate_common.translate_min_max_type(params.uncertainty_validvalues_limits),
        vertical_wavenumber_valid_values_limits=translate_common.translate_min_max_type(
            params.vertical_wavenumber_validvalues_limits
        ),
        lower_height_limit=translate_common.translate_float_with_unit(params.lower_height_limit),
        upsampling_factor=params.upsampling_factor,
        vertical_reflectivity_default_profile=translate_vertical_reflectivity_profile(
            params.vertical_reflectivity_default_profile
        ),
    )


def translate_processing_parameters_l2a_fh_to_model(
    params: main_annotation_l2a_fh.BIOMASSL2aMainADSProcessingParametersFH,
) -> LATEST_VERSION.ProcessingParametersL2AType:
    """Translate FH L2A processing parameters to their model."""
    return LATEST_VERSION.ProcessingParametersL2AType(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime_to_model(params.product_generation_time),
        general_configuration_parameters=translate_common_annotation_l2l3.translate_general_configuration_to_model(
            params.general_configuration
        ),
        vertical_reflectivity_option=LATEST_VERSION.VerticalProfileOptionType(params.vertical_reflectivity_option),
        vertical_reflectivity_default_profile=translate_vertical_reflectivity_profile_to_model(
            params.vertical_reflectivity_default_profile
        ),
        model_inversion=LATEST_VERSION.ModelInversionType(params.model_inversion),
        spectral_decorrelation_compensation_flag=translate_common.translate_bool_to_model(
            params.spectral_decorrelation_compensation_flag
        ),
        snrdecorrelation_compensation=translate_common.translate_bool_to_model(
            params.snr_decorrelation_compensation_flag
        ),
        correct_terrain_slopes_flag=translate_common.translate_bool_to_model(params.correct_terrain_slopes_flag),
        normalised_height_estimation_range=translate_common.translate_min_max_type_to_model(
            params.normalised_height_estimation_range
        ),
        normalised_wavenumber_estimation_range=translate_common_annotation_l2l3.translate_min_max_num_type_to_model(
            params.normalised_wavenumber_estimation_range
        ),
        ground_to_volume_ratio_range=translate_common_annotation_l2l3.translate_min_max_num_type_to_model(
            params.ground_to_volume_ratio_range
        ),
        temporal_decorrelation_estimation_range=translate_common_annotation_l2l3.translate_min_max_num_type_to_model(
            params.temporal_decorrelation_estimation_range
        ),
        temporal_decorrelation_ground_to_volume_ratio=params.temporal_decorrelation_ground_to_volume_ratio,
        residual_decorrelation=params.residual_decorrelation,
        product_resolution=translate_common.translate_float_with_unit_to_model(
            params.product_resolution, common.UomType.M
        ),
        uncertainty_validvalues_limits=translate_common.translate_min_max_type_to_model(
            params.uncertainty_valid_values_limits
        ),
        vertical_wavenumber_validvalues_limits=translate_common.translate_min_max_type_to_model(
            params.vertical_wavenumber_valid_values_limits
        ),
        lower_height_limit=translate_common.translate_float_with_unit_to_model(
            params.lower_height_limit, common.UomType.M
        ),
        upsampling_factor=params.upsampling_factor,
        compression_options=translate_compression_options_l2a_fh_to_model(params.compression_options),
    )


MainAnnotation = v450.MainAnnotation | v445.MainAnnotation


def translate_main_annotation_l2a_fh(
    annotation: MainAnnotation,
) -> main_annotation_l2a_fh.MainAnnotationL2aFH:
    """Translate an FH L2A main annotation model to its in-memory representation."""
    return main_annotation_l2a_fh.MainAnnotationL2aFH(
        product=translate_common_annotation_l2l3.translate_product_l2a(annotation.product, baseline=0),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2a(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2a_fh(annotation.processing_parameters),
        annotation_lut=translate_common.translate_layer_list(annotation.annotation_lut),
    )


def translate_main_annotation_l2a_fh_to_model(
    annotation: main_annotation_l2a_fh.MainAnnotationL2aFH,
) -> LATEST_VERSION.MainAnnotation:
    """Translate an FH L2A main annotation to its generated model."""
    return LATEST_VERSION.MainAnnotation(
        product=translate_common_annotation_l2l3.translate_product_l2a_to_model(annotation.product),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a_to_model(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2a_to_model(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2a_fh_to_model(annotation.processing_parameters),
        annotation_lut=translate_common.translate_layer_list_to_model(annotation.annotation_lut),
    )


def read_main_annotation_l2a_fh(
    file: Path,
) -> main_annotation_l2a_fh.MainAnnotationL2aFH:
    """Read main annotation of l2a fh products"""
    main_annotation_model = reader_core.reader(
        file,
        "l2a fh",
        [
            ("v450", v450.MainAnnotation),
            ("v445", v445.MainAnnotation),
        ],
    )
    return translate_main_annotation_l2a_fh(main_annotation_model)


def write_main_annotation_l2a_fh(
    file: Path,
    annotation: main_annotation_l2a_fh.MainAnnotationL2aFH,
):
    """Write main annotation of L2A FH products."""
    main_annotation_model = translate_main_annotation_l2a_fh_to_model(annotation)
    file.write_text(serialize(main_annotation_model), encoding="utf-8")
