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
from bps.transcoder.io import main_annotation_l2a_tfh, reader_core, translate_common, translate_common_annotation_l2l3
from bps.transcoder.io.v445 import main_annotation_models_l2a_tfh as v445
from bps.transcoder.io.v450 import main_annotation_models_l2a_tfh as v450

CompressionOptionsL2A = v450.CompressionOptionsL2A | v445.CompressionOptionsL2A
ProcessingParametersL2AType = v450.ProcessingParametersL2AType | v445.ProcessingParametersL2AType
MainAnnotation = v450.MainAnnotation | v445.MainAnnotation
LATEST_VERSION = v450


def translate_compression_options_l2a_tfh(
    options: CompressionOptionsL2A,
) -> main_annotation_l2a_tfh.CompressionOptionsL2ATFH:
    if isinstance(options, v450.CompressionOptionsL2A):
        geometric_lut_options = main_annotation_l2a_tfh.CompressionFactorLeastSignificantDigitType(
            options.ads.geometric_luts.compression_factor,
            options.ads.geometric_luts.least_significant_digit,
        )
    else:
        assert isinstance(options, v445.CompressionOptionsL2A)
        bps_logger.warning("Geometric LUT options are not available for older versions; using default values.")
        geometric_lut_options = main_annotation_l2a_tfh.CompressionFactorLeastSignificantDigitType(
            compression_factor=5, least_significant_digit=5
        )

    return main_annotation_l2a_tfh.CompressionOptionsL2ATFH(
        mds=main_annotation_l2a_tfh.CompressionOptionsL2ATFH.Mds(
            tfh=main_annotation_l2a_tfh.CompressionFactorMaxZErrorType(
                options.mds.tfh.compression_factor, options.mds.tfh.max_z_error
            ),
            quality=main_annotation_l2a_tfh.CompressionFactorMaxZErrorType(
                options.mds.quality.compression_factor, options.mds.quality.max_z_error
            ),
        ),
        ads=main_annotation_l2a_tfh.CompressionOptionsL2ATFH.Ads(
            fnf=main_annotation_l2a_tfh.CompressionFactorType(options.ads.fnf.compression_factor),
            geometric_luts=geometric_lut_options,
        ),
        mds_block_size=options.mds_block_size,
        ads_block_size=options.ads_block_size,
    )


def translate_compression_options_l2a_tfh_to_model(
    options: main_annotation_l2a_tfh.CompressionOptionsL2ATFH,
) -> LATEST_VERSION.CompressionOptionsL2A:
    model = LATEST_VERSION.CompressionOptionsL2A
    return model(
        mds=model.Mds(
            tfh=model.Mds.Tfh(
                compression_factor=options.mds.tfh.compression_factor,
                max_z_error=options.mds.tfh.max_z_error,
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


def translate_processing_parameters_l2a_tfh(
    params: ProcessingParametersL2AType,
) -> main_annotation_l2a_tfh.BIOMASSL2aMainADSProcessingParametersTOMOFH:
    """Translate TFH L2A processing parameters from their model."""
    return main_annotation_l2a_tfh.BIOMASSL2aMainADSProcessingParametersTOMOFH(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime(params.product_generation_time),
        general_configuration=translate_common_annotation_l2l3.translate_general_configuration(
            params.general_configuration_parameters
        ),
        compression_options=translate_compression_options_l2a_tfh(params.compression_options),
        enable_super_resolution=translate_common.translate_bool(params.enable_super_resolution),
        product_resolution=translate_common.translate_float_with_unit(params.product_resolution),
        regularization_noise_factor=params.regularization_noise_factor,
        power_threshold=params.power_threshold,
        median_factor=params.median_factor,
        estimation_valid_values_limits=translate_common.translate_min_max_type_with_unit(
            params.estimation_valid_values_limits
        ),
    )


def translate_processing_parameters_l2a_tfh_to_model(
    params: main_annotation_l2a_tfh.BIOMASSL2aMainADSProcessingParametersTOMOFH,
) -> LATEST_VERSION.ProcessingParametersL2AType:
    """Translate TFH L2A processing parameters to their model."""
    return LATEST_VERSION.ProcessingParametersL2AType(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime_to_model(params.product_generation_time),
        general_configuration_parameters=translate_common_annotation_l2l3.translate_general_configuration_to_model(
            params.general_configuration
        ),
        enable_super_resolution=translate_common.translate_bool_to_model(params.enable_super_resolution),
        product_resolution=translate_common.translate_float_with_unit_to_model(
            params.product_resolution, common.UomType.M
        ),
        regularization_noise_factor=params.regularization_noise_factor,
        power_threshold=params.power_threshold,
        median_factor=params.median_factor,
        estimation_valid_values_limits=translate_common.translate_min_max_type_with_unit_to_model(
            params.estimation_valid_values_limits
        ),
        compression_options=translate_compression_options_l2a_tfh_to_model(params.compression_options),
    )


def translate_main_annotation_l2a_tfh(
    annotation: MainAnnotation,
) -> main_annotation_l2a_tfh.MainAnnotationL2aTFH:
    """Translate a TFH L2A main annotation model to its in-memory representation."""
    return main_annotation_l2a_tfh.MainAnnotationL2aTFH(
        product=translate_common_annotation_l2l3.translate_product_l2a(annotation.product, baseline=0),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2a(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2a_tfh(annotation.processing_parameters),
        annotation_lut=translate_common.translate_layer_list(annotation.annotation_lut),
    )


def translate_main_annotation_l2a_tfh_to_model(
    annotation: main_annotation_l2a_tfh.MainAnnotationL2aTFH,
) -> LATEST_VERSION.MainAnnotation:
    """Translate a TFH L2A main annotation to its generated model."""
    return LATEST_VERSION.MainAnnotation(
        product=translate_common_annotation_l2l3.translate_product_l2a_to_model(annotation.product),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a_to_model(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2a_to_model(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2a_tfh_to_model(annotation.processing_parameters),
        annotation_lut=translate_common.translate_layer_list_to_model(annotation.annotation_lut),
    )


def read_main_annotation_l2a_tfh(
    file: Path,
) -> main_annotation_l2a_tfh.MainAnnotationL2aTFH:
    """Read main annotation of l2a tfh products"""
    main_annotation_model = reader_core.reader(
        file,
        "l2a tfh",
        [
            ("v450", v450.MainAnnotation),
            ("v445", v445.MainAnnotation),
        ],
    )
    return translate_main_annotation_l2a_tfh(main_annotation_model)


def write_main_annotation_l2a_tfh(
    file: Path,
    annotation: main_annotation_l2a_tfh.MainAnnotationL2aTFH,
):
    """Write main annotation of L2A TFH products."""
    main_annotation_model = translate_main_annotation_l2a_tfh_to_model(annotation)
    file.write_text(serialize(main_annotation_model), encoding="utf-8")
