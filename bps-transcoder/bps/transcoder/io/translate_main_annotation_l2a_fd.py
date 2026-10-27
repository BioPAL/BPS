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

from bps.common.io import common
from bps.common.io.parsing import parse, serialize
from bps.transcoder.io import (
    main_annotation_l2a_fd,
    translate_common,
    translate_common_annotation_l2l3,
)
from bps.transcoder.io.v450 import main_annotation_models_l2a_fd


def translate_compression_options_l2a_fd(
    options: main_annotation_models_l2a_fd.CompressionOptionsL2A,
) -> main_annotation_l2a_fd.CompressionOptionsL2AFD:
    return main_annotation_l2a_fd.CompressionOptionsL2AFD(
        mds=main_annotation_l2a_fd.CompressionOptionsL2AFD.Mds(
            fd=main_annotation_l2a_fd.CompressionFactorType(options.mds.fd.compression_factor),
            probability_ofchange=main_annotation_l2a_fd.CompressionFactorMaxZErrorType(
                options.mds.probability_ofchange.compression_factor,
                options.mds.probability_ofchange.max_z_error,
            ),
            cfm=main_annotation_l2a_fd.CompressionFactorType(options.mds.cfm.compression_factor),
        ),
        ads=main_annotation_l2a_fd.CompressionOptionsL2AFD.Ads(
            fnf=main_annotation_l2a_fd.CompressionFactorType(options.ads.fnf.compression_factor),
            acm=main_annotation_l2a_fd.CompressionFactorLeastSignificantDigitType(
                options.ads.acm.compression_factor,
                options.ads.acm.least_significant_digit,
            ),
            number_of_averages=main_annotation_l2a_fd.CompressionFactorType(
                options.ads.number_of_averages.compression_factor
            ),
            geometric_luts=main_annotation_l2a_fd.CompressionFactorLeastSignificantDigitType(
                options.ads.geometric_luts.compression_factor,
                options.ads.geometric_luts.least_significant_digit,
            ),
        ),
        mds_block_size=options.mds_block_size,
        ads_block_size=options.ads_block_size,
    )


def translate_compression_options_l2a_fd_to_model(
    options: main_annotation_l2a_fd.CompressionOptionsL2AFD,
) -> main_annotation_models_l2a_fd.CompressionOptionsL2A:
    model = main_annotation_models_l2a_fd.CompressionOptionsL2A
    return model(
        mds=model.Mds(
            fd=model.Mds.Fd(compression_factor=options.mds.fd.compression_factor),
            probability_ofchange=model.Mds.ProbabilityOfchange(
                compression_factor=options.mds.probability_ofchange.compression_factor,
                max_z_error=options.mds.probability_ofchange.max_z_error,
            ),
            cfm=model.Mds.Cfm(compression_factor=options.mds.cfm.compression_factor),
        ),
        ads=model.Ads(
            fnf=model.Ads.Fnf(compression_factor=options.ads.fnf.compression_factor),
            acm=model.Ads.Acm(
                compression_factor=options.ads.acm.compression_factor,
                least_significant_digit=options.ads.acm.least_significant_digit,
            ),
            number_of_averages=model.Ads.NumberOfAverages(
                compression_factor=options.ads.number_of_averages.compression_factor
            ),
            geometric_luts=model.Ads.GeometricLuts(
                compression_factor=options.ads.geometric_luts.compression_factor,
                least_significant_digit=options.ads.geometric_luts.least_significant_digit,
            ),
        ),
        mds_block_size=options.mds_block_size,
        ads_block_size=options.ads_block_size,
    )


def translate_processing_parameters_l2a_fd(
    params: main_annotation_models_l2a_fd.ProcessingParametersL2AType,
) -> main_annotation_l2a_fd.BIOMASSL2aMainADSProcessingParametersFD:
    """Translate FD L2A processing parameters from their model."""
    return main_annotation_l2a_fd.BIOMASSL2aMainADSProcessingParametersFD(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime(params.product_generation_time),
        general_configuration=translate_common_annotation_l2l3.translate_general_configuration(
            params.general_configuration_parameters
        ),
        compression_options=translate_compression_options_l2a_fd(params.compression_options),
        emphasized_forest_height=translate_common.translate_float_with_unit(params.emphasized_forest_height),
        operational_mode=params.operational_mode.value,
        significance_level=params.significance_level,
        product_resolution=translate_common.translate_float_with_unit(params.product_resolution),
        numerical_determinant_limit=params.numerical_determinant_limit,
        upsampling_factor=params.upsampling_factor,
        images_pair_selection=(
            translate_common_annotation_l2l3.translate_acquisition_list(params.images_pair_selection)
            if params.images_pair_selection is not None
            else None
        ),
        disable_ground_cancellation_flag=(
            translate_common.translate_bool(params.disable_ground_cancellation_flag)
            if params.disable_ground_cancellation_flag is not None
            else None
        ),
    )


def translate_processing_parameters_l2a_fd_to_model(
    params: main_annotation_l2a_fd.BIOMASSL2aMainADSProcessingParametersFD,
) -> main_annotation_models_l2a_fd.ProcessingParametersL2AType:
    """Translate FD L2A processing parameters to their model."""
    operational_mode = main_annotation_models_l2a_fd.OperationalModeType(params.operational_mode)
    return main_annotation_models_l2a_fd.ProcessingParametersL2AType(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime_to_model(params.product_generation_time),
        general_configuration_parameters=translate_common_annotation_l2l3.translate_general_configuration_to_model(
            params.general_configuration
        ),
        emphasized_forest_height=translate_common.translate_float_with_unit_to_model(
            params.emphasized_forest_height, common.UomType.M
        ),
        operational_mode=operational_mode,
        images_pair_selection=(
            translate_common_annotation_l2l3.translate_acquisition_list_to_model(params.images_pair_selection)
            if operational_mode.name == "INSAR_PAIR" and params.images_pair_selection is not None
            else None
        ),
        disable_ground_cancellation_flag=(
            translate_common.translate_bool_to_model(params.disable_ground_cancellation_flag)
            if params.disable_ground_cancellation_flag is not None
            else None
        ),
        significance_level=params.significance_level,
        product_resolution=translate_common.translate_float_with_unit_to_model(
            params.product_resolution, common.UomType.M
        ),
        numerical_determinant_limit=params.numerical_determinant_limit,
        upsampling_factor=params.upsampling_factor,
        compression_options=translate_compression_options_l2a_fd_to_model(params.compression_options),
    )


def translate_main_annotation_l2a_fd_fd(
    annotation: main_annotation_models_l2a_fd.MainAnnotation,
) -> main_annotation_l2a_fd.MainAnnotationL2aFD:
    """Translate an FD L2A main annotation model to its in-memory representation."""
    return main_annotation_l2a_fd.MainAnnotationL2aFD(
        product=translate_common_annotation_l2l3.translate_product_l2a(annotation.product, baseline=0),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2a(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2a_fd(annotation.processing_parameters),
        annotation_lut=translate_common.translate_layer_list(annotation.annotation_lut),
    )


def translate_main_annotation_l2a_fd_fd_to_model(
    annotation: main_annotation_l2a_fd.MainAnnotationL2aFD,
) -> main_annotation_models_l2a_fd.MainAnnotation:
    """Translate an FD L2A main annotation to its generated model."""
    return main_annotation_models_l2a_fd.MainAnnotation(
        product=translate_common_annotation_l2l3.translate_product_l2a_to_model(annotation.product),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a_to_model(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2a_to_model(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2a_fd_to_model(annotation.processing_parameters),
        annotation_lut=translate_common.translate_layer_list_to_model(annotation.annotation_lut),
    )


def read_main_annotation_l2a_fd(
    file: Path,
) -> main_annotation_l2a_fd.MainAnnotationL2aFD:
    """Read main annotation of l2a products"""
    main_annotation_model: main_annotation_models_l2a_fd.MainAnnotation = parse(
        file.read_text(encoding="utf-8"),
        main_annotation_models_l2a_fd.MainAnnotation,
    )

    return translate_main_annotation_l2a_fd_fd(main_annotation_model)


def write_main_annotation_l2a_fd(
    file: Path,
    annotation: main_annotation_l2a_fd.MainAnnotationL2aFD,
):
    """Write main annotation of l2a products"""
    main_annotation_model = translate_main_annotation_l2a_fd_fd_to_model(annotation)
    main_annotation_text = serialize(main_annotation_model)
    file.write_text(main_annotation_text, encoding="utf-8")
