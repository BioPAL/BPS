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
from bps.common.io.parsing import serialize
from bps.transcoder.io import (
    main_annotation_l2a_gn,
    reader_core,
    translate_common,
    translate_common_annotation_l2l3,
)
from bps.transcoder.io.v445 import main_annotation_models_l2a_gn as v445
from bps.transcoder.io.v450 import main_annotation_models_l2a_gn as v450

CompressionOptionsL2A = v450.CompressionOptionsL2A | v445.CompressionOptionsL2A
ProcessingParametersL2AType = v450.ProcessingParametersL2AType | v445.ProcessingParametersL2AType
MainAnnotation = v450.MainAnnotation | v445.MainAnnotation
LATEST_VERSION = v450


def translate_compression_options_l2a_gn(
    options: CompressionOptionsL2A,
) -> main_annotation_l2a_gn.CompressionOptionsL2AGN:
    if isinstance(options, v450.CompressionOptionsL2A):
        geometric_lut_options = main_annotation_l2a_gn.CompressionFactorLeastSignificantDigitType(
            options.ads.geometric_luts.compression_factor,
            options.ads.geometric_luts.least_significant_digit,
        )
    else:
        assert isinstance(options, v445.CompressionOptionsL2A)
        # The field was renamed from local_incidence_angle to geometric_luts in newer versions
        geometric_lut_options = main_annotation_l2a_gn.CompressionFactorLeastSignificantDigitType(
            options.ads.local_incidence_angle.compression_factor,
            options.ads.local_incidence_angle.least_significant_digit,
        )

    return main_annotation_l2a_gn.CompressionOptionsL2AGN(
        mds=main_annotation_l2a_gn.CompressionOptionsL2AGN.Mds(
            gn=main_annotation_l2a_gn.CompressionFactorMaxZErrorType(
                options.mds.gn.compression_factor, options.mds.gn.max_z_error
            )
        ),
        ads=main_annotation_l2a_gn.CompressionOptionsL2AGN.Ads(
            fnf=main_annotation_l2a_gn.CompressionFactorType(options.ads.fnf.compression_factor),
            geometric_luts=geometric_lut_options,
        ),
        mds_block_size=options.mds_block_size,
        ads_block_size=options.ads_block_size,
    )


def translate_compression_options_l2a_gn_to_model(
    options: main_annotation_l2a_gn.CompressionOptionsL2AGN,
) -> LATEST_VERSION.CompressionOptionsL2A:
    model = LATEST_VERSION.CompressionOptionsL2A
    return model(
        mds=model.Mds(
            gn=model.Mds.Gn(
                compression_factor=options.mds.gn.compression_factor,
                max_z_error=options.mds.gn.max_z_error,
            )
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


def translate_processing_parameters_l2a_gn(
    params: ProcessingParametersL2AType,
) -> main_annotation_l2a_gn.BIOMASSL2aMainADSProcessingParametersGN:
    """Translate GN L2A processing parameters from their model."""
    return main_annotation_l2a_gn.BIOMASSL2aMainADSProcessingParametersGN(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime(params.product_generation_time),
        general_configuration=translate_common_annotation_l2l3.translate_general_configuration(
            params.general_configuration_parameters
        ),
        compression_options=translate_compression_options_l2a_gn(params.compression_options),
        emphasized_forest_height=translate_common.translate_float_with_unit(params.emphasized_forest_height),
        operational_mode=params.operational_mode.value,
        radiometric_calibration_flag=translate_common.translate_bool(params.radiometric_calibration_flag),
        product_resolution=translate_common.translate_float_with_unit(params.product_resolution),
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


def translate_processing_parameters_l2a_gn_to_model(
    params: main_annotation_l2a_gn.BIOMASSL2aMainADSProcessingParametersGN,
) -> LATEST_VERSION.ProcessingParametersL2AType:
    """Translate GN L2A processing parameters to their model."""
    operational_mode = LATEST_VERSION.OperationalModeType(params.operational_mode)
    return LATEST_VERSION.ProcessingParametersL2AType(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime_to_model(params.product_generation_time),
        general_configuration_parameters=translate_common_annotation_l2l3.translate_general_configuration_to_model(
            params.general_configuration
        ),
        emphasized_forest_height=translate_common.translate_float_with_unit_to_model(
            params.emphasized_forest_height, common.UomType.M
        ),
        operational_mode=operational_mode,
        radiometric_calibration_flag=translate_common.translate_bool_to_model(params.radiometric_calibration_flag),
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
        product_resolution=translate_common.translate_float_with_unit_to_model(
            params.product_resolution, common.UomType.M
        ),
        upsampling_factor=params.upsampling_factor,
        compression_options=translate_compression_options_l2a_gn_to_model(params.compression_options),
    )


def translate_main_annotation_l2a_gn(
    annotation: MainAnnotation,
) -> main_annotation_l2a_gn.MainAnnotationL2aGN:
    """Translate a GN L2A main annotation model to its in-memory representation."""
    return main_annotation_l2a_gn.MainAnnotationL2aGN(
        product=translate_common_annotation_l2l3.translate_product_l2a(annotation.product, baseline=0),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2a(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2a_gn(annotation.processing_parameters),
        annotation_lut=translate_common.translate_layer_list(annotation.annotation_lut),
    )


def translate_main_annotation_l2a_gn_to_model(
    annotation: main_annotation_l2a_gn.MainAnnotationL2aGN,
) -> LATEST_VERSION.MainAnnotation:
    """Translate a GN L2A main annotation to its generated model."""
    return LATEST_VERSION.MainAnnotation(
        product=translate_common_annotation_l2l3.translate_product_l2a_to_model(annotation.product),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a_to_model(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2a_to_model(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2a_gn_to_model(annotation.processing_parameters),
        annotation_lut=translate_common.translate_layer_list_to_model(annotation.annotation_lut),
    )


def read_main_annotation_l2a_gn(
    file: Path,
) -> main_annotation_l2a_gn.MainAnnotationL2aGN:
    """Read main annotation of l2a gn products"""
    main_annotation_model = reader_core.reader(
        file,
        "l2a gn",
        [
            ("v450", v450.MainAnnotation),
            ("v445", v445.MainAnnotation),
        ],
    )
    return translate_main_annotation_l2a_gn(main_annotation_model)


def write_main_annotation_l2a_gn(
    file: Path,
    annotation: main_annotation_l2a_gn.MainAnnotationL2aGN,
):
    """Write main annotation of L2A GN products."""
    main_annotation_model = translate_main_annotation_l2a_gn_to_model(annotation)
    file.write_text(serialize(main_annotation_model), encoding="utf-8")
