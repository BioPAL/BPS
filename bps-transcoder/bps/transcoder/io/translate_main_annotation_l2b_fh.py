# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Conversions between generated and in-memory L2B FH annotations."""

from pathlib import Path

from bps.common.io.parsing import parse, serialize
from bps.transcoder.io import main_annotation_l2b_fh, translate_common, translate_common_annotation_l2l3
from bps.transcoder.io.v450 import main_annotation_models_l2b_fh


def translate_compression_options_l2b_fh(
    compression_options: main_annotation_models_l2b_fh.CompressionOptionsL2B,
) -> main_annotation_l2b_fh.CompressionOptionsL2B:
    return main_annotation_l2b_fh.CompressionOptionsL2B(
        mds=main_annotation_l2b_fh.CompressionOptionsL2B.Mds(
            fh=main_annotation_l2b_fh.CompressionFactorMaxZErrorType(
                compression_options.mds.fh.compression_factor,
                compression_options.mds.fh.max_z_error,
            ),
            quality=main_annotation_l2b_fh.CompressionFactorMaxZErrorType(
                compression_options.mds.quality.compression_factor,
                compression_options.mds.quality.max_z_error,
            ),
            bps_fnf=main_annotation_l2b_fh.CompressionFactorType(compression_options.mds.bps_fnf.compression_factor),
            heat_map=main_annotation_l2b_fh.CompressionFactorMaxZErrorType(
                compression_options.mds.heat_map.compression_factor,
                compression_options.mds.heat_map.max_z_error,
            ),
            acquisition_id_image=main_annotation_l2b_fh.CompressionFactorType(
                compression_options.mds.acquisition_id_image.compression_factor
            ),
        ),
        mds_block_size=compression_options.mds_block_size,
    )


def translate_compression_options_l2b_fh_to_model(
    compression_options: main_annotation_l2b_fh.CompressionOptionsL2B,
) -> main_annotation_models_l2b_fh.CompressionOptionsL2B:
    return main_annotation_models_l2b_fh.CompressionOptionsL2B(
        mds=main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds(
            fh=main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds.Fh(
                compression_factor=compression_options.mds.fh.compression_factor,
                max_z_error=compression_options.mds.fh.max_z_error,
            ),
            quality=main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds.Quality(
                compression_factor=compression_options.mds.quality.compression_factor,
                max_z_error=compression_options.mds.quality.max_z_error,
            ),
            bps_fnf=main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds.BpsFnf(
                compression_factor=compression_options.mds.bps_fnf.compression_factor
            ),
            heat_map=main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds.HeatMap(
                compression_factor=compression_options.mds.heat_map.compression_factor,
                max_z_error=compression_options.mds.heat_map.max_z_error,
            ),
            acquisition_id_image=main_annotation_models_l2b_fh.CompressionOptionsL2B.Mds.AcquisitionIdImage(
                compression_factor=compression_options.mds.acquisition_id_image.compression_factor
            ),
        ),
        mds_block_size=compression_options.mds_block_size,
    )


def translate_processing_parameters_l2b_fh(
    params: main_annotation_models_l2b_fh.ProcessingParametersL2BType,
) -> main_annotation_l2b_fh.ProcessingParametersAnnotation:
    return main_annotation_l2b_fh.ProcessingParametersAnnotation(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime(params.product_generation_time),
        forest_masking_flag=translate_common.translate_bool(params.forest_masking_flag),
        bps_fnf=translate_common_annotation_l2l3.translate_bps_fnf_type(params.bps_fnf),
        minimum_l2a_coverage=params.minimum_l2a_coverage,
        roll_off_factor_azimuth=params.roll_off_factor_azimuth,
        roll_off_factor_range=params.roll_off_factor_range,
        compression_options=translate_compression_options_l2b_fh(params.compression_options),
    )


def translate_processing_parameters_l2b_fh_to_model(
    params: main_annotation_l2b_fh.ProcessingParametersAnnotation,
) -> main_annotation_models_l2b_fh.ProcessingParametersL2BType:
    return main_annotation_models_l2b_fh.ProcessingParametersL2BType(
        processor_version=params.processor_version,
        product_generation_time=translate_common.translate_datetime_to_model(params.product_generation_time),
        forest_masking_flag=translate_common.translate_bool_to_model(params.forest_masking_flag),
        bps_fnf=translate_common_annotation_l2l3.translate_bps_fnf_type_to_model(params.bps_fnf),
        minimum_l2a_coverage=params.minimum_l2a_coverage,
        roll_off_factor_azimuth=params.roll_off_factor_azimuth,
        roll_off_factor_range=params.roll_off_factor_range,
        compression_options=translate_compression_options_l2b_fh_to_model(params.compression_options),
    )


def translate_main_annotation_l2b_fh(
    annotation: main_annotation_models_l2b_fh.MainAnnotation,
) -> main_annotation_l2b_fh.MainAnnotationL2bFH:
    return main_annotation_l2b_fh.MainAnnotationL2bFH(
        product=translate_common_annotation_l2l3.translate_productl2bl3(annotation.product),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2bl3_list(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2b_fh(annotation.processing_parameters),
    )


def translate_main_annotation_l2b_fh_to_model(
    annotation: main_annotation_l2b_fh.MainAnnotationL2bFH,
) -> main_annotation_models_l2b_fh.MainAnnotation:
    return main_annotation_models_l2b_fh.MainAnnotation(
        product=translate_common_annotation_l2l3.translate_productl2bl3_to_model(annotation.product),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a_to_model(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2bl3_list_to_model(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2b_fh_to_model(annotation.processing_parameters),
    )


def read_main_annotation_l2b_fh(annotation_path: Path) -> main_annotation_l2b_fh.MainAnnotationL2bFH:
    main_annotation_model: main_annotation_models_l2b_fh.MainAnnotation = parse(
        annotation_path.read_text(encoding="utf-8"), main_annotation_models_l2b_fh.MainAnnotation
    )
    return translate_main_annotation_l2b_fh(main_annotation_model)


def write_main_annotation_l2b_fh(file: Path, annotation: main_annotation_l2b_fh.MainAnnotationL2bFH) -> None:
    main_annotation_model = translate_main_annotation_l2b_fh_to_model(annotation)
    file.write_text(serialize(main_annotation_model), encoding="utf-8")
