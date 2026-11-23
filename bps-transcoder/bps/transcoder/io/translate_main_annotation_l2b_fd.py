# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Conversions between generated and in-memory L2B FD annotations."""

from pathlib import Path

from bps.common.io.parsing import serialize
from bps.transcoder.io import (
    main_annotation_l2b_fd,
    reader_core,
    translate_common,
    translate_common_annotation_l2l3,
)
from bps.transcoder.io.v445 import main_annotation_models_l2b_fd as v445
from bps.transcoder.io.v450 import main_annotation_models_l2b_fd as v450

CompressionOptionsL2B = v450.CompressionOptionsL2B | v445.CompressionOptionsL2B
ProcessingParametersL2BType = v450.ProcessingParametersL2BType | v445.ProcessingParametersL2BType
MainAnnotation = v450.MainAnnotation | v445.MainAnnotation
LATEST_VERSION = v450


def translate_compression_options_l2b_fd(
    compression_options: CompressionOptionsL2B,
) -> main_annotation_l2b_fd.CompressionOptionsL2B:
    return main_annotation_l2b_fd.CompressionOptionsL2B(
        mds=main_annotation_l2b_fd.CompressionOptionsL2B.Mds(
            fd=main_annotation_l2b_fd.CompressionFactorType(compression_options.mds.fd.compression_factor),
            probability_of_change=main_annotation_l2b_fd.CompressionFactorMaxZErrorType(
                compression_options.mds.probability_of_change.compression_factor,
                compression_options.mds.probability_of_change.max_z_error,
            ),
            cfm=main_annotation_l2b_fd.CompressionFactorType(compression_options.mds.cfm.compression_factor),
            heat_map=main_annotation_l2b_fd.CompressionFactorType(compression_options.mds.heat_map.compression_factor),
            acquisition_id_image=main_annotation_l2b_fd.CompressionFactorType(
                compression_options.mds.acquisition_id_image.compression_factor
            ),
        ),
        mds_block_size=compression_options.mds_block_size,
    )


def translate_compression_options_l2b_fd_to_model(
    compression_options: main_annotation_l2b_fd.CompressionOptionsL2B,
) -> LATEST_VERSION.CompressionOptionsL2B:
    return LATEST_VERSION.CompressionOptionsL2B(
        mds=LATEST_VERSION.CompressionOptionsL2B.Mds(
            fd=LATEST_VERSION.CompressionOptionsL2B.Mds.Fd(
                compression_factor=compression_options.mds.fd.compression_factor
            ),
            probability_of_change=LATEST_VERSION.CompressionOptionsL2B.Mds.ProbabilityOfChange(
                compression_factor=compression_options.mds.probability_of_change.compression_factor,
                max_z_error=compression_options.mds.probability_of_change.max_z_error,
            ),
            cfm=LATEST_VERSION.CompressionOptionsL2B.Mds.Cfm(
                compression_factor=compression_options.mds.cfm.compression_factor
            ),
            heat_map=LATEST_VERSION.CompressionOptionsL2B.Mds.HeatMap(
                compression_factor=compression_options.mds.heat_map.compression_factor
            ),
            acquisition_id_image=LATEST_VERSION.CompressionOptionsL2B.Mds.AcquisitionIdImage(
                compression_factor=compression_options.mds.acquisition_id_image.compression_factor
            ),
        ),
        mds_block_size=compression_options.mds_block_size,
    )


def translate_processing_parameters_l2b_fd(
    processing_model: ProcessingParametersL2BType,
) -> main_annotation_l2b_fd.ProcessingParametersAnnotation:
    return main_annotation_l2b_fd.ProcessingParametersAnnotation(
        processor_version=processing_model.processor_version,
        product_generation_time=translate_common.translate_datetime(processing_model.product_generation_time),
        minimum_l2a_coverage=processing_model.minimum_l2a_coverage,
        compression_options=translate_compression_options_l2b_fd(processing_model.compression_options),
    )


def translate_processing_parameters_l2b_fd_to_model(
    processing_model: main_annotation_l2b_fd.ProcessingParametersAnnotation,
) -> LATEST_VERSION.ProcessingParametersL2BType:
    return LATEST_VERSION.ProcessingParametersL2BType(
        processor_version=processing_model.processor_version,
        product_generation_time=translate_common.translate_datetime_to_model(processing_model.product_generation_time),
        minimum_l2a_coverage=processing_model.minimum_l2a_coverage,
        compression_options=translate_compression_options_l2b_fd_to_model(processing_model.compression_options),
    )


def translate_main_annotation_l2b_fd(
    annotation: MainAnnotation,
) -> main_annotation_l2b_fd.MainAnnotationL2bFD:
    return main_annotation_l2b_fd.MainAnnotationL2bFD(
        product=translate_common_annotation_l2l3.translate_productl2bl3(annotation.product),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2bl3_list(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2b_fd(annotation.processing_parameters),
    )


def translate_main_annotation_l2b_fd_to_model(
    annotation: main_annotation_l2b_fd.MainAnnotationL2bFD,
) -> LATEST_VERSION.MainAnnotation:
    return LATEST_VERSION.MainAnnotation(
        product=translate_common_annotation_l2l3.translate_productl2bl3_to_model(annotation.product),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a_to_model(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2bl3_list_to_model(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2b_fd_to_model(annotation.processing_parameters),
    )


def read_main_annotation_l2b_fd(
    file: Path,
) -> main_annotation_l2b_fd.MainAnnotationL2bFD:
    """Read main annotation of l2b fd products"""
    main_annotation_model = reader_core.reader(
        file,
        "l2b fd",
        [
            ("v450", v450.MainAnnotation),
            ("v445", v445.MainAnnotation),
        ],
    )
    return translate_main_annotation_l2b_fd(main_annotation_model)


def write_main_annotation_l2b_fd(file: Path, annotation: main_annotation_l2b_fd.MainAnnotationL2bFD) -> None:
    main_annotation_model = translate_main_annotation_l2b_fd_to_model(annotation)
    file.write_text(serialize(main_annotation_model), encoding="utf-8")
