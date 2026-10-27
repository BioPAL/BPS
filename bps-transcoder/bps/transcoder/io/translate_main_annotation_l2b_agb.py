# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Conversions between generated and in-memory L2B AGB annotations."""

from pathlib import Path

from bps.common.io import common
from bps.common.io.parsing import parse, serialize
from bps.transcoder.io import (
    main_annotation_l2b_agb,
    translate_common,
    translate_common_annotation_l2l3,
)
from bps.transcoder.io.v450 import main_annotation_models_l2b_agb


def translate_reference_selection_type(
    value: main_annotation_models_l2b_agb.ReferenceSelectionType,
) -> main_annotation_l2b_agb.ReferenceSelectionType:
    return main_annotation_l2b_agb.ReferenceSelectionType(value.value)


def translate_reference_selection_type_to_model(
    value: main_annotation_l2b_agb.ReferenceSelectionType,
) -> main_annotation_models_l2b_agb.ReferenceSelectionType:
    return main_annotation_models_l2b_agb.ReferenceSelectionType(value.value)


def translate_agb_index(
    value: main_annotation_models_l2b_agb.AgbIndexingType,
) -> main_annotation_l2b_agb.AgbIndexingType:
    return main_annotation_l2b_agb.AgbIndexingType(value.value)


def translate_agb_index_to_model(
    value: main_annotation_l2b_agb.AgbIndexingType,
) -> main_annotation_models_l2b_agb.AgbIndexingType:
    return main_annotation_models_l2b_agb.AgbIndexingType(value.value)


def translate_estimated_parameters_polarisation_type(
    value: main_annotation_models_l2b_agb.EstimatedParametersPolarisationTypePolarisations,
) -> main_annotation_l2b_agb.EstimatedParametersPolarisationType:
    return main_annotation_l2b_agb.EstimatedParametersPolarisationType(value.value)


def translate_estimated_parameters_polarisation_type_to_model(
    value: main_annotation_l2b_agb.EstimatedParametersPolarisationType,
) -> main_annotation_models_l2b_agb.EstimatedParametersPolarisationTypePolarisations:
    return main_annotation_models_l2b_agb.EstimatedParametersPolarisationTypePolarisations(value.value)


def translate_backscatter_limits(
    limits: main_annotation_models_l2b_agb.BackscatterLimitsType,
) -> dict[common.PolarisationType, common.MinMaxType]:
    return {
        common.PolarisationType.HH: translate_common.translate_min_max_type(limits.hh),
        common.PolarisationType.HV: translate_common.translate_min_max_type(limits.vh),
        common.PolarisationType.VV: translate_common.translate_min_max_type(limits.vv),
    }


def translate_backscatter_limits_to_model(
    limits: dict[common.PolarisationType, common.MinMaxType],
) -> main_annotation_models_l2b_agb.BackscatterLimitsType:
    return main_annotation_models_l2b_agb.BackscatterLimitsType(
        hh=translate_common.translate_min_max_type_to_model(limits[common.PolarisationType.HH]),
        vh=translate_common.translate_min_max_type_to_model(limits[common.PolarisationType.HV]),
        vv=translate_common.translate_min_max_type_to_model(limits[common.PolarisationType.VV]),
    )


def translate_estimated_parameters(
    estimated_parameters_model: main_annotation_models_l2b_agb.EstimatedParametersL2BAgb,
) -> main_annotation_l2b_agb.EstimatedParametersL2BAgb:
    return main_annotation_l2b_agb.EstimatedParametersL2BAgb(
        rho=estimated_parameters_model.rho,
        polarisation=[
            main_annotation_l2b_agb.EstimatedParametersPolarisation(
                polarisations=translate_estimated_parameters_polarisation_type(item.polarisations),
                lcm=[
                    main_annotation_l2b_agb.EstimatedParametersLcm(
                        classes=lcm.classes,
                        date=[
                            main_annotation_l2b_agb.EstimatedParametersDate(
                                dates=date.dates,
                                n=main_annotation_l2b_agb.PercentEstimate(mean=date.n.mean, std=date.n.std),
                                a=main_annotation_l2b_agb.PercentEstimate(mean=date.a.mean, std=date.a.std),
                                l=main_annotation_l2b_agb.PercentEstimate(mean=date.l.mean, std=date.l.std),
                            )
                            for date in lcm.date
                        ],
                    )
                    for lcm in item.lcm
                ],
            )
            for item in estimated_parameters_model.polarisation
        ],
    )


def translate_estimated_parameters_to_model(
    estimated_parameters_model: main_annotation_l2b_agb.EstimatedParametersL2BAgb,
) -> main_annotation_models_l2b_agb.EstimatedParametersL2BAgb:
    return main_annotation_models_l2b_agb.EstimatedParametersL2BAgb(
        rho=estimated_parameters_model.rho,
        polarisation=[
            main_annotation_models_l2b_agb.EstimatedParametersPolarisationType(
                polarisations=translate_estimated_parameters_polarisation_type_to_model(item.polarisations),
                lcm=[
                    main_annotation_models_l2b_agb.EstimatedParametersPolarisationType.Lcm(
                        classes=lcm.classes,
                        date=[
                            main_annotation_models_l2b_agb.EstimatedParametersPolarisationType.Lcm.Date(
                                dates=date.dates,
                                n=main_annotation_models_l2b_agb.EstimatedParametersPolarisationType.Lcm.Date.N(
                                    mean=date.n.mean, std=date.n.std
                                ),
                                a=main_annotation_models_l2b_agb.EstimatedParametersPolarisationType.Lcm.Date.A(
                                    mean=date.a.mean, std=date.a.std
                                ),
                                l=main_annotation_models_l2b_agb.EstimatedParametersPolarisationType.Lcm.Date.L(
                                    mean=date.l.mean, std=date.l.std
                                ),
                            )
                            for date in lcm.date
                        ],
                    )
                    for lcm in item.lcm
                ],
            )
            for item in estimated_parameters_model.polarisation
        ],
    )


def translate_compression_options_l2b_agb(
    compression_options: main_annotation_models_l2b_agb.CompressionOptionsL2B,
) -> main_annotation_l2b_agb.CompressionOptionsL2B:
    """Translate L2B AGB compression options from the model to the in-memory representation."""
    return main_annotation_l2b_agb.CompressionOptionsL2B(
        mds=main_annotation_l2b_agb.CompressionOptionsL2B.Mds(
            agb=main_annotation_l2b_agb.CompressionFactorMaxZErrorType(
                compression_options.mds.agb.compression_factor,
                compression_options.mds.agb.max_z_error,
            ),
            agbstandard_deviation=main_annotation_l2b_agb.CompressionFactorMaxZErrorType(
                compression_options.mds.agbstandard_deviation.compression_factor,
                compression_options.mds.agbstandard_deviation.max_z_error,
            ),
            bps_fnf=main_annotation_l2b_agb.CompressionFactorType(compression_options.mds.bps_fnf.compression_factor),
            heat_map=main_annotation_l2b_agb.CompressionFactorType(compression_options.mds.heat_map.compression_factor),
            acquisition_id_image=main_annotation_l2b_agb.CompressionFactorType(
                compression_options.mds.acquisition_id_image.compression_factor
            ),
        ),
        mds_block_size=compression_options.mds_block_size,
    )


def translate_compression_options_l2b_agb_to_model(
    compression_options: main_annotation_l2b_agb.CompressionOptionsL2B,
) -> main_annotation_models_l2b_agb.CompressionOptionsL2B:
    return main_annotation_models_l2b_agb.CompressionOptionsL2B(
        mds=main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds(
            agb=main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds.Agb(
                compression_factor=compression_options.mds.agb.compression_factor,
                max_z_error=compression_options.mds.agb.max_z_error,
            ),
            agbstandard_deviation=main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds.AgbstandardDeviation(
                compression_factor=compression_options.mds.agbstandard_deviation.compression_factor,
                max_z_error=compression_options.mds.agbstandard_deviation.max_z_error,
            ),
            bps_fnf=main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds.BpsFnf(
                compression_factor=compression_options.mds.bps_fnf.compression_factor
            ),
            heat_map=main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds.HeatMap(
                compression_factor=compression_options.mds.heat_map.compression_factor
            ),
            acquisition_id_image=main_annotation_models_l2b_agb.CompressionOptionsL2B.Mds.AcquisitionIdImage(
                compression_factor=compression_options.mds.acquisition_id_image.compression_factor
            ),
        ),
        mds_block_size=compression_options.mds_block_size,
    )


def translate_processing_parameters_l2b_agb(
    processing_model: main_annotation_models_l2b_agb.ProcessingParametersL2BType,
) -> main_annotation_l2b_agb.ProcessingParametersAnnotation:
    """Translate L2B AGB processing parameters from the model to the in-memory representation."""
    estimated_parameters_model = processing_model.estimated_parameters
    compression_options_model = processing_model.compression_options
    return main_annotation_l2b_agb.ProcessingParametersAnnotation(
        processor_version=processing_model.processor_version,
        product_generation_time=translate_common.translate_datetime(processing_model.product_generation_time),
        forest_masking_flag=translate_common.translate_bool(processing_model.forest_masking_flag),
        minimum_l2a_coverage=processing_model.minimum_l2a_coverage,
        rejected_landcover_classes=translate_common.translate_int_array(processing_model.rejected_landcover_classes),
        backscatter_limits=translate_backscatter_limits(processing_model.backscatter_limits),
        angle_limits=translate_common.translate_min_max_type_with_unit(processing_model.angle_limits),
        mean_agblimits=translate_common.translate_min_max_type_with_unit(processing_model.mean_agblimits),
        std_agblimits=translate_common.translate_min_max_type_with_unit(processing_model.std_agblimits),
        relative_agblimits=translate_common.translate_min_max_type(processing_model.relative_agblimits),
        reference_selection=translate_reference_selection_type(processing_model.reference_selection),
        indexing_l=translate_agb_index(processing_model.indexing_l),
        indexing_a=translate_agb_index(processing_model.indexing_a),
        indexing_n=translate_agb_index(processing_model.indexing_n),
        use_constant_n=translate_common.translate_bool(processing_model.use_constant_n),
        values_constant_n=translate_common.translate_float_array(processing_model.values_constant_n),
        regression_solver=processing_model.regression_solver,
        regression_matrix_subsampling_factor=processing_model.regression_matrix_subsampling_factor,
        minimum_percentage_of_fillable_voids=processing_model.minimum_percentage_of_fillable_voids,
        estimated_parameters=translate_estimated_parameters(estimated_parameters_model),
        compression_options=translate_compression_options_l2b_agb(compression_options_model),
    )


def translate_processing_parameters_l2b_agb_to_model(
    processing_model: main_annotation_l2b_agb.ProcessingParametersAnnotation,
) -> main_annotation_models_l2b_agb.ProcessingParametersL2BType:
    """Translate L2B AGB processing parameters from the model to the in-memory representation."""
    return main_annotation_models_l2b_agb.ProcessingParametersL2BType(
        processor_version=processing_model.processor_version,
        product_generation_time=translate_common.translate_datetime_to_model(processing_model.product_generation_time),
        forest_masking_flag=translate_common.translate_bool_to_model(processing_model.forest_masking_flag),
        minimum_l2a_coverage=processing_model.minimum_l2a_coverage,
        rejected_landcover_classes=translate_common.translate_int_array_to_model(
            processing_model.rejected_landcover_classes
        ),
        backscatter_limits=translate_backscatter_limits_to_model(processing_model.backscatter_limits),
        angle_limits=translate_common.translate_min_max_type_with_unit_to_model(processing_model.angle_limits),
        mean_agblimits=translate_common.translate_min_max_type_with_unit_to_model(processing_model.mean_agblimits),
        std_agblimits=translate_common.translate_min_max_type_with_unit_to_model(processing_model.std_agblimits),
        relative_agblimits=translate_common.translate_min_max_type_to_model(processing_model.relative_agblimits),
        reference_selection=translate_reference_selection_type_to_model(processing_model.reference_selection),
        indexing_l=translate_agb_index_to_model(processing_model.indexing_l),
        indexing_a=translate_agb_index_to_model(processing_model.indexing_a),
        indexing_n=translate_agb_index_to_model(processing_model.indexing_n),
        use_constant_n=translate_common.translate_bool_to_model(processing_model.use_constant_n),
        values_constant_n=translate_common.translate_float_array_to_model(processing_model.values_constant_n),
        regression_solver=processing_model.regression_solver,
        regression_matrix_subsampling_factor=processing_model.regression_matrix_subsampling_factor,
        minimum_percentage_of_fillable_voids=processing_model.minimum_percentage_of_fillable_voids,
        estimated_parameters=translate_estimated_parameters_to_model(processing_model.estimated_parameters),
        compression_options=translate_compression_options_l2b_agb_to_model(processing_model.compression_options),
    )


def translate_main_annotation_l2b_agb(
    annotation: main_annotation_models_l2b_agb.MainAnnotation,
) -> main_annotation_l2b_agb.MainAnnotationL2bAGB:
    return main_annotation_l2b_agb.MainAnnotationL2bAGB(
        product=translate_common_annotation_l2l3.translate_productl2bl3(annotation.product),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2bl3_list(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2b_agb(annotation.processing_parameters),
    )


def translate_main_annotation_l2b_agb_to_model(
    annotation: main_annotation_l2b_agb.MainAnnotationL2bAGB,
) -> main_annotation_models_l2b_agb.MainAnnotation:
    return main_annotation_models_l2b_agb.MainAnnotation(
        product=translate_common_annotation_l2l3.translate_productl2bl3_to_model(annotation.product),
        raster_image=translate_common_annotation_l2l3.translate_raster_image_l2a_to_model(annotation.raster_image),
        input_information=translate_common_annotation_l2l3.translate_input_information_l2bl3_list_to_model(
            annotation.input_information
        ),
        processing_parameters=translate_processing_parameters_l2b_agb_to_model(annotation.processing_parameters),
    )


def read_main_annotation_l2b_agb(annotation_path: Path) -> main_annotation_l2b_agb.MainAnnotationL2bAGB:
    main_annotation_model: main_annotation_models_l2b_agb.MainAnnotation = parse(
        annotation_path.read_text(encoding="utf-8"), main_annotation_models_l2b_agb.MainAnnotation
    )
    return translate_main_annotation_l2b_agb(main_annotation_model)


def write_main_annotation_l2b_agb(file: Path, annotation: main_annotation_l2b_agb.MainAnnotationL2bAGB) -> None:
    main_annotation_model = translate_main_annotation_l2b_agb_to_model(annotation)
    file.write_text(serialize(main_annotation_model), encoding="utf-8")
