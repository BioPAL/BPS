# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Common annotations l2l3 translate
---------------------------------
"""

from bps.common.io import common
from bps.transcoder.io import common_annotation_l2l3, translate_common
from bps.transcoder.io.v445 import common_annotation_models_l2 as v445
from bps.transcoder.io.v450 import common_annotation_models_l2 as v450

BpsFnfType = v450.BpsFnfType | v445.BpsFnfType
StringListType = v450.StringListType | v445.StringListType
PixelRepresentationChoiceType = v450.PixelRepresentationChoiceType | v445.PixelRepresentationChoiceType
PixelTypeChoiceType = v450.PixelTypeChoiceType | v445.PixelTypeChoiceType
NoDataValueChoiceType = v450.NoDataValueChoiceType | v445.NoDataValueChoiceType
PolarisationListType = v450.PolarisationListType | v445.PolarisationListType
AcquisitionType = v450.AcquisitionType | v445.AcquisitionType
AcquisitionListType = v450.AcquisitionListType | v445.AcquisitionListType
GeneralConfigurationParametersType = v450.GeneralConfigurationParametersType | v445.GeneralConfigurationParametersType
ProductL2AType = v450.ProductL2AType | v445.ProductL2AType
RasterImageType = v450.RasterImageType | v445.RasterImageType
InputInformationL2AType = v450.InputInformationL2AType | v445.InputInformationL2AType
InputInformationL2BL3ListType = v450.InputInformationL2BL3ListType | v445.InputInformationL2BL3ListType
CalAbcoverageType = v450.CalAbcoverageType | v445.CalAbcoverageType
PercentPixelsType = v450.PercentPixelsType | v445.PercentPixelsType
CalAbfilteredCoverageType = v450.CalAbfilteredCoverageType | v445.CalAbfilteredCoverageType
GncoverageListType = v450.GncoverageListType | v445.GncoverageListType
MinMaxNumType = v450.MinMaxNumType | v445.MinMaxNumType
ProductL2BL3Type = v450.ProductL2BL3Type | v445.ProductL2BL3Type
GncoverageTilesListType = v450.GncoverageTilesListType | v445.GncoverageTilesListType
CalAbcoverageTilesListType = v450.CalAbcoverageTilesListType | v445.CalAbcoverageTilesListType

LATEST_VERSION = v450


def translate_bps_fnf_type(
    bps_fnf: BpsFnfType,
) -> common_annotation_l2l3.BpsFnfType:
    return common_annotation_l2l3.BpsFnfType(bps_fnf.value)


def translate_bps_fnf_type_to_model(
    bps_fnf: common_annotation_l2l3.BpsFnfType,
) -> LATEST_VERSION.BpsFnfType:
    return LATEST_VERSION.BpsFnfType(bps_fnf.value)


def translate_stringlist(value: StringListType) -> list[str]:
    return value.id


def translate_stringlist_to_model(value: list[str]) -> LATEST_VERSION.StringListType:
    return LATEST_VERSION.StringListType(id=value)


def translate_optional_stringlist(value: StringListType | None) -> list[str]:
    if value is None:
        return []
    return translate_stringlist(value)


def translate_optional_stringlist_to_model(value: list[str]) -> LATEST_VERSION.StringListType | None:
    if not value:
        return None
    return translate_stringlist_to_model(value)


def translate_pixel_representation_choice(
    choice: PixelRepresentationChoiceType,
) -> common_annotation_l2l3.PixelRepresentationChoiceType:
    return common_annotation_l2l3.PixelRepresentationChoiceType(
        **{
            field_name: (translate_common.translate_pixel_representation_type(value) if value is not None else None)
            for field_name, value in vars(choice).items()
        }
    )


def translate_pixel_representation_choice_to_model(
    choice: common_annotation_l2l3.PixelRepresentationChoiceType,
) -> LATEST_VERSION.PixelRepresentationChoiceType:
    return LATEST_VERSION.PixelRepresentationChoiceType(
        **{
            field_name: (
                translate_common.translate_pixel_representation_type_to_model(value) if value is not None else None
            )
            for field_name, value in vars(choice).items()
        }
    )


def translate_pixel_type_choice(
    choice: PixelTypeChoiceType,
) -> common_annotation_l2l3.PixelTypeChoiceType:
    return common_annotation_l2l3.PixelTypeChoiceType(
        **{
            field_name: translate_common.translate_pixel_type_type(value) if value is not None else None
            for field_name, value in vars(choice).items()
        }
    )


def translate_pixel_type_choice_to_model(
    choice: common_annotation_l2l3.PixelTypeChoiceType,
) -> LATEST_VERSION.PixelTypeChoiceType:
    return LATEST_VERSION.PixelTypeChoiceType(
        **{
            field_name: translate_common.translate_pixel_type_type_to_model(value) if value is not None else None
            for field_name, value in vars(choice).items()
        }
    )


def translate_no_data_value_choice(
    choice: NoDataValueChoiceType,
) -> common_annotation_l2l3.NoDataValueChoiceType:
    return common_annotation_l2l3.NoDataValueChoiceType(
        float_no_data_value=choice.float_no_data_value,
        int_no_data_value=choice.int_no_data_value,
    )


def translate_no_data_value_choice_to_model(
    choice: common_annotation_l2l3.NoDataValueChoiceType,
) -> LATEST_VERSION.NoDataValueChoiceType:
    return LATEST_VERSION.NoDataValueChoiceType(
        float_no_data_value=choice.float_no_data_value,
        int_no_data_value=choice.int_no_data_value,
    )


def translate_polarisation_list(
    polarisation_list: PolarisationListType,
) -> list[common.PolarisationType]:
    if len(polarisation_list.polarisation) != polarisation_list.count:
        raise RuntimeError(
            "Inconsistency in polarisation list: "
            + f"{len(polarisation_list.polarisation)} polarisations and count: {polarisation_list.count} do not match"
        )
    return [translate_common.translate_polarisation_type(value) for value in polarisation_list.polarisation]


def translate_polarisation_list_to_model(
    polarisation_list: list[common.PolarisationType],
) -> LATEST_VERSION.PolarisationListType:
    model_polarisations = [translate_common.translate_polarisation_type_to_model(value) for value in polarisation_list]
    return LATEST_VERSION.PolarisationListType(
        polarisation=model_polarisations,
        count=len(model_polarisations),
    )


def translate_acquisition_type(
    acquisition: AcquisitionType,
) -> common_annotation_l2l3.AcquisitionType:
    quality_parameters_list = acquisition.sta_quality.sta_quality_parameters_list
    if len(quality_parameters_list.sta_quality_parameters) != quality_parameters_list.count:
        raise RuntimeError(
            "Inconsistency in STA quality parameters: "
            + f"{len(quality_parameters_list.sta_quality_parameters)} entries and "
            + f"count: {quality_parameters_list.count} do not match"
        )

    return common_annotation_l2l3.AcquisitionType(
        folder_name=acquisition.folder_name,
        sta_quality=common_annotation_l2l3.StaQualityType(
            overall_product_quality_index=acquisition.sta_quality.overall_product_quality_index,
            sta_quality_parameters_list=[
                common_annotation_l2l3.StaQualityParametersType(
                    invalid_l1a_data_samples=parameters.invalid_l1a_data_samples,
                    rfi_decorrelation=parameters.rfi_decorrelation,
                    rfi_decorrelation_threshold=parameters.rfi_decorrelation_threshold,
                    faraday_decorrelation=parameters.faraday_decorrelation,
                    faraday_decorrelation_threshold=parameters.faraday_decorrelation_threshold,
                    invalid_residual_shifts_fraction=parameters.invalid_residual_shifts_fraction,
                    residual_shifts_quality_threshold=parameters.residual_shifts_quality_threshold,
                    invalid_ground_phases_screen_estimates_fraction=(
                        parameters.invalid_ground_phases_screen_estimates_fraction
                    ),
                    ground_phases_screen_quality_threshold=parameters.ground_phases_screen_quality_threshold,
                    skp_decomposition_index=parameters.skp_decomposition_index,
                    polarisation=translate_common.translate_polarisation_type(parameters.polarisation),
                )
                for parameters in quality_parameters_list.sta_quality_parameters
            ],
        ),
        reference_image=translate_common.translate_bool(acquisition.reference_image),
        average_wavenumber=acquisition.average_wavenumber,
    )


def translate_acquisition_type_to_model(
    acquisition: common_annotation_l2l3.AcquisitionType,
) -> LATEST_VERSION.AcquisitionType:
    quality_parameters = [
        LATEST_VERSION.StaQualityParametersType(
            invalid_l1a_data_samples=parameters.invalid_l1a_data_samples,
            rfi_decorrelation=parameters.rfi_decorrelation,
            rfi_decorrelation_threshold=parameters.rfi_decorrelation_threshold,
            faraday_decorrelation=parameters.faraday_decorrelation,
            faraday_decorrelation_threshold=parameters.faraday_decorrelation_threshold,
            invalid_residual_shifts_fraction=parameters.invalid_residual_shifts_fraction,
            residual_shifts_quality_threshold=parameters.residual_shifts_quality_threshold,
            invalid_ground_phases_screen_estimates_fraction=(
                parameters.invalid_ground_phases_screen_estimates_fraction
            ),
            ground_phases_screen_quality_threshold=parameters.ground_phases_screen_quality_threshold,
            skp_decomposition_index=parameters.skp_decomposition_index,
            polarisation=translate_common.translate_polarisation_type_to_model(parameters.polarisation),
        )
        for parameters in acquisition.sta_quality.sta_quality_parameters_list
    ]
    model_quality_parameters_list = LATEST_VERSION.StaQualityParametersListType(
        sta_quality_parameters=quality_parameters,
        count=len(quality_parameters),
    )
    return LATEST_VERSION.AcquisitionType(
        folder_name=acquisition.folder_name,
        sta_quality=LATEST_VERSION.StaQualityType(
            overall_product_quality_index=acquisition.sta_quality.overall_product_quality_index,
            sta_quality_parameters_list=model_quality_parameters_list,
        ),
        reference_image=translate_common.translate_bool_to_model(acquisition.reference_image),
        average_wavenumber=acquisition.average_wavenumber,
    )


def translate_acquisition_list(
    acquisition_list: AcquisitionListType,
) -> list[common_annotation_l2l3.AcquisitionType]:
    if len(acquisition_list.acquisition) != acquisition_list.count:
        raise RuntimeError(
            "Inconsistency in acquisition list: "
            + f"{len(acquisition_list.acquisition)} acquisitions and count: {acquisition_list.count} do not match"
        )

    return [translate_acquisition_type(acquisition) for acquisition in acquisition_list.acquisition]


def translate_acquisition_list_to_model(
    acquisition_list: list[common_annotation_l2l3.AcquisitionType],
) -> LATEST_VERSION.AcquisitionListType:
    model_acquisitions = [translate_acquisition_type_to_model(acquisition) for acquisition in acquisition_list]

    return LATEST_VERSION.AcquisitionListType(
        acquisition=model_acquisitions,
        count=len(model_acquisitions),
    )


def translate_general_configuration(
    configuration: GeneralConfigurationParametersType,
) -> common_annotation_l2l3.GeneralConfigurationParametersType:
    """Translate shared L2 processing configuration to its in-memory type."""
    return common_annotation_l2l3.GeneralConfigurationParametersType(
        apply_calibration_screen=common_annotation_l2l3.CalibrationScreenType(
            configuration.apply_calibration_screen.value
        ),
        forest_coverage_threshold=configuration.forest_coverage_threshold,
        forest_mask_interpolation_threshold=configuration.forest_mask_interpolation_threshold,
        subsetting_rule=common_annotation_l2l3.SubsettingRuleType(configuration.subsetting_rule.value),
        polarisation_combination_method=translate_common.translate_polarisation_combination_method(
            configuration.polarisation_combination_method
        ),
    )


def translate_general_configuration_to_model(
    configuration: common_annotation_l2l3.GeneralConfigurationParametersType,
) -> LATEST_VERSION.GeneralConfigurationParametersType:
    """Translate shared in-memory L2 processing configuration to its model."""
    return LATEST_VERSION.GeneralConfigurationParametersType(
        apply_calibration_screen=LATEST_VERSION.CalibrationScreenType(configuration.apply_calibration_screen.value),
        forest_coverage_threshold=configuration.forest_coverage_threshold,
        forest_mask_interpolation_threshold=configuration.forest_mask_interpolation_threshold,
        subsetting_rule=LATEST_VERSION.SubsettingRuleType(configuration.subsetting_rule.value),
        polarisation_combination_method=translate_common.translate_polarisation_combination_method_to_model(
            configuration.polarisation_combination_method
        ),
    )


def translate_product_l2a(
    product: ProductL2AType,
    baseline: int,
) -> common_annotation_l2l3.BIOMASSL2aMainADSproduct:
    """Translate the L2A product model to its in-memory representation."""
    selected_reference_image = product.selected_reference_image
    return common_annotation_l2l3.BIOMASSL2aMainADSproduct(
        mission=translate_common.translate_mission_type(product.mission),
        tile_id_list=product.tile_id.id,
        basin_id_list=product.basin_id.id,
        product_type=translate_common.translate_product_type(product.product_type),
        start_time=translate_common.translate_datetime(product.start_time),
        stop_time=translate_common.translate_datetime(product.stop_time),
        radar_carrier_frequency=translate_common.translate_double_with_unit(product.radar_carrier_frequency),
        mission_phase_id=translate_common.translate_mission_phase_id(product.mission_phase_id),
        sensor_mode=translate_common.translate_sensor_mode(product.sensor_mode),
        global_coverage_id=product.global_coverage_id,
        swath=translate_common.translate_swath_type(product.swath),
        major_cycle_id=product.major_cycle_id,
        absolute_orbit_number=product.absolute_orbit_number.val,
        relative_orbit_number=product.relative_orbit_number,
        orbit_pass=translate_common.translate_orbit_pass(product.orbit_pass),
        datatake_id=product.data_take_id.val,
        frame=product.frame,
        platform_heading=translate_common.translate_double_with_unit(product.platform_heading),
        baseline=baseline,
        forest_coverage_percentage=product.forest_coverage_percentage,
        selected_reference_image=(
            common_annotation_l2l3.SelectedReferenceImageType(
                value=selected_reference_image.value,
                acquisition_id=selected_reference_image.acquisition_id,
            )
            if selected_reference_image is not None
            else None
        ),
    )


def translate_product_l2a_to_model(
    product: common_annotation_l2l3.BIOMASSL2aMainADSproduct,
) -> LATEST_VERSION.ProductL2AType:
    """Translate the in-memory L2A product representation to its model."""
    selected_reference_image = (
        LATEST_VERSION.SelectedReferenceImageType(
            value=product.selected_reference_image.value,
            acquisition_id=product.selected_reference_image.acquisition_id,
        )
        if product.selected_reference_image is not None
        else None
    )

    return LATEST_VERSION.ProductL2AType(
        mission=translate_common.translate_mission_type_to_model(product.mission),
        tile_id=translate_stringlist_to_model(product.tile_id_list),
        basin_id=translate_stringlist_to_model(product.basin_id_list),
        product_type=translate_common.translate_product_type_to_model(product.product_type),
        start_time=translate_common.translate_datetime_to_model(product.start_time),
        stop_time=translate_common.translate_datetime_to_model(product.stop_time),
        radar_carrier_frequency=translate_common.translate_double_with_unit_to_model(
            product.radar_carrier_frequency, common.UomType.HZ
        ),
        mission_phase_id=translate_common.translate_mission_phase_id_to_model(product.mission_phase_id),
        sensor_mode=translate_common.translate_sensor_mode_to_model(product.sensor_mode),
        global_coverage_id=product.global_coverage_id,
        swath=translate_common.translate_swath_type_to_model(product.swath),
        major_cycle_id=product.major_cycle_id,
        absolute_orbit_number=LATEST_VERSION.IntegerListType(val=product.absolute_orbit_number),
        relative_orbit_number=product.relative_orbit_number,
        orbit_pass=translate_common.translate_orbit_pass_to_model(product.orbit_pass),
        data_take_id=LATEST_VERSION.IntegerListType(val=product.datatake_id),
        frame=product.frame,
        platform_heading=translate_common.translate_double_with_unit_to_model(
            product.platform_heading, common.UomType.DEG
        ),
        forest_coverage_percentage=product.forest_coverage_percentage,
        selected_reference_image=selected_reference_image,
    )


def translate_raster_image_l2a(
    raster_image: RasterImageType,
) -> common_annotation_l2l3.BIOMASSL2aMainADSRasterImage:
    """Translate the L2A raster-image model to its in-memory representation."""
    return common_annotation_l2l3.BIOMASSL2aMainADSRasterImage(
        footprint=translate_common.translate_float_array_with_units(raster_image.footprint),
        first_latitude_value=translate_common.translate_float_with_unit(raster_image.first_latitude_value),
        first_longitude_value=translate_common.translate_float_with_unit(raster_image.first_longitude_value),
        latitude_spacing=translate_common.translate_float_with_unit(raster_image.latitude_spacing),
        longitude_spacing=translate_common.translate_float_with_unit(raster_image.longitude_spacing),
        number_of_samples=raster_image.number_of_samples,
        number_of_lines=raster_image.number_of_lines,
        projection=translate_common.translate_projection_type(raster_image.projection),
        datum=translate_common.translate_datum(raster_image.datum),
        pixel_representation=translate_pixel_representation_choice(raster_image.pixel_representation),
        pixel_type=translate_pixel_type_choice(raster_image.pixel_type),
        no_data_value=translate_no_data_value_choice(raster_image.no_data_value),
    )


def translate_raster_image_l2a_to_model(
    raster_image: common_annotation_l2l3.BIOMASSL2aMainADSRasterImage,
) -> LATEST_VERSION.RasterImageType:
    """Translate the in-memory L2A raster-image representation to its model."""
    return LATEST_VERSION.RasterImageType(
        footprint=translate_common.translate_float_array_with_units_to_model(
            raster_image.footprint, common.UomType.DEG
        ),
        first_latitude_value=translate_common.translate_float_with_unit_to_model(
            raster_image.first_latitude_value, common.UomType.DEG
        ),
        first_longitude_value=translate_common.translate_float_with_unit_to_model(
            raster_image.first_longitude_value, common.UomType.DEG
        ),
        latitude_spacing=translate_common.translate_float_with_unit_to_model(
            raster_image.latitude_spacing, common.UomType.DEG
        ),
        longitude_spacing=translate_common.translate_float_with_unit_to_model(
            raster_image.longitude_spacing, common.UomType.DEG
        ),
        number_of_samples=raster_image.number_of_samples,
        number_of_lines=raster_image.number_of_lines,
        projection=translate_common.translate_projection_type_to_model(raster_image.projection),
        datum=translate_common.translate_datum_to_model(raster_image.datum),
        pixel_representation=translate_pixel_representation_choice_to_model(raster_image.pixel_representation),
        pixel_type=translate_pixel_type_choice_to_model(raster_image.pixel_type),
        no_data_value=translate_no_data_value_choice_to_model(raster_image.no_data_value),
    )


def translate_input_information_l2a(
    input_information: InputInformationL2AType,
) -> common_annotation_l2l3.BIOMASSL2aMainADSInputInformation:
    """Translate the L2A input-information model to its in-memory representation."""
    return common_annotation_l2l3.BIOMASSL2aMainADSInputInformation(
        product_type=translate_common.translate_product_type(input_information.product_type),
        overall_products_quality_index=input_information.overall_products_quality_index,
        nominal_stack=input_information.nominal_stack,
        polarisation_list=translate_polarisation_list(input_information.polarisation_list),
        projection=translate_common.translate_projection_type(input_information.projection),
        footprint=translate_common.translate_float_array_with_units(input_information.footprint),
        vertical_wavenumbers=translate_common.translate_min_max_type_with_unit(input_information.vertical_wavenumbers),
        height_of_ambiguity=translate_common.translate_min_max_type_with_unit(input_information.height_of_ambiguity),
        acquisition_list=translate_acquisition_list(input_information.acquisition_list),
    )


def translate_input_information_l2a_to_model(
    input_information: common_annotation_l2l3.BIOMASSL2aMainADSInputInformation,
) -> LATEST_VERSION.InputInformationL2AType:
    """Translate the in-memory L2A input-information representation to its model."""
    return LATEST_VERSION.InputInformationL2AType(
        product_type=translate_common.translate_product_type_to_model(input_information.product_type),
        overall_products_quality_index=input_information.overall_products_quality_index,
        nominal_stack=input_information.nominal_stack,
        polarisation_list=translate_polarisation_list_to_model(input_information.polarisation_list),
        projection=translate_common.translate_projection_type_to_model(input_information.projection),
        footprint=translate_common.translate_float_array_with_units_to_model(
            input_information.footprint, common.UomType.DEG
        ),
        vertical_wavenumbers=translate_common.translate_min_max_type_with_unit_to_model(
            input_information.vertical_wavenumbers
        ),
        height_of_ambiguity=translate_common.translate_min_max_type_with_unit_to_model(
            input_information.height_of_ambiguity
        ),
        acquisition_list=translate_acquisition_list_to_model(input_information.acquisition_list),
    )


def translate_input_information_l2bl3_list(
    input_information: InputInformationL2BL3ListType,
) -> common_annotation_l2l3.InputInformationL2BL3ListType:
    """Translate an L2B/L3 input-information list to its in-memory representation."""
    l2a_inputs = []
    for input_item in input_information.l2a_inputs:
        l2a_inputs.append(
            common_annotation_l2l3.InputInformationL2BL3ListType.L2AInputs(
                l2a_product_folder_name=input_item.l2a_product_folder_name,
                l2a_product_date=input_item.l2a_product_date,
                l1_inputs=translate_input_information_l2a(input_item.l1_inputs),
                significance_level=input_item.significance_level,
            )
        )
    return common_annotation_l2l3.InputInformationL2BL3ListType(l2a_inputs=l2a_inputs, count=len(l2a_inputs))


def translate_input_information_l2bl3_list_to_model(
    input_information: common_annotation_l2l3.InputInformationL2BL3ListType,
) -> LATEST_VERSION.InputInformationL2BL3ListType:
    """Translate an in-memory L2B/L3 input-information list to its model."""
    l2a_inputs = []
    for input_item in input_information.l2a_inputs:
        l2a_inputs.append(
            LATEST_VERSION.InputInformationL2BL3ListType.L2AInputs(
                l2a_product_folder_name=input_item.l2a_product_folder_name,
                l2a_product_date=input_item.l2a_product_date,
                l1_inputs=translate_input_information_l2a_to_model(input_item.l1_inputs),
                significance_level=input_item.significance_level,
            )
        )
    return LATEST_VERSION.InputInformationL2BL3ListType(l2a_inputs=l2a_inputs, count=len(l2a_inputs))


def translate_cal_ab_coverage_to_model(
    coverage: common_annotation_l2l3.CalAbcoverageType,
) -> LATEST_VERSION.CalAbcoverageType:
    def translate_percent(
        value: common_annotation_l2l3.PercentPixelsType,
    ) -> LATEST_VERSION.PercentPixelsType:
        return LATEST_VERSION.PercentPixelsType(percentage=value.percentage, pixels=value.pixels)

    return LATEST_VERSION.CalAbcoverageType(
        original_coverage=translate_percent(coverage.original_coverage),
        filtered_coverage=LATEST_VERSION.CalAbfilteredCoverageType(
            after_global_filtering=translate_percent(coverage.filtered_coverage.after_global_filtering),
            after_agbvalue_filtering=translate_percent(coverage.filtered_coverage.after_agbvalue_filtering),
            after_agbstd_filtering=translate_percent(coverage.filtered_coverage.after_agbstd_filtering),
            after_agbrelative_std_filtering=translate_percent(
                coverage.filtered_coverage.after_agbrelative_std_filtering
            ),
            after_lcmclass_filtering=translate_percent(coverage.filtered_coverage.after_lcmclass_filtering),
        ),
        id=coverage.id,
    )


def translate_cal_ab_coverage(
    coverage: CalAbcoverageType,
) -> common_annotation_l2l3.CalAbcoverageType:
    def translate_percent(
        value: PercentPixelsType,
    ) -> common_annotation_l2l3.PercentPixelsType:
        return common_annotation_l2l3.PercentPixelsType(percentage=value.percentage, pixels=value.pixels)

    return common_annotation_l2l3.CalAbcoverageType(
        original_coverage=translate_percent(coverage.original_coverage),
        filtered_coverage=common_annotation_l2l3.CalAbfilteredCoverageType(
            after_global_filtering=translate_percent(coverage.filtered_coverage.after_global_filtering),
            after_agbvalue_filtering=translate_percent(coverage.filtered_coverage.after_agbvalue_filtering),
            after_agbstd_filtering=translate_percent(coverage.filtered_coverage.after_agbstd_filtering),
            after_agbrelative_std_filtering=translate_percent(
                coverage.filtered_coverage.after_agbrelative_std_filtering
            ),
            after_lcmclass_filtering=translate_percent(coverage.filtered_coverage.after_lcmclass_filtering),
        ),
        id=coverage.id,
    )


def translate_gn_coverage_to_model(
    coverage: common_annotation_l2l3.GncoverageListType,
) -> LATEST_VERSION.GncoverageListType:
    def translate_percent(
        value: common_annotation_l2l3.PercentPixelsType,
    ) -> LATEST_VERSION.PercentPixelsType:
        return LATEST_VERSION.PercentPixelsType(percentage=value.percentage, pixels=value.pixels)

    return LATEST_VERSION.GncoverageListType(
        gn=[
            LATEST_VERSION.GncoverageType(
                original_coverage=translate_percent(item.original_coverage),
                filtered_coverage=LATEST_VERSION.GnfilteredCoverageType(
                    after_global_filtering=translate_percent(item.filtered_coverage.after_global_filtering),
                    after_sigma_filtering=translate_percent(item.filtered_coverage.after_sigma_filtering),
                    after_angle_filtering=translate_percent(item.filtered_coverage.after_angle_filtering),
                ),
                acquisition_id=item.acquisition_id,
            )
            for item in coverage.gn
        ],
        id=coverage.id,
    )


def translate_gn_coverage(
    coverage: GncoverageListType,
) -> common_annotation_l2l3.GncoverageListType:
    def translate_percent(
        value: PercentPixelsType,
    ) -> common_annotation_l2l3.PercentPixelsType:
        return common_annotation_l2l3.PercentPixelsType(percentage=value.percentage, pixels=value.pixels)

    return common_annotation_l2l3.GncoverageListType(
        gn=[
            common_annotation_l2l3.GncoverageType(
                original_coverage=translate_percent(item.original_coverage),
                filtered_coverage=common_annotation_l2l3.GnfilteredCoverageType(
                    after_global_filtering=translate_percent(item.filtered_coverage.after_global_filtering),
                    after_sigma_filtering=translate_percent(item.filtered_coverage.after_sigma_filtering),
                    after_angle_filtering=translate_percent(item.filtered_coverage.after_angle_filtering),
                ),
                acquisition_id=item.acquisition_id,
            )
            for item in coverage.gn
        ],
        id=coverage.id,
    )


def translate_min_max_num_type_to_model(
    value: common_annotation_l2l3.MinMaxNumType,
) -> LATEST_VERSION.MinMaxNumType:
    return LATEST_VERSION.MinMaxNumType(
        min=value.min,
        max=value.max,
        num=value.num,
    )


def translate_min_max_num_type(
    value: MinMaxNumType,
) -> common_annotation_l2l3.MinMaxNumType:
    return common_annotation_l2l3.MinMaxNumType(
        min=value.min,
        max=value.max,
        num=value.num,
    )


def translate_productl2bl3(
    annotation: ProductL2BL3Type,
) -> common_annotation_l2l3.ProductL2bL3Type:
    """Translate a ProductL2BL3Type annotation from the model representation to the in-memory representation."""
    return common_annotation_l2l3.ProductL2bL3Type(
        mission=translate_common.translate_mission_type(annotation.mission),
        tile_id=annotation.tile_id,
        basin_id=translate_optional_stringlist(annotation.basin_id),
        product_type=translate_common.translate_product_type(annotation.product_type),
        start_time=translate_common.translate_datetime(annotation.start_time),
        stop_time=translate_common.translate_datetime(annotation.stop_time),
        radar_carrier_frequency=translate_common.translate_double_with_unit(annotation.radar_carrier_frequency),
        mission_phase_id=translate_common.translate_mission_phase_id(annotation.mission_phase_id),
        sensor_mode=translate_common.translate_sensor_mode(annotation.sensor_mode),
        global_coverage_id=annotation.global_coverage_id,
        contributing_tiles=translate_optional_stringlist(annotation.contributing_tiles),
        cal_ab_coverage_per_tile=[translate_cal_ab_coverage(item) for item in annotation.cal_abcoverage_per_tile.tile]
        if annotation.cal_abcoverage_per_tile is not None
        else [],
        gncoverage_per_tile=[translate_gn_coverage(item) for item in annotation.gncoverage_per_tile.tile]
        if annotation.gncoverage_per_tile is not None
        else [],
    )


def translate_productl2bl3_to_model(
    annotation: common_annotation_l2l3.ProductL2bL3Type,
) -> LATEST_VERSION.ProductL2BL3Type:
    """Translate a ProductL2bL3Type annotation from the in-memory representation to the model representation."""
    return LATEST_VERSION.ProductL2BL3Type(
        mission=translate_common.translate_mission_type_to_model(annotation.mission),
        tile_id=annotation.tile_id,
        basin_id=translate_stringlist_to_model(annotation.basin_id),
        product_type=translate_common.translate_product_type_to_model(annotation.product_type),
        start_time=translate_common.translate_datetime_to_model(annotation.start_time),
        stop_time=translate_common.translate_datetime_to_model(annotation.stop_time),
        radar_carrier_frequency=translate_common.translate_double_with_unit_to_model(
            annotation.radar_carrier_frequency, units=common.UomType.HZ
        ),
        mission_phase_id=translate_common.translate_mission_phase_id_to_model(annotation.mission_phase_id),
        sensor_mode=translate_common.translate_sensor_mode_to_model(annotation.sensor_mode),
        global_coverage_id=annotation.global_coverage_id,
        contributing_tiles=translate_optional_stringlist_to_model(annotation.contributing_tiles),
        cal_abcoverage_per_tile=translate_optional_cal_ab_coverage_per_tile_to_model(
            annotation.cal_ab_coverage_per_tile
        ),
        gncoverage_per_tile=translate_optional_gn_coverage_per_tile_to_model(annotation.gncoverage_per_tile),
    )


def translate_optional_gn_coverage_per_tile(
    annotation: GncoverageTilesListType | None,
) -> list[common_annotation_l2l3.GncoverageListType]:
    if annotation is None:
        return []
    return [translate_gn_coverage(item) for item in annotation.tile]


def translate_optional_gn_coverage_per_tile_to_model(
    annotation: list[common_annotation_l2l3.GncoverageListType],
) -> LATEST_VERSION.GncoverageTilesListType | None:
    if not annotation:
        return None
    return LATEST_VERSION.GncoverageTilesListType(tile=[translate_gn_coverage_to_model(item) for item in annotation])


def translate_optional_cal_ab_coverage_per_tile(
    annotation: CalAbcoverageTilesListType | None,
) -> list[common_annotation_l2l3.CalAbcoverageType]:
    if annotation is None:
        return []
    return [translate_cal_ab_coverage(item) for item in annotation.tile]


def translate_optional_cal_ab_coverage_per_tile_to_model(
    annotation: list[common_annotation_l2l3.CalAbcoverageType],
) -> LATEST_VERSION.CalAbcoverageTilesListType | None:
    if not annotation:
        return None
    return LATEST_VERSION.CalAbcoverageTilesListType(
        tile=[translate_cal_ab_coverage_to_model(item) for item in annotation]
    )
