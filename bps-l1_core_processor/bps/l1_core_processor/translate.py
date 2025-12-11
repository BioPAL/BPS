# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Translate functions
-------------------
"""

from math import atan2

from bps.common.io import aresys_configuration_models as are_conf
from bps.common.io import aresys_inputfile_models as are_input
from bps.common.io.aresys_configuration_models.models import UnitTypes
from bps.common.io.aresys_inputfile_models.models import (
    PfselectorAreaType,
    PfselectorAzimuthTimeIntervalType,
    PfselectorIndexIntervalType,
    PfselectorRangeTimeIntervalType,
    PfselectorRasterCoordinatesSwathType,
    PfselectorRasterCoordinatesType,
    PfselectorSwathNamesType,
    PfselectorSwathNameType,
    PfselectorTimeCoordinatesType,
)
from bps.l1_core_processor.input_file import (
    AntennaProducts,
    BPSL1CoreProcessorInputFile,
    CoreProcessorInputs,
)
from bps.l1_core_processor.pf_selector_input_file import (
    IndexInterval,
    PFSelectorAreaRasterCoordinates,
    PFSelectorAreaTimeCoordinates,
)
from bps.l1_core_processor.processing_options import (
    AntennaPatternCompensationLevel,
    BPSL1CoreProcessingOptions,
    EarthModel,
)
from bps.l1_core_processor.processing_parameters import (
    AzimuthConf,
    CalibrationConstantsConf,
    DopplerEstimatorStripmapConf,
    IonosphericCalibrationConf,
    IonosphericFeatureTracking,
    IonosphericSquintSensitivity,
    MultilookerConf,
    NoiseMapConf,
    PolarimetricProcessorConf,
    Quantity,
    RadiometricCalibrationConf,
    RangeFocuserConf,
    RFIMitigationConf,
    SarfocProcessingParameters,
    SlantToGroundConf,
    WindowConf,
)


def translate_area_time_coordinates(area_to_process: PFSelectorAreaTimeCoordinates) -> PfselectorAreaType:
    azimuth_time_interval = None
    if area_to_process.azimuth_time_interval is not None:
        azimuth_time_interval = PfselectorAzimuthTimeIntervalType(
            absolute_start_time=str(area_to_process.azimuth_time_interval.start_time),
            duration=area_to_process.azimuth_time_interval.duration,
        )

    range_time_interval: list[PfselectorRangeTimeIntervalType] = []
    if area_to_process.range_time_interval is not None:
        range_time_interval.append(
            PfselectorRangeTimeIntervalType(
                absolute_start_time=area_to_process.range_time_interval.start_time,
                duration=area_to_process.range_time_interval.duration,
            )
        )

    swaths = None
    if len(area_to_process.swaths) > 0:
        swaths = PfselectorSwathNamesType(
            [PfselectorSwathNameType(name=swath_name) for swath_name in area_to_process.swaths]
        )

    return PfselectorAreaType(
        time_coordinates=PfselectorTimeCoordinatesType(
            swaths=swaths,
            azimuth_time_interval=azimuth_time_interval,
            range_time_interval=range_time_interval,
        )
    )


def translate_area_raster_coordinates(area_to_process: PFSelectorAreaRasterCoordinates) -> PfselectorAreaType:
    def to_index(interval: IndexInterval) -> PfselectorIndexIntervalType:
        return PfselectorIndexIntervalType(start_index=interval.start_index, length=interval.length)

    raster_coordinates = PfselectorRasterCoordinatesType(swath=[])

    for swath in area_to_process.swaths:
        raster_coordinates_swath = PfselectorRasterCoordinatesSwathType(name=swath.name)
        if swath.lines_interval is not None:
            raster_coordinates_swath.line_interval = to_index(swath.lines_interval)
        if swath.samples_interval is not None:
            raster_coordinates_swath.sample_interval = [to_index(swath.samples_interval)]
        raster_coordinates.swath.append(raster_coordinates_swath)

    return PfselectorAreaType(raster_coordinates=raster_coordinates)


def translate_core_processor_input_to_model(
    input_file: CoreProcessorInputs,
) -> are_input.Sarfocinput:
    """Translate the core processor input file object to the XSD model structure

    Parameters
    ----------
    input_file : CoreProcessorInputs
        Core processor inputs object

    Returns
    -------
    aresys_inputfile_models.Sarfocinput
        Corresponding XSD model structure
    """
    sarfoc_step = are_input.Sarfocinput(
        input_level0_product=str(input_file.input_level0_product),
        processing_options_file=str(input_file.processing_options_file),
        processing_parameters_file=str(input_file.processing_parameters_file),
        output_path=str(input_file.output_directory),
    )

    if input_file.input_chirp_replica_product:
        sarfoc_step.input_chirp_replica_product = str(input_file.input_chirp_replica_product)

    if input_file.input_per_line_dechirping_reference_times_product:
        sarfoc_step.input_per_line_dechirping_reference_times_product = str(
            input_file.input_per_line_dechirping_reference_times_product
        )

    if input_file.input_per_line_correction_factors_product:
        sarfoc_step.input_per_line_correction_factors_product = str(
            input_file.input_per_line_correction_factors_product
        )

    if input_file.input_noise_product:
        sarfoc_step.input_noise_product = str(input_file.input_noise_product)

    if input_file.input_processing_dc_poly_file_name:
        sarfoc_step.input_processing_dcpoly_file_name = str(input_file.input_processing_dc_poly_file_name)

    if input_file.polarization_to_process:
        ...

    if input_file.area_to_process is not None:
        if isinstance(input_file.area_to_process, PFSelectorAreaTimeCoordinates):
            sarfoc_step.area_to_process = translate_area_time_coordinates(input_file.area_to_process)
        elif isinstance(input_file.area_to_process, PFSelectorAreaRasterCoordinates):
            sarfoc_step.area_to_process = translate_area_raster_coordinates(input_file.area_to_process)
        else:
            raise RuntimeError("Different area selection options not supported yet")

    return sarfoc_step


def translate_bps_antenna_patterns_to_model(
    patterns: AntennaProducts,
) -> are_input.InputBiomassAntennaPattern2Dtype:
    """Translate BPS antenna patterns product object to the corresponding XSD model."""
    return are_input.InputBiomassAntennaPattern2Dtype(
        input_antenna_pattern_d1_hproduct=str(patterns.d1h_pattern_product),
        input_antenna_pattern_d2_hproduct=str(patterns.d2h_pattern_product),
        input_antenna_pattern_d1_vproduct=str(patterns.d1v_pattern_product),
        input_antenna_pattern_d2_vproduct=str(patterns.d2v_pattern_product),
        input_txpower_tracking_product=str(patterns.tx_power_tracking_product),
    )


def translate_bps_l1_core_processor_input_file_to_model(
    input_file: BPSL1CoreProcessorInputFile,
) -> are_input.AresysXmlInput:
    """Translate BPS L1 core processor input file object to the corresponding XSD model.

    Parameters
    ----------
    input_file : BPSL1CoreProcessorInputFile
        Input file of the BPS L1 core processor

    Returns
    -------
    are_input.AresysXmlInput
        Corresponding XSD model structure
    """
    core_processor_inputs = translate_core_processor_input_to_model(input_file.core_processor_input)
    bps_l1_core_processor_step = are_input.Bpsl1CoreProcessorInputType(
        core_processor=core_processor_inputs,
        bpsconfiguration_file=str(input_file.bps_configuration_file),
        bpslog_file=str(input_file.bps_log_file),
    )

    if input_file.input_antenna_products is not None:
        bps_l1_core_processor_step.input_biomass_antenna_pattern2_d = translate_bps_antenna_patterns_to_model(
            input_file.input_antenna_products
        )
    if input_file.input_geomagnetic_field_model_product:
        bps_l1_core_processor_step.input_geomagnetic_field_model_product = str(
            input_file.input_geomagnetic_field_model_product
        )

    if input_file.input_tec_map_product:
        bps_l1_core_processor_step.input_tec_map_product = str(input_file.input_tec_map_product)

    if input_file.input_climatological_model_file:
        bps_l1_core_processor_step.input_climatological_model_file = str(input_file.input_climatological_model_file)

    if input_file.input_faraday_rotation_product:
        bps_l1_core_processor_step.input_faraday_rotation_product = str(input_file.input_faraday_rotation_product)

    if input_file.input_phase_screen_product:
        bps_l1_core_processor_step.input_phase_screen_product = str(input_file.input_phase_screen_product)

    return are_input.AresysXmlInput(
        [are_input.AresysXmlInputType.Step(bpsl1_core_processor=bps_l1_core_processor_step, number=1, total=1)]
    )


def translate_dem_info_to_model(
    dem_info: BPSL1CoreProcessingOptions.ExternalResources.DemInfo,
) -> are_conf.DigitalElevationModelType:
    """Translate dem information object to the corresponding XSD model.

    Parameters
    ----------
    dem_info : BPSL1CoreProcessingOptions.ExternalResources.DemInfo
        Dem information object

    Returns
    -------
    aresys_configuration_models.DigitalElevationModelType
        Corresponding XSD model structure
    """
    model = are_conf.DigitalElevationModelType(type_value=are_conf.DemTypes(dem_info.earth_model.value))

    if dem_info.geoid_file:
        model.geoid_file_name = str(dem_info.geoid_file)

    if dem_info.earth_model == EarthModel.COPERNICUS:
        model.index_file_name = str(dem_info.entry_point)
    else:
        model.repository = str(dem_info.entry_point)

    return model


def translate_antenna_pattern_compensation_level_to_model(
    level: AntennaPatternCompensationLevel,
) -> are_conf.BpsantennaPatternCompensationType:
    """Translate antenna pattern compensation level to the XSD model structure"""
    return are_conf.BpsantennaPatternCompensationType[level.value]


def translate_bps_l1_core_processor_processing_options_to_model(
    processing_options: BPSL1CoreProcessingOptions,
) -> are_conf.AresysXmlDoc:
    """Translate the BPS L1 Core Processor processing options file to the XSD model structure

    Parameters
    ----------
    processing_options : BPSL1CoreProcessingOptions
        BPS L1 Core Processor processing options object

    Returns
    -------
    aresys_configuration_models.AresysXmlDoc
        Corresponding XSD model structure
    """
    steps = are_conf.SarfocProcessingStepsType(
        processing_step=[
            are_conf.SarfocProcessingStepType(value=enabled, id=step.value)
            for step, enabled in processing_options.steps.items()
        ]
    )

    core_processor_settings = are_conf.SarfocProcessingSettingsType(
        digital_elevation_model=[
            are_conf.SarfocDigitalElevationModelType(value=earth_model.value, id=step.value)
            for step, earth_model in processing_options.settings.dem.items()
        ],
        prfchange_data_post_processing=processing_options.settings.prf_change_data_post_processing,
        rfimitigation_settings=are_conf.SarfocProcessingSettingsType.RfimitigationSettings(
            chirp_source=are_conf.RfimitigationSettingsChirpSource[processing_options.settings.rfi_use_chirp_product],
            mode=are_conf.RfimitigationSettingsMode[processing_options.settings.rfi_operation_mode],
        ),
    )

    settings = are_conf.Bpsl1CoreProcessingSettingsType(
        core_processing_settings=core_processor_settings,
        azimuth_compression=are_conf.Bpsl1CoreProcessingSettingsType.AzimuthCompression(
            antenna_pattern_compensation=translate_antenna_pattern_compensation_level_to_model(
                processing_options.settings.apc_level
            ),
            elevation_mispointing_deg=processing_options.settings.elevation_mispointing_deg,
        ),
        polarimetric_compensator=are_conf.Bpsl1CoreProcessingSettingsType.PolarimetricCompensator(
            enable_ionospheric_calibration=processing_options.settings.ionospheric_calibration_enabled
        ),
    )

    output_products = are_conf.SarfocOutputProductsType(
        product=[
            are_conf.SarfocProductType(value=output_name, id=product_id.value)
            for product_id, output_name in processing_options.output_products.items()
        ]
    )

    dem_repos = are_conf.SarfocExternalResourcesType(
        digital_elevation_model=[
            translate_dem_info_to_model(dem_info) for dem_info in processing_options.external_resources.dem_info_list
        ],
        prfresampling_filter_product=(
            str(processing_options.external_resources.prf_resampling_filter_product)
            if processing_options.external_resources.prf_resampling_filter_product
            else None
        ),
    )

    interface_settings = are_conf.Bpsl1CoreProcessorInterfaceSettingsType(
        products_format=processing_options.interface_settings.products_format.value,
        enable_quick_look_generation=processing_options.interface_settings.enable_quick_look_generation,
        remove_intermediate_products=processing_options.interface_settings.remove_intermediate_products,
    )

    processing_options_model = are_conf.Bpsl1CoreProcessorConfType(
        processing_steps=steps,
        processing_settings=settings,
        output_products=output_products,
        external_resources=dem_repos,
        interface_settings=interface_settings,
    )

    return are_conf.AresysXmlDoc(
        number_of_channels=1,
        version_number=2.6,
        description="BPSL1CoreProcessor processing options",
        channel=[
            are_conf.AresysXmlDocType.Channel(bpsl1_core_processor_conf=[processing_options_model], number=1, total=1)
        ],
    )


def translate_rfi_frequency_domain_conf_to_model(
    conf: RFIMitigationConf.FrequencyDomainConf,
) -> are_conf.RfifrequencyDomainRemovalConfType:
    """Translate RFI frequency domain configuration object to the XSD model structure"""
    return are_conf.RfifrequencyDomainRemovalConfType(
        beam=conf.swath,
        remove_interferences=(int(conf.remove_interferences) if conf.remove_interferences else None),
        block_size=conf.block_size,
        periodgram_size=conf.periodgram_size,
        persistent_rfithreshold=conf.persistent_rfi_threshold,
        isolated_rfithreshold=conf.isolated_rfi_threshold,
        power_loss_threshold=conf.power_loss_threshold,
        threshold_std=conf.threshold_std,
        percentile_low=conf.percentile_low,
        percentile_high=conf.percentile_high,
        filtering_mode=are_conf.RfifrequencyDomainRemovalConfTypeFilteringMode[conf.filtering_mode],
    )


def translate_rfi_time_domain_conf_to_model(
    conf: RFIMitigationConf.TimeDomainConf,
) -> are_conf.RfitimeDomainRemovalConfType:
    """Translate RFI time domain configuration object to the XSD model structure"""
    return are_conf.RfitimeDomainRemovalConfType(
        beam=conf.swath,
        correction_mode=are_conf.RfitimeDomainRemovalConfTypeCorrectionMode(conf.correction_mode.name),
        percentile_threshold=conf.percentile_threshold,
        median_filter_block_lines=conf.median_filter_block_lines,
        lines_in_estimate_block=conf.lines_in_estimate_block,
        box_filter_azimuth_dimension=conf.box_filter_azimuth_dimension,
        box_filter_range_dimension=conf.box_filter_range_dimension,
        morph_open_line_length=conf.morph_open_line_length,
        morph_open_close_iterations=conf.morph_open_close_iterations,
        morph_close_line_length=conf.morph_close_line_length,
        morph_close_before_open=conf.morph_close_before_open,
    )


def translate_rfi_mitigation_conf_to_model(
    conf: RFIMitigationConf,
) -> are_conf.RfimitigationConfType:
    """Translate RFI mitigation configuration object to the XSD model structure"""
    model = are_conf.RfimitigationConfType(
        beam=conf.swath,
        rfimitigation_method=are_conf.RfimitigationMethodsType(conf.rfi_mitigation_method.name),
        rfimask_composition_method=are_conf.RfimaskCompositionMethodsType(conf.rfi_mask_composition_method.name),
    )

    if conf.frequency_domain_conf:
        model.rfimitigation_frequency_domain_conf = [
            translate_rfi_frequency_domain_conf_to_model(conf.frequency_domain_conf)
        ]
    if conf.time_domain_conf:
        model.rfimitigation_time_domain_conf = translate_rfi_time_domain_conf_to_model(conf.time_domain_conf)

    return model


def translate_unit_to_model(unit: Quantity.Unit) -> UnitTypes:
    """Translate quantity unit to model"""
    unit_to_model = {
        Quantity.Unit.NORMALIZED: UnitTypes.NORMALIZED,
        Quantity.Unit.HZ: UnitTypes.HZ,
        Quantity.Unit.S: UnitTypes.S,
    }

    return unit_to_model[unit]


def translate_window_conf_to_model(
    conf: WindowConf,
) -> are_conf.WindowConfType:
    """Translate window configuration object to the XSD model structure"""

    return are_conf.WindowConfType(
        window_type=are_conf.Windows(conf.window_type.name),
        window_parameter=conf.window_parameter,
        window_look_bandwidth=are_conf.WindowConfType.WindowLookBandwidth(
            value=conf.window_look_bandwidth.value,
            unit=translate_unit_to_model(conf.window_look_bandwidth.unit),
        ),
        window_transition_bandwidth=are_conf.WindowConfType.WindowTransitionBandwidth(
            value=conf.window_transition_bandwidth.value,
            unit=translate_unit_to_model(conf.window_transition_bandwidth.unit),
        ),
    )


def translate_output_prf_value_to_model(
    value: float,
) -> are_conf.RangeConfType.PostProcessing:
    """Translate the range focuser post pocessing output PRF value to the XSD model structure"""

    output_prf = are_conf.RangeConfType.PostProcessing.AzimuthResampling.OutputPrf(value=value)
    azimuth_resampling = are_conf.RangeConfType.PostProcessing.AzimuthResampling(output_prf=output_prf)
    post_processing = are_conf.RangeConfType.PostProcessing(azimuth_resampling=azimuth_resampling)

    return post_processing


def translate_range_focuser_conf_to_model(
    conf: RangeFocuserConf,
) -> are_conf.RangeConfType:
    """Translate range focuser configuration object to the XSD model structure"""
    assert conf.focusing_method is not None
    return are_conf.RangeConfType(
        beam=conf.swath,
        flag_ortog=int(conf.flag_ortog),
        swstbias=conf.swst_bias,
        apply_range_spectral_weighting_window=int(conf.apply_range_spectral_weighting_window),
        range_spectral_weighting_window=translate_window_conf_to_model(conf.range_spectral_weighting_window),
        range_decimation_factor=conf.range_decimation_factor,
        apply_rx_gain_correction=conf.apply_rx_gain_correction,
        focusing_method=are_conf.RangeFocusingMethodType(conf.focusing_method.name),
        output_border_policies=(
            are_conf.RangeConfType.OutputBorderPolicies(
                range=are_conf.OutputBorderPolicyType(conf.output_range_border_policy.name)
            )
            if conf.output_range_border_policy is not None
            else None
        ),
        post_processing=(
            translate_output_prf_value_to_model(conf.output_prf_value) if conf.output_prf_value is not None else None
        ),
    )


def translate_doppler_estimator_conf_to_model(
    conf: DopplerEstimatorStripmapConf,
) -> are_conf.StripmapDcConfType:
    """Translate doppler estimator configuration object to the XSD model structure"""
    model = are_conf.StripmapDcConfType(
        beam=conf.swath,
        blocks=conf.blocks,
        blockl=conf.blockl,
        undersampling_snrdcazimuth_ratio=conf.undersampling_snrd_cazimuth_ratio,
        undersampling_snrdcrange_ratio=conf.undersampling_snrd_crange_ratio,
        az_max_frequency_search_bin_number=conf.az_max_frequency_search_bin_number,
        rg_max_frequency_search_bin_number=conf.rg_max_frequency_search_bin_number,
        az_max_frequency_search_norm_band=conf.az_max_frequency_search_norm_band,
        rg_max_frequency_search_norm_band=conf.rg_max_frequency_search_norm_band,
        nummlbf=conf.nummlbf,
        nbestblocks=conf.nbestblocks,
        rg_band=conf.rg_band,
        an_len=conf.an_len,
        lookbf=conf.lookbf,
        lookbt=conf.lookbt,
        lookrp=conf.lookrp,
        lookrs=conf.lookrs,
        decfac=conf.decfac,
        flength=conf.flength,
        dftstep=conf.dftstep,
        peakwid=conf.peakwid,
        minamb=conf.minamb,
        maxamb=conf.maxamb,
        sthr=conf.sthr,
        varth=conf.varth,
        pol_weights=are_conf.StripmapDcConfType.PolWeights(conf.pol_weights),
        dc_estimation_method=are_conf.DcEstimationMethodsTypes(conf.dc_estimation_method.name),
        attitude_fitting=are_conf.AttitudeFittingTypes(conf.attitude_fitting.name),
        poly_changing_freq=conf.poly_changing_freq,
        perform_joint_estimation=(int(conf.joint_estimation) if conf.joint_estimation is not None else None),
    )

    if conf.poly_estimation_constraint:
        model.poly_estimation_constraint = are_conf.PolyEstimationConstraintTypes(conf.poly_estimation_constraint.name)

    if conf.dc_core_algorithm:
        ...

    return model


def translate_azimuth_focuser_conf_to_model(
    conf: AzimuthConf,
) -> are_conf.AzimuthConfType:
    """Translate azimuth focuser configuration object to the XSD model structure"""

    model = are_conf.AzimuthConfType(
        beam=conf.swath,
        lines_in_block=conf.lines_in_block,
        samples_in_block=conf.samples_in_block,
        azimuth_overlap=conf.azimuth_overlap,
        range_overlap=conf.range_overlap,
        perform_interpolation=conf.perform_interpolation,
        stolt_padding=conf.stolt_padding,
        range_modulation=int(conf.range_modulation),
        apply_azimuth_spectral_weighting_window=int(conf.apply_azimuth_spectral_weighting_window),
        azimuth_spectral_weighting_window=translate_window_conf_to_model(conf.azimuth_spectral_weighting_window),
        apply_rg_shift=int(conf.apply_rg_shift),
        apply_az_shift=int(conf.apply_az_shift),
        whitening_flag=int(conf.whitening_flag),
        antenna_length=conf.antenna_length,
        pad_result=conf.pad_result,
        lines_to_skip_dc_fr=conf.lines_to_skip_dc_fr,
        samples_to_skip_dc_fr=conf.samples_to_skip_dc_fr,
        azimuth_time_bias=conf.azimuth_time_bias,
    )
    if conf.focusing_method:
        model.focusing_method = are_conf.FocusingMethodTypes(conf.focusing_method.name)
    if conf.az_proc_bandwidth:
        model.az_proc_bandwidth = are_conf.AzimuthConfType.AzProcBandwidth(
            value=conf.az_proc_bandwidth.value,
            unit=translate_unit_to_model(conf.az_proc_bandwidth.unit),
        )
    if conf.bistatic_delay_correction_mode:
        model.bistatic_delay_correction_mode = are_conf.BistaticDelayCorrectionTypes(
            conf.bistatic_delay_correction_mode.name
        )
    if conf.antenna_shift_compensation_mode:
        model.antenna_shift_compensation_mode = are_conf.AntennaShiftCompensationModeType(
            conf.antenna_shift_compensation_mode.name
        )
    if conf.apply_pol_channels_coregistration is not None:
        model.apply_pol_channels_coregistration = int(conf.apply_pol_channels_coregistration)
    if conf.nominal_block_memory_size_cpu or conf.nominal_block_memory_size_gpu:
        model.nominal_block_memory_size = are_conf.AzimuthConfType.NominalBlockMemorySize(
            cpu=(
                are_conf.MemorySizeType(conf.nominal_block_memory_size_cpu)
                if conf.nominal_block_memory_size_cpu
                else None
            )
        )
        if conf.nominal_block_memory_size_gpu:
            model.nominal_block_memory_size.gpu = [are_conf.MemorySizeType(conf.nominal_block_memory_size_gpu)]

    return model


def translate_radiometric_calibration_conf_to_model(
    conf: RadiometricCalibrationConf,
) -> are_conf.RangeCompensatorConfType:
    """Translate radiometric calibration configuration object to the XSD model structure"""
    model = are_conf.RangeCompensatorConfType(
        beam=conf.swath,
        rslreference_distance=conf.rsl_reference_distance,
        perform_rslcompensation=int(conf.perform_rsl_compensation),
        perform_pattern_compensation=int(conf.perform_pattern_compensation),
        perform_line_correction=conf.perform_line_correction,
        fast_mode=int(conf.fast_mode) if conf.fast_mode else None,
        external_calibration_factor=are_conf.RangeCompensatorConfType.ExternalCalibrationFactor(
            complex_alg=are_conf.ComplexAlg(
                real_value=conf.external_calibration_factor.real,
                imaginary_value=conf.external_calibration_factor.imag,
            ),
            apply=(int(conf.apply_external_calibration_factor) if conf.apply_external_calibration_factor else None),
        ),
    )

    perform_incidence_compensation = (
        conf.output_quantity is not None and conf.output_quantity != RadiometricCalibrationConf.OutputQuantity.BETA
    )

    output_quantity = None

    if perform_incidence_compensation:
        assert conf.output_quantity
        output_quantity = are_conf.OutputQuantityType(conf.output_quantity.name)

    model.perform_incidence_compensation = are_conf.RangeCompensatorConfType.PerformIncidenceCompensation(
        value=int(perform_incidence_compensation), output_quantity=output_quantity
    )

    if conf.processing_gain:
        model.processing_gain = are_conf.Fcomplex(
            complex_alg=are_conf.ComplexAlg(
                real_value=conf.processing_gain.real,
                imaginary_value=conf.processing_gain.imag,
            )
        )

    return model


def translate_polarimetric_processor_conf_to_model(
    conf: PolarimetricProcessorConf,
) -> are_conf.PolarimetricProcessorConfType:
    """Translate polarimetric processor configuration object to the XSD model structure"""
    return are_conf.PolarimetricProcessorConfType(
        beam=conf.swath,
        enable_channel_imbalance_compensation=int(conf.enable_channel_imbalance_compensation),
        enable_cross_talk_compensation=int(conf.enable_cross_talk_compensation),
    )


def translate_float_to_real_value_degrees(number: float) -> are_conf.RealValueDegrees:
    """Translate float value to specific conf type to the XSD model structure"""
    return are_conf.RealValueDegrees(value=number)


def translate_float_to_real_value_meters(number: float) -> are_conf.RealValueMeters:
    """Translate float value to specific conf type to the XSD model structure"""
    return are_conf.RealValueMeters(value=number)


def translate_ionospheric_squint_sensitivity_to_model(
    conf: IonosphericSquintSensitivity,
) -> are_conf.IonosphericCalibrationConfType.SquintSensitivity:
    """Translate squint sensitivity object to the XSD model structure"""
    return are_conf.IonosphericCalibrationConfType.SquintSensitivity(
        number_of_looks=conf.number_of_looks,
        height_step=conf.height_step,
        faraday_rotation_bias=translate_float_to_real_value_degrees(conf.faraday_rotation_bias),
    )


def translate_ionospheric_feature_tracking_to_model(
    conf: IonosphericFeatureTracking,
) -> are_conf.IonosphericCalibrationConfType.FeatureTracking:
    """Translate feature tracking object to the XSD model structure"""
    return are_conf.IonosphericCalibrationConfType.FeatureTracking(
        max_offset=conf.max_offset,
        profile_step=conf.profile_step,
        normalized_min_value_threshold=conf.normalized_min_value_threshold,
    )


def translate_ionospheric_calibration_conf_to_model(
    conf: IonosphericCalibrationConf,
) -> are_conf.IonosphericCalibrationConfType:
    """Translate polarimetric processor configuration object to the XSD model structure"""
    return are_conf.IonosphericCalibrationConfType(
        beam=conf.swath,
        perform_defocusing_on_ionospheric_height=int(conf.perform_defocusing_on_ionospheric_height),
        perform_faraday_rotation_correction=int(conf.perform_defocusing_on_ionospheric_height),
        perform_phase_screen_correction=int(conf.perform_phase_screen_correction),
        perform_group_delay_correction=int(conf.perform_group_delay_correction),
        ionospheric_height_estimation_method=are_conf.IonosphericCalibrationConfTypeIonosphericHeightEstimationMethod(
            conf.ionospheric_height_estimation_method.value
        ),
        squint_sensitivity=(
            translate_ionospheric_squint_sensitivity_to_model(conf.squint_sensitivity)
            if conf.squint_sensitivity is not None
            else None
        ),
        feature_tracking=(
            translate_ionospheric_feature_tracking_to_model(conf.feature_tracking)
            if conf.feature_tracking is not None
            else None
        ),
        zthreshold=conf.z_threshold,
        gaussian_filter_max_size_azimuth=conf.gaussian_filter_max_size_azimuth,
        gaussian_filter_max_size_range=conf.gaussian_filter_max_size_range,
        gaussian_filter_default_size_azimuth=conf.gaussian_filter_default_size_azimuth,
        gaussian_filter_default_size_range=conf.gaussian_filter_default_size_range,
        default_ionospheric_height=translate_float_to_real_value_meters(conf.default_ionospheric_height),
        max_ionospheric_height=translate_float_to_real_value_meters(conf.max_ionospheric_height),
        min_ionospheric_height=translate_float_to_real_value_meters(conf.min_ionospheric_height),
        azimuth_block_size=conf.azimuth_block_size,
        azimuth_block_overlap=conf.azimuth_block_overlap,
    )


def translate_complex_to_model(complex_number: complex) -> are_conf.ComplexNumberType:
    """Translate complex object to the XSD model structure"""
    return are_conf.ComplexNumberType(
        amplitude=abs(complex_number),
        phase=atan2(complex_number.imag, complex_number.real),
    )


def translate_calibration_constants_conf_to_model(
    conf: CalibrationConstantsConf,
) -> are_conf.CalibrationConstantsConfType:
    """Translate calibration constants configuration object to the XSD model structure"""
    channel_imbalance_values = are_conf.CalibrationConstantsConfType.Radiometric.ChannelImbalanceValues(
        rx=translate_complex_to_model(complex_number=conf.channel_imbalance_rx),
        tx=translate_complex_to_model(complex_number=conf.channel_imbalance_tx),
    )

    # Cross talk mapping TBC
    cross_talk_correction = are_conf.CalibrationConstantsConfType.Radiometric.CrossTalkCorrection(
        xtalk1=translate_complex_to_model(conf.cross_talk_hv_rx),
        xtalk2=translate_complex_to_model(conf.cross_talk_vh_rx),
        xtalk3=translate_complex_to_model(conf.cross_talk_vh_tx),
        xtalk4=translate_complex_to_model(conf.cross_talk_hv_tx),
        alpha=None,
    )

    radiometric = are_conf.CalibrationConstantsConfType.Radiometric(
        channel_imbalance_values=channel_imbalance_values,
        cross_talk_correction=cross_talk_correction,
    )
    geometric = are_conf.CalibrationConstantsConfType.Geometric(
        internal_delay_hh=conf.internal_delay_hh,
        internal_delay_hv=conf.internal_delay_hv,
        internal_delay_vh=conf.internal_delay_vh,
        internal_delay_vv=conf.internal_delay_vv,
    )

    return are_conf.CalibrationConstantsConfType(radiometric=radiometric, geometric=geometric)


def translate_multilooker_conf_to_model(
    conf: MultilookerConf,
) -> are_conf.MultilookConfType:
    """Translate multilooker configuration object to the XSD model structure"""
    model = are_conf.MultilookConfType(
        beam=conf.swath,
        multilook_conf_name=conf.multilook_conf_name,
        normalization_factor=conf.normalization_factor,
    )

    if conf.azimuth_time_weighting_window_info:
        model.apply_azimuth_time_weighting_window = conf.azimuth_time_weighting_window_info.apply
        model.azimuth_time_weighting_window = translate_window_conf_to_model(
            conf.azimuth_time_weighting_window_info.window
        )

    def translate_central_frequencies_to_model(
        conf: list[Quantity],
    ) -> list[are_conf.MultilookDirectionConfType.CentralFrequency]:
        return [
            are_conf.MultilookDirectionConfType.CentralFrequency(
                value=central_frequency.value,
                unit=translate_unit_to_model(central_frequency.unit),
            )
            for central_frequency in conf
        ]

    def translate_multilooker_single_dir(
        conf: MultilookerConf.MultilookerDoubleDirectionConf.MultilookerSingleDirectionConf,
    ) -> are_conf.MultilookDirectionConfType:
        return are_conf.MultilookDirectionConfType(
            pfactor=conf.p_factor,
            qfactor=conf.q_factor,
            weighting_window=translate_window_conf_to_model(conf.weighting_window),
            central_frequency=translate_central_frequencies_to_model(conf.central_frequency),
        )

    if isinstance(conf.multilook_conf, MultilookerConf.MultilookerDoubleDirectionConf):
        if conf.multilook_conf.fast_multilook:
            model.fast_multilook = translate_multilooker_single_dir(conf.multilook_conf.fast_multilook)
        if conf.multilook_conf.slow_multilook:
            model.slow_multilook = translate_multilooker_single_dir(conf.multilook_conf.slow_multilook)
    else:
        assert isinstance(conf.multilook_conf, MultilookerConf.PresumConf)
        model.presum = are_conf.MultilookConfType.Presum(
            fast_factor=conf.multilook_conf.fast_factor,
            slow_factor=conf.multilook_conf.slow_factor,
        )

    if conf.invalid_value:
        model.invalid_value = are_conf.ComplexAlg(
            real_value=conf.invalid_value.real, imaginary_value=conf.invalid_value.imag
        )

    return model


def translate_noise_map_generator_conf_to_model(
    conf: NoiseMapConf,
) -> are_conf.NoiseMapGeneratorConfType:
    """Translate noise map generator configuration object to the XSD model structure"""
    return are_conf.NoiseMapGeneratorConfType(
        noise_normalization_constant=conf.noise_normalization_constant, beam=conf.swath
    )


def translate_slant_to_ground_conf_to_model(
    conf: SlantToGroundConf,
) -> are_conf.Slant2GroundConfType:
    """Translate slnt to ground configuration object to the XSD model structure"""
    model = are_conf.Slant2GroundConfType(ground_step=conf.ground_step, beam=conf.swath)

    if conf.invalid_value:
        model.invalid_value = are_conf.ComplexAlg(
            real_value=conf.invalid_value.real, imaginary_value=conf.invalid_value.imag
        )

    return model


def translate_sarfoc_processing_parameters_to_model(
    processing_parameters: SarfocProcessingParameters,
) -> are_conf.AresysXmlDoc:
    """Translate the Sarfoc processing options file to the XSD model structure

    Parameters
    ----------
    processing_parameters : SarfocProcessingParameters
        Sarfoc processing options object

    Returns
    -------
    aresys_configuration_models.AresysXmlDoc
        Corresponding XSD model structure
    """
    rfi_conf = [translate_rfi_mitigation_conf_to_model(c) for c in processing_parameters.rfi_mitigation_conf]
    range_conf = [translate_range_focuser_conf_to_model(c) for c in processing_parameters.range_focuser_conf]
    dcest_conf = [translate_doppler_estimator_conf_to_model(c) for c in processing_parameters.doppler_estimator_conf]
    azimuth_conf = [translate_azimuth_focuser_conf_to_model(c) for c in processing_parameters.azimuth_conf]
    radio_conf = [
        translate_radiometric_calibration_conf_to_model(c) for c in processing_parameters.radiometric_calibration_conf
    ]
    multilooker_conf = [translate_multilooker_conf_to_model(c) for c in processing_parameters.multilooker_conf]
    polarimetric_processor_conf = [
        translate_polarimetric_processor_conf_to_model(c) for c in processing_parameters.polarimetric_processor_conf
    ]
    ionospheric_calibration_conf = [
        translate_ionospheric_calibration_conf_to_model(c) for c in processing_parameters.ionospheric_calibration_conf
    ]
    calibration_constants_conf = translate_calibration_constants_conf_to_model(
        processing_parameters.calibration_constants_conf
    )
    noise_map_generator_conf = [
        translate_noise_map_generator_conf_to_model(c) for c in processing_parameters.noise_map_conf
    ]
    slant_to_ground_conf = [
        translate_slant_to_ground_conf_to_model(c) for c in processing_parameters.slant_to_ground_conf
    ]
    return are_conf.AresysXmlDoc(
        number_of_channels=1,
        version_number=2.6,
        description="Sarfoc processing parameters",
        channel=[
            are_conf.AresysXmlDocType.Channel(
                rfimitigation_conf=rfi_conf,
                range_conf=range_conf,
                dcest_conf_stripmap=dcest_conf,
                azimuth_conf=azimuth_conf,
                range_compensator_conf=radio_conf,
                polarimetric_processor_conf=polarimetric_processor_conf,
                ionospheric_calibration_conf=ionospheric_calibration_conf,
                calibration_constants_conf=[calibration_constants_conf],
                multi_processor_conf=multilooker_conf,
                noise_map_generator_conf=noise_map_generator_conf,
                slant2_ground_conf=slant_to_ground_conf,
                number=1,
                total=1,
            )
        ],
    )
