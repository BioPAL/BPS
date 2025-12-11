# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Translation Utilities
---------------------
"""

from pathlib import Path

from bps.common.io import aresys_configuration_models as are_conf
from bps.common.io import aresys_inputfile_models as are_input
from bps.stack_coreg_processor.configuration import (
    CoregistrationOutputProductsConfType,
    CoregStackProcessorInternalConfiguration,
    FullAccuracyPostProcessingConfType,
    FullAccuracyPreProcessingConfType,
    GeneralCoregStackProcessorInternalConfiguration,
    NonStationaryAreasConfType,
    ReinterpolationConfType,
)
from bps.stack_coreg_processor.input_file import (
    BPSCoregProcessorInputFile,
    CoregProcessorInputFile,
)
from bps.stack_coreg_processor.utils import StackCoregProcessorRuntimeError


def translate_coreg_input_file_to_model(
    input_file: CoregProcessorInputFile,
) -> are_input.GenericCoregistratorInputType:
    """
    Translate the Stack Coregistration Processor input file object to
    the XSD model structure.

    Parameters
    ----------
    input_file: StackCoregProcessorInputFile
        StackCoregProcessor input file object

    Raises
    ------
    StackCoregProcessorRuntimeError

    Returns
    -------
    aresys_inputfile_models.GenericCoregistratorInputType
        Corresponding XSD model structure.

    """
    if input_file.ecef_grid_product is None:
        raise StackCoregProcessorRuntimeError("Missing required mandatory ECEF grid product")

    return are_input.GenericCoregistratorInputType(
        master_level1_product=str(input_file.primary_product),
        slave_level1_product=str(input_file.secondary_product),
        dem_xyzproduct=str(input_file.ecef_grid_product),
        output_path=str(input_file.output_path),
        config_file_name=str(input_file.coreg_conf_file),
        dem_xyzproduct_channels=are_input.DemXyzproductChannelsType.XYZ_ONLY,
        external_shifts_az=_optionally_to_str(input_file.az_shifts_product),
        external_shifts_rg=_optionally_to_str(input_file.rg_shifts_product),
    )


def translate_full_accuracy_preproc_conf_to_model(
    conf: FullAccuracyPreProcessingConfType,
) -> are_conf.FullAccuracyPreProcessingConfType:
    """
    Translate the FullAccuracyPreProcessingConfType object to the XSD model.

    Parameters
    ----------
    conf: FullAccuracyPreProcessingConfType
        The FineFit configuration object.

    Returns
    -------
    aresys_configuration_models.FullAccuracyPreProcessingConfType
        The XSD object.

    """
    return are_conf.FullAccuracyPreProcessingConfType(
        enable_common_band_range_filter=conf.enable_common_band_range_filter,
        range_max_shift=conf.range_max_shift,
        azimuth_max_shift=conf.azimuth_max_shift,
        range_block_size=conf.range_block_size,
        azimuth_block_size=conf.azimuth_block_size,
        coarse_input=conf.coarse_input,
        range_min_overlap=conf.range_min_overlap,
        azimuth_min_overlap=conf.azimuth_min_overlap,
        memory=conf.memory,
        verbose=conf.verbose,
        report_level=conf.report_level,
        coreg_reference_polarization=conf.coreg_reference_polarization.value,
    )


def translate_non_stationary_coreg_conf(
    conf: NonStationaryAreasConfType,
) -> are_conf.NonStationaryCoregConfType | None:
    """
    Translate the NonStationaryAreasConfType into an XSD model.

    Parameters
    ----------
    conf: NonStationaryAreasConfType
        Optionally, the configuration object.

    Returns
    -------
    aresys_configuration_models.NonStationaryCoregConfType | None
        The XSD serializable object, if the input is not None. None otherwise.

    """
    if conf is None:
        return conf

    return are_conf.NonStationaryCoregConfType(
        lpfilter_type=are_conf.FilterMask(conf.low_pass_filter_type.upper()),
        order=conf.low_pass_filter_order,
        parameter=conf.low_pass_filter_std_dev,
        interp_type=are_conf.InterpType.LINEAR,
    )


def translate_full_accuracy_postproc_conf_type_to_model(
    conf: FullAccuracyPostProcessingConfType,
) -> are_conf.FullAccuracyPostProcessingConfType:
    """
    Translate a FullAccuracyPostProcessingConfType object to the XSD model.

    Parameters
    ----------
    conf: FullAccuracyPostProcessingConfType
        The full-accuracy post-processing conf object.

    Returns
    -------
    aresys_configuration_models.FullAccuracyPostProcessingConfType
        The XSD object.

    """
    return are_conf.FullAccuracyPostProcessingConfType(
        residual_shift_fitting_model=are_conf.FullAccuracyPostProcessingConfTypeResidualShiftFittingModel(
            value="MODEL_BASED"
        ),
        weight_threshold_refine_rg=conf.weight_threshold_refine_rg,
        weight_threshold_refine_az=conf.weight_threshold_refine_az,
        quality_threshold_for_automatic_mode=conf.quality_threshold_for_automatic_mode,
        min_valid_blocks=conf.min_valid_blocks,
        non_stationary_coreg_conf=translate_non_stationary_coreg_conf(conf.non_stationary_coreg_conf),
    )


def translate_reinterpolation_conf_type_to_model(
    conf: ReinterpolationConfType,
) -> are_conf.ReinterpolationConfType:
    """
    Translate a ReinterpolationConfType object to the XSD model.

    Parameters
    ----------
    conf: ReinterpolationConfType
        The interpolation configuration object.

    Returns
    -------
    aresys_configuration_models.ReinterpolationConfType
        The XSD object.

    """
    return are_conf.ReinterpolationConfType(
        filter_length=conf.filter_length,
        bank_size=conf.bank_size,
        bandwidth=conf.bandwidth,
        range_overlap=conf.range_overlap,
        demodulation_type=conf.demodulation_type,
        unsigned_flag=conf.unsigned_flag,
        memory=conf.memory,
        verbose=conf.verbose,
        report_level=conf.report_level,
    )


def translate_coregistration_output_products_conf_type_to_model(
    conf: CoregistrationOutputProductsConfType,
) -> are_conf.CoregistrationOutputProductsConfType:
    """
    Translate a CoregistrationOutputProdyctsConfType object to the XSD model.

    Parameters
    ----------
    conf: CoregistrationOutputProductsConfType
        The coregistration output product configuration object.

    Returns
    -------
    aresys_configuration_models.CoregistrationOutputProductsConfType
        The XSD object.

    """
    return are_conf.CoregistrationOutputProductsConfType(
        remove_ancillary_coregistration_data=conf.remove_ancillary_coregistration_data,
        provide_coregistration_shifts=conf.provide_coregistration_shifts,
        provide_geometry_shifts=conf.provide_geometry_shifts,
        provide_coregistration_accuracy_stats=conf.provide_coregistration_accuracy_stats,
        provide_products_for_each_polarization=conf.provide_products_for_each_polarization,
        xcorr_azimuth_min_overlap=conf.xcorr_azimuth_min_overlap,
        shifts_only_estimation=conf.shifts_only_estimation,
        provide_wavenumbers=conf.provide_wavenumbers,
        provide_absolute_primary_distance=conf.provide_absolute_primary_distance,
    )


def translate_bps_coreg_input_file_to_model(
    input_file: BPSCoregProcessorInputFile,
) -> are_input.AresysXmlInput:
    """
    Translate the Stack Coregistration Processor input file object to
    the XSD model structure.

    Parameters
    ----------
    input_file: StackCoregProcessorInputFile
        StackCoregProcessor input file object.

    Return
    ------
    aresys_inputfile_models.AresysXmlInput
        Corresponding XSD model structure.

    """
    bpsstack_processor_step = are_input.BpsstackProcessorInputType(
        coregistration=translate_coreg_input_file_to_model(input_file.coregistration_input),
        bpsconfiguration_file=str(input_file.bps_configuration_file),
        bpslog_file=str(input_file.bps_log_file),
    )

    return are_input.AresysXmlInput(
        [are_input.AresysXmlInputType.Step(bpsstack_processor=bpsstack_processor_step, number=1, total=1)]
    )


def translate_coreg_configuration_to_model(
    conf: CoregStackProcessorInternalConfiguration,
) -> are_conf.AresysXmlDoc:
    """
    Translate the stack coreg processor configuration to XSD model.

    Parameters
    ----------
    conf: CoregStackProcessorInternalConfiguration
        The stack coreg processor configuration.

    Return
    ------
    aresys_configuration_models.AresysXmlDoc
        The XSD model.

    """
    # The coregistration configurations.
    coreg_stack_conf = are_conf.StaprocessorConfType(
        full_accuracy_pre_processing_conf=translate_full_accuracy_preproc_conf_to_model(
            conf.full_accuracy_preproc_conf,
        ),
        full_accuracy_post_processing_conf=translate_full_accuracy_postproc_conf_type_to_model(
            conf.full_accuracy_postproc_conf,
        ),
        reinterpolation_conf=translate_reinterpolation_conf_type_to_model(
            conf.reinterp_conf,
        ),
        coregistration_output_products_conf=translate_coregistration_output_products_conf_type_to_model(
            conf.coreg_output_products_conf
        ),
        coreg_mode=conf.coreg_mode.value,
        earth_geometry="DEM",  # DEM or WGS84, we always rely on the XYZ prodouct.
        remove_temporary_products=conf.temp_remove_flag,
        memory_sargeometry=conf.memory_sar_geometry,
        digital_elevation_model_repository="",
        xsd_schema_repository="",
    )

    return are_conf.AresysXmlDoc(
        number_of_channels=1,
        version_number="2.1",
        description="Stack coreg configuration",
        channel=[
            are_conf.AresysXmlDocType.Channel(
                staprocessor_conf=[coreg_stack_conf],
                number=1,
                total=1,
            )
        ],
    )


def translate_log_conf_to_model(
    log_conf: GeneralCoregStackProcessorInternalConfiguration.LoggerConfType,
) -> are_conf.LoggerConfType:
    """
    Translate the logger's configuration to XSD model.

    Parameters
    ----------
    log_conf: GeneralCoregStackProcessorInternalConfiguration.LoggerConfType
        The logger configuration object.

    Return
    ------
    aresys_configuration_models.LoggerConfType
        The XSD model.

    """
    report_level = None
    if log_conf.report_level:
        report_level = log_conf.report_level.value
    log_conf_model = are_conf.LoggerConfType(
        enable_log_file=log_conf.enable_log_file,
        enable_std_output=log_conf.enable_std_out,
        report_level=report_level,
    )
    return log_conf_model


def _optionally_to_str(path: Path | None) -> str | None:
    """Cast Path to str if not None"""
    return str(path) if path is not None else None
