# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Internal Interface Utilities
----------------------------
"""

from pathlib import Path

from bps.common.io import common
from bps.stack_coreg_processor.configuration import (
    CoregistrationOutputProductsConfType,
    CoregStackProcessorInternalConfiguration,
    FullAccuracyPostProcessingConfType,
    FullAccuracyPreProcessingConfType,
    NonStationaryAreasConfType,
)
from bps.stack_coreg_processor.input_file import (
    BPSCoregProcessorInputFile,
    CoregProcessorInputFile,
)
from bps.stack_coreg_processor.interface import StackCoregProcInterfaceFiles
from bps.stack_processor.interface.external.aux_pps import (
    AuxiliaryStaprocessingParameters,
    CoregistrationConf,
)
from bps.stack_processor.interface.internal.intermediates import (
    CoregistrationOutputProducts,
    StackPreProcessorOutputProducts,
)

# Map AUX-PPS configuration values to Aresys configuration values,
# whenever they mismatch.
AUX_PPS_TO_COREG_CONF = {
    common.CoregistrationMethodType.AUTOMATIC: CoregStackProcessorInternalConfiguration.CoregMode.AUTOMATIC,
    common.CoregistrationMethodType.GEOMETRY: CoregStackProcessorInternalConfiguration.CoregMode.GEOMETRY,
    common.CoregistrationMethodType.GEOMETRY_AND_DATA: CoregStackProcessorInternalConfiguration.CoregMode.FULL_ACCURACY,
}


def fill_stack_coreg_processor_config(
    aux_pps: AuxiliaryStaprocessingParameters,
    coregistration_method: common.CoregistrationMethodType,
    execution_policy: common.CoregistrationExecutionPolicyType | None = None,
    export_distance_product: bool = True,
) -> CoregStackProcessorInternalConfiguration:
    """
    Fill the coregistration configuration.

    Parameters
    ----------
    aux_pps: AuxiliaryStaprocessingParameters
        The AUX-PPS configuration object.

    coregistration_method: common.CoregistrationMethodType
        The coregistration method.

    execution_policy: Optional[common.CoregistrationExecutionPolicyType] = None
        Optionally, the coregistration execution policy. Takes that of the
        aux_pps if None is provided.

    export_distance_product: bool = True
        As to whether the products containing the distance from the
        primary grid to the primary trajectory (_DAPD) should be exported or not.

    Return
    ------
    CoregStackProecssingConfig
        The coregistrator configuration.

    """
    return CoregStackProcessorInternalConfiguration(
        coreg_mode=AUX_PPS_TO_COREG_CONF[coregistration_method],
        full_accuracy_preproc_conf=fill_full_accuracy_preproc_conf(aux_pps.coregistration),
        full_accuracy_postproc_conf=fill_full_accuracy_postproc_conf(
            aux_pps.coregistration,
            nonstationary_flag=not (
                coregistration_method is common.CoregistrationMethodType.GEOMETRY
                or aux_pps.coregistration.model_based_fit_flag
            ),
        ),
        coreg_output_products_conf=fill_coreg_output_products_conf(
            coregistration_execution_policy=(
                aux_pps.coregistration.coregistration_execution_policy if execution_policy is None else execution_policy
            ),
            coregistration_method=coregistration_method,
            export_distance_product=export_distance_product,
        ),
    )


def fill_coreg_output_products_conf(
    *,
    coregistration_execution_policy: common.CoregistrationExecutionPolicyType,
    coregistration_method: common.CoregistrationMethodType,
    export_distance_product: bool,
) -> CoregistrationOutputProductsConfType:
    """
    Populate a CoregistrationOutputProductsContType object.

    Parameters
    ----------
    coregistration_execution_policy: common.CoregistrationExecutionPolicyType
        The execution policy of the coregistrator.

    coregistration_method: common.CoregistrationMethodType
        The coregistration method.

    export_distance_product: bool
        Export the absolute distance product (_DAPD).

    Return
    ------
    CoregistrationConfType
        The coregistration configuration object.

    """
    return CoregistrationOutputProductsConfType(
        provide_coregistration_shifts=int(
            coregistration_execution_policy is not common.CoregistrationExecutionPolicyType.WARPING_ONLY
        ),
        provide_coregistration_accuracy_stats=int(
            coregistration_execution_policy is not common.CoregistrationExecutionPolicyType.WARPING_ONLY
        ),
        shifts_only_estimation=int(
            coregistration_execution_policy is common.CoregistrationExecutionPolicyType.SHIFT_ESTIMATION_ONLY
        ),
        provide_absolute_primary_distance=int(
            export_distance_product
            and coregistration_execution_policy is not common.CoregistrationExecutionPolicyType.WARPING_ONLY
        ),
        provide_wavenumbers=int(
            coregistration_execution_policy is not common.CoregistrationExecutionPolicyType.WARPING_ONLY
        ),
        provide_geometry_shifts=int(
            AUX_PPS_TO_COREG_CONF[coregistration_method] is not common.CoregistrationMethodType.GEOMETRY
            and coregistration_execution_policy is not common.CoregistrationExecutionPolicyType.WARPING_ONLY
        ),
    )


def fill_full_accuracy_preproc_conf(
    coreg_conf: CoregistrationConf,
) -> FullAccuracyPreProcessingConfType:
    """
    Fill a the full-accuracy pre-processing configuration object
    from AUX-PPS.

    Parameters
    ----------
    coreg_conf: CoregistrationConf
        The coreg configuration from AUX-PPS.

    Return
    ------
    FullAccuracyPreProcessingConfType
        The output configuration object.

    """
    return FullAccuracyPreProcessingConfType(
        coreg_reference_polarization=coreg_conf.polarization_used,
        enable_common_band_range_filter=int(coreg_conf.range_spectral_filtering_flag),
        azimuth_max_shift=coreg_conf.azimuth_max_shift,
        azimuth_block_size=coreg_conf.azimuth_block_size,
        azimuth_min_overlap=coreg_conf.azimuth_min_overlap,
        range_max_shift=coreg_conf.range_max_shift,
        range_block_size=coreg_conf.range_block_size,
        report_level=_aux_pps_debug_flag_to_coreg_report_level(coreg_conf.export_debug_products_flag),
    )


def fill_full_accuracy_postproc_conf(
    coreg_conf: CoregistrationConf,
    nonstationary_flag: bool,
) -> FullAccuracyPostProcessingConfType:
    """
    Fill a the full-accuracy post-processing configuration object
    from AUX-PPS.

    Parameters
    ----------
    coreg_conf: CoregistrationConf
        The coreg configuration from AUX-PPS.

    nonstationary_flag: bool
        If true, write the non-stationary areas configuration.

    Return
    ------
    FullAccuracyPostProcessingConfType
        The output configuration object.

    """
    return FullAccuracyPostProcessingConfType(
        quality_threshold_for_automatic_mode=coreg_conf.fitting_quality_threshold,
        min_valid_blocks=coreg_conf.min_valid_blocks,
        non_stationary_coreg_conf=NonStationaryAreasConfType(
            low_pass_filter_type=coreg_conf.low_pass_filter_type,
            low_pass_filter_order=coreg_conf.low_pass_filter_order,
            low_pass_filter_std_dev=coreg_conf.low_pass_filter_std_dev,
        )
        if nonstationary_flag
        else None,
    )


def fill_stack_coreg_processor_input_files(
    *,
    stack_pre_proc_output_products: tuple[StackPreProcessorOutputProducts, ...],
    coreg_output_products: tuple[CoregistrationOutputProducts, ...],
    coreg_proc_interface_files: StackCoregProcInterfaceFiles,
    bps_configuration_file: Path,
    bps_log_file: Path,
    coreg_primary_image_index: int,
    warping_only: bool = False,
) -> tuple[BPSCoregProcessorInputFile, ...]:
    """Fill the input file of the Stack Coregistration Processor."""
    # The pre-processor product associated to the coreg primary.
    primary_product = stack_pre_proc_output_products[coreg_primary_image_index]
    return tuple(
        BPSCoregProcessorInputFile(
            coregistration_input=CoregProcessorInputFile(
                primary_product=primary_product.raw_data_product,
                secondary_product=scs_product.raw_data_product,
                ecef_grid_product=primary_product.xyz_product,
                output_path=coreg_output_product.coreg_product.parent,
                coreg_conf_file=(
                    coreg_proc_interface_files.coreg_primary_config_file
                    if index == coreg_primary_image_index
                    else coreg_proc_interface_files.coreg_config_file
                ),
                rg_shifts_product=(coreg_output_product.rg_shifts_product if warping_only else None),
                az_shifts_product=(coreg_output_product.az_shifts_product if warping_only else None),
            ),
            bps_configuration_file=bps_configuration_file,
            bps_log_file=bps_log_file,
        )
        for index, (scs_product, coreg_output_product) in enumerate(
            zip(stack_pre_proc_output_products, coreg_output_products)
        )
    )


def _aux_pps_debug_flag_to_coreg_report_level(
    export_debug_products_flag: bool,
    *,
    nominal_level: int = 0,
    debug_level: int = 4,
) -> int:
    """
    The coregistrator offers several debug levels from 0 to 4, where 0
    is no reporting (nominal) and 4 is run exporting debug symbols. Those
    are the only reporting levels supported by STA_P.
    """
    return debug_level if export_debug_products_flag else nominal_level
