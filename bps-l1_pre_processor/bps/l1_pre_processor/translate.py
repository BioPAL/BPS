# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Translation module
------------------
"""

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import aresys_configuration_models, aresys_inputfile_models
from bps.l1_pre_processor.configuration import (
    L1PreProcessorConfiguration,
    L1PreProcessorConfigurationFile,
)
from bps.l1_pre_processor.input_file import L1PreProcessorInputFile


def translate_time_of_interest_to_model(
    time_of_interest: tuple[PreciseDateTime, PreciseDateTime],
) -> aresys_inputfile_models.TimeOfInterestType:
    """Time of interest translate"""
    return aresys_inputfile_models.TimeOfInterestType(
        start=aresys_inputfile_models.TimeOfInterestType.Start(value=str(time_of_interest[0])),
        stop=aresys_inputfile_models.TimeOfInterestType.Stop(value=str(time_of_interest[1])),
    )


def translate_l1preprocessor_input_file_to_model(
    input_file: L1PreProcessorInputFile,
) -> aresys_inputfile_models.AresysXmlInput:
    """Translate the L1 PreProcessor input file object to the XSD model structure

    Parameters
    ----------
    input_file : L1PreProcessorInputFile
        L1 PreProcessor input file object

    Returns
    -------
    aresys_inputfile_models.AresysXmlInput
        Corresponding XSD model structure
    """
    biomass_step = aresys_inputfile_models.BiomassL0ImportPreProcType(
        input_l0_sproduct=str(input_file.input_l0s_product),
        input_aux_orb_file=str(input_file.input_aux_orb_file),
        input_aux_att_file=str(input_file.input_aux_att_file),
        input_aux_ins_file=str(input_file.input_aux_ins_file),
        bpsconfiguration_file=str(input_file.bps_configuration_file),
        bpslog_file=str(input_file.bps_log_file),
        configuration_file=str(input_file.input_configuration_file),
        output_raw_data_product=str(input_file.output_raw_data_product),
    )

    if input_file.input_iersbullettin_file is not None:
        biomass_step.input_iersbullettin_file = str(input_file.input_iersbullettin_file)

    if input_file.time_of_interest is not None:
        biomass_step.time_of_interest = translate_time_of_interest_to_model(input_file.time_of_interest)

    if input_file.input_l0m_product:
        biomass_step.input_l0_mproduct = str(input_file.input_l0m_product)

    if input_file.intermediate_dyn_cal_product:
        biomass_step.intermediate_dyn_cal_product = str(input_file.intermediate_dyn_cal_product)

    if input_file.intermediate_pgpproduct:
        biomass_step.intermediate_pgpproduct = str(input_file.intermediate_pgpproduct)

    if input_file.output_tx_power_tracking_product:
        biomass_step.output_tx_power_tracking_product = str(input_file.output_tx_power_tracking_product)

    if input_file.output_chirp_replica_product:
        biomass_step.output_chirp_replica_product = str(input_file.output_chirp_replica_product)

    if input_file.output_per_line_correction_factors_product:
        biomass_step.output_per_line_correction_factors_product = str(
            input_file.output_per_line_correction_factors_product
        )

    if input_file.output_est_noise_product:
        biomass_step.output_est_noise_product = str(input_file.output_est_noise_product)

    if input_file.output_channel_delays_file:
        biomass_step.output_channel_delays_file = str(input_file.output_channel_delays_file)

    if input_file.output_channel_imbalance_file:
        biomass_step.output_channel_imbalance_file = str(input_file.output_channel_imbalance_file)

    if input_file.output_ssp_headers_file:
        biomass_step.output_sspheaders_file = aresys_inputfile_models.OutputSspheadersFileType(
            str(input_file.output_ssp_headers_file),
            format=aresys_inputfile_models.OutputSspheadersFileTypeFormat.CSV,
        )

    if input_file.output_report_file:
        biomass_step.output_report_file = str(input_file.output_report_file)

    return aresys_inputfile_models.AresysXmlInput(
        [aresys_inputfile_models.AresysXmlInputType.Step(biomass_l0_import_pre_proc=biomass_step, number=1, total=1)]
    )


def translate_l1preprocessor_configuration_to_model(
    conf: L1PreProcessorConfiguration,
) -> aresys_configuration_models.BiomassL0ImportPreProcConfType:
    """Translate L1 pre processor configuration to the XSD model structure

    Parameters
    ----------
    conf : L1PreProcessorConfiguration
        configuration object

    Returns
    -------
    aresys_configuration_models.BiomassL0ImportPreProcConfType
        configuration XSD model
    """
    l0_import_conf = aresys_configuration_models.BiomassL0ImportConfType(
        max_ispgap=conf.max_isp_gap,
        max_time_gap=conf.max_isp_gap,  # max_time_gap is deprecated
        raw_mean_expected=conf.raw_mean_expected,
        raw_mean_threshold=conf.raw_mean_threshold,
        raw_std_expected=conf.raw_std_expected,
        raw_std_threshold=conf.raw_std_threshold,
        beam=conf.swath,
    )

    raw_data_corrections_conf = aresys_configuration_models.BiomassRawDataCorrectionsConfType(
        correct_bias=conf.correct_bias,
        correct_gain_imbalance=conf.correct_gain_imbalance,
        correct_non_orthogonality=conf.correct_non_orthogonality,
        beam=conf.swath,
    )

    int_cal_conf = aresys_configuration_models.BiomassIntCalConfType(
        aresys_configuration_models.BiomassIntCalConfTypeInternalCalibrationSource(
            conf.internal_calibration_source.value
        ),
        max_drift_amplitude_std_fraction=conf.max_drift_amplitude_std_fraction,
        max_drift_phase_std_fraction=conf.max_drift_phase_std_fraction,
        max_drift_amplitude_error=conf.max_drift_amplitude_error,
        max_drift_phase_error=conf.max_drift_phase_error,
        max_invalid_drift_fraction=conf.max_invalid_drift_fraction,
        beam=conf.swath,
    )

    return aresys_configuration_models.BiomassL0ImportPreProcConfType(
        biomass_l0_import_conf=l0_import_conf,
        biomass_raw_data_corrections_conf=raw_data_corrections_conf,
        biomass_int_cal_conf=int_cal_conf,
        enable_channel_delays_annotation=conf.enable_channel_delays_annotation,
        enable_int_cal=conf.enable_internal_calibration,
        beam=conf.swath,
    )


def translate_l1preprocessor_configuration_file_to_model(
    conf: L1PreProcessorConfigurationFile,
) -> aresys_configuration_models.AresysXmlDoc:
    """Translate L1 pre processor configuration file ojbject to the XSD model structure

    Parameters
    ----------
    conf : L1PreProcessorConfigurationFile
        configuration file object

    Returns
    -------
    aresys_configuration_models.AresysXmlDoc
        configuration file XSD model
    """
    configurations = [translate_l1preprocessor_configuration_to_model(c) for c in conf.configurations]

    return aresys_configuration_models.AresysXmlDoc(
        number_of_channels=1,
        version_number=2.6,
        description="L1 Pre Processor processing parameters",
        channel=[
            aresys_configuration_models.AresysXmlDocType.Channel(
                biomass_l0_import_pre_proc_conf=configurations,
                number=1,
                total=1,
            )
        ],
    )
