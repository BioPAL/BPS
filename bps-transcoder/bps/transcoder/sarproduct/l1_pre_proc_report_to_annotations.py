# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Utilities for L1 preprocessor related annotations"""

from dataclasses import dataclass, field
from typing import Literal, Self

import numpy as np
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io import common
from bps.transcoder.io import common_annotation_l1
from bps.transcoder.io.preprocessor_report import L1PreProcAnnotations
from bps.transcoder.sarproduct.biomass_l1product import QualityParameters


@dataclass
class RawDataStatistics:
    """Raw data statistics"""

    bias_i: float = 0.0
    bias_q: float = 0.0
    std_dev_i: float = 0.0
    std_dev_q: float = 0.0
    quadrature_departure: float = 0.0
    gain_imbalance: float = 0.0
    polarization: str = "H/H"


@dataclass
class RawDataAnalysis:
    """Raw data analysis"""

    num_isp_header_errors: int = 0
    num_isp_missing: int = 0
    raw_data_statistics_list: list[RawDataStatistics] = field(default_factory=list)

    @classmethod
    def from_polarization_list(cls, polarization_list: list[Literal["H/H", "V/V", "H/V", "V/H"]]) -> Self:
        """Create a raw data analysis instance from the polarization list"""
        return cls(
            raw_data_statistics_list=[
                RawDataStatistics(polarization=polarization) for polarization in polarization_list
            ]
        )

    @classmethod
    def from_report(cls, report: L1PreProcAnnotations) -> Self:
        """Fill the raw data analysis from the report"""
        return cls(
            num_isp_header_errors=report.corrupted_packets,
            num_isp_missing=report.missing_packets,
            raw_data_statistics_list=[
                RawDataStatistics(
                    bias_i=stats.bias_i,
                    bias_q=stats.bias_q,
                    std_dev_i=stats.std_dev_i,
                    std_dev_q=stats.std_dev_q,
                    quadrature_departure=stats.quadrature_departure,
                    gain_imbalance=stats.std_dev_i / stats.std_dev_q,
                    polarization=stats.polarization,
                )
                for stats in report.raw_data_statistics
            ],
        )

    def to_annotation(self) -> common_annotation_l1.RawDataAnalysisType:
        """Convert the raw data analysis to the annotation type"""
        polarisation_dict = {
            "H/H": common.PolarisationType.HH,
            "H/V": common.PolarisationType.HV,
            "V/H": common.PolarisationType.VH,
            "V/V": common.PolarisationType.VV,
        }

        return common_annotation_l1.RawDataAnalysisType(
            error_counters=common_annotation_l1.ErrorCountersType(
                num_isp_header_errors=self.num_isp_header_errors,
                num_isp_missing=self.num_isp_missing,
            ),
            raw_data_statistics_list=[
                common_annotation_l1.RawDataStatisticsType(
                    i_bias=stats.bias_i,
                    q_bias=stats.bias_q,
                    iq_quadrature_departure=stats.quadrature_departure,
                    iq_gain_imbalance=stats.gain_imbalance,
                    polarisation=polarisation_dict[stats.polarization],
                )
                for stats in self.raw_data_statistics_list
            ],
        )


@dataclass
class InternalCalibrationSequence:
    """Internal calibration sequence"""

    time: PreciseDateTime
    drift: complex = complex(1.0, 0.0)
    channel_delay: float = 0.0
    channel_imbalance_tx: complex = complex(1.0, 0.0)
    channel_imbalance_rx: complex = complex(1.0, 0.0)
    transmit_power_tracking_d1: complex = complex(0.7071067811865475, 0.0)
    receive_power_tracking_d1: complex = complex(0.7071067811865475, 0.0)
    transmit_power_tracking_d2: complex = complex(0.7071067811865475, 0.0)
    receive_power_tracking_d2: complex = complex(0.7071067811865475, 0.0)

    model_drift: complex = complex(1.0, 0.0)
    relative_drift_valid_flag: bool = True
    absolute_drift_valid_flag: bool = True
    cross_correlation_bandwidth: float = 6000000.0
    cross_correlation_pslr: float = -13.2
    cross_correlation_islr: float = -10.0
    cross_correlation_peak_location: float = 0.0
    reconstructed_replica_valid_flag: bool = True


def fill_internal_calibration_sequence(
    sequence: InternalCalibrationSequence,
) -> common_annotation_l1.InternalCalibrationSequenceType:
    """Fill internal calibration sequence type"""
    return common_annotation_l1.InternalCalibrationSequenceType(
        azimuth_time=sequence.time,
        drift_amplitude=abs(sequence.drift),
        drift_phase=float(np.angle(sequence.drift)),
        model_drift_amplitude=abs(sequence.model_drift),
        model_drift_phase=float(np.angle(sequence.model_drift)),
        relative_drift_valid_flag=sequence.relative_drift_valid_flag,
        absolute_drift_valid_flag=sequence.absolute_drift_valid_flag,
        cross_correlation_bandwidth=sequence.cross_correlation_bandwidth,
        cross_correlation_pslr=sequence.cross_correlation_pslr,
        cross_correlation_islr=sequence.cross_correlation_islr,
        cross_correlation_peak_location=sequence.cross_correlation_peak_location,
        reconstructed_replica_valid_flag=sequence.reconstructed_replica_valid_flag,
        internal_time_delay=sequence.channel_delay,
        internal_tx_channel_imbalance_amplitude=abs(sequence.channel_imbalance_tx),
        internal_tx_channel_imbalance_phase=float(np.angle(sequence.channel_imbalance_tx)),
        internal_rx_channel_imbalance_amplitude=abs(sequence.channel_imbalance_rx),
        internal_rx_channel_imbalance_phase=float(np.angle(sequence.channel_imbalance_rx)),
        transmit_power_tracking_d1_amplitude=abs(sequence.transmit_power_tracking_d1),
        transmit_power_tracking_d1_phase=float(np.angle(sequence.transmit_power_tracking_d1)),
        receive_power_tracking_d1_amplitude=abs(sequence.receive_power_tracking_d1),
        receive_power_tracking_d1_phase=float(np.angle(sequence.receive_power_tracking_d1)),
        transmit_power_tracking_d2_amplitude=abs(sequence.transmit_power_tracking_d2),
        transmit_power_tracking_d2_phase=float(np.angle(sequence.transmit_power_tracking_d2)),
        receive_power_tracking_d2_amplitude=abs(sequence.receive_power_tracking_d2),
        receive_power_tracking_d2_phase=float(np.angle(sequence.receive_power_tracking_d2)),
    )


@dataclass
class InternalCalibrationParameters:
    """Internal calibration parameters"""

    sequences: dict[Literal["H/H", "H/V", "V/H", "V/V"], list[InternalCalibrationSequence]] = field(
        default_factory=dict
    )

    @classmethod
    def from_report(
        cls,
        report: L1PreProcAnnotations,
        polarization_list: list[Literal["H/H", "H/V", "V/H", "V/V"]],
        quality_parameters: QualityParameters,
    ) -> Self:
        """Fill the internal calibration parameters from the report"""
        sequences: dict[Literal["H/H", "H/V", "V/H", "V/V"], list[InternalCalibrationSequence]] = {}
        for polarization in polarization_list:
            tx_pol, rx_pol = polarization.split("/")

            channel_delay = report.channel_delays.get(polarization, 0.0)

            chirp_replica_parameters = report.chirp_replica_parameters.get(polarization)
            if chirp_replica_parameters is not None:
                cross_correlation_bandwidth = chirp_replica_parameters.bandwidth
                cross_correlation_pslr = chirp_replica_parameters.pslr
                cross_correlation_islr = chirp_replica_parameters.islr
                cross_correlation_peak_location = chirp_replica_parameters.location_error
                reconstructed_replica_valid_flag = chirp_replica_parameters.validity_flag
            else:
                cross_correlation_bandwidth = InternalCalibrationSequence.cross_correlation_bandwidth
                cross_correlation_pslr = InternalCalibrationSequence.cross_correlation_pslr
                cross_correlation_islr = InternalCalibrationSequence.cross_correlation_islr
                cross_correlation_peak_location = InternalCalibrationSequence.cross_correlation_peak_location
                reconstructed_replica_valid_flag = InternalCalibrationSequence.reconstructed_replica_valid_flag

            drift = [data.drifts.get(polarization, complex(0.0, 0.0)) for data in report.int_cal_sequences]
            if drift:
                drift_amplitude_mean = np.mean(np.abs(drift))
                drift_amplitude_std = np.std(np.abs(drift))
                drift_phase_mean = np.mean(np.angle(drift))
                drift_phase_std = np.std(np.angle(drift))
                drift_amplitude_relative_deviation = [
                    np.abs(np.abs(d) - drift_amplitude_mean) / drift_amplitude_std for d in drift
                ]
                drift_phase_relative_deviation = [
                    np.abs(np.angle(d) - drift_phase_mean) / drift_phase_std for d in drift
                ]
                drift_relative_check = [
                    True
                    if (
                        a <= quality_parameters.max_drift_amplitude_std_fraction
                        and p <= quality_parameters.max_drift_phase_std_fraction
                    )
                    else False
                    for a, p in zip(drift_amplitude_relative_deviation, drift_phase_relative_deviation)
                ]
                drift_amplitude_absolute_deviation = [
                    np.abs(np.abs(d) / drift_amplitude_mean - np.abs(InternalCalibrationSequence.model_drift))
                    for d in drift
                ]
                drift_phase_absolute_deviation = [
                    np.abs(np.angle(d) - drift_phase_mean - np.angle(InternalCalibrationSequence.model_drift))
                    for d in drift
                ]
                drift_absolute_check = [
                    True
                    if (
                        a <= quality_parameters.max_drift_amplitude_error
                        and p <= quality_parameters.max_drift_phase_error
                    )
                    else False
                    for a, p in zip(drift_amplitude_absolute_deviation, drift_phase_absolute_deviation)
                ]
            else:
                drift_relative_check = [True] * len(report.int_cal_sequences)
                drift_absolute_check = [True] * len(report.int_cal_sequences)

            sequences[polarization] = [
                InternalCalibrationSequence(
                    time=data.time,
                    drift=data.drifts.get(polarization, complex(0.0, 0.0)),
                    relative_drift_valid_flag=drift_rel_check,
                    absolute_drift_valid_flag=drift_abs_check,
                    cross_correlation_bandwidth=cross_correlation_bandwidth,
                    cross_correlation_pslr=cross_correlation_pslr,
                    cross_correlation_islr=cross_correlation_islr,
                    cross_correlation_peak_location=cross_correlation_peak_location,
                    reconstructed_replica_valid_flag=reconstructed_replica_valid_flag,
                    channel_delay=channel_delay,
                    channel_imbalance_tx=report.channel_imbalance_tx,
                    channel_imbalance_rx=report.channel_imbalance_rx,
                    transmit_power_tracking_d1=data.excitation_coeffs.get((tx_pol, "D1", "TX"), complex(0.0, 0.0)),
                    transmit_power_tracking_d2=data.excitation_coeffs.get((tx_pol, "D2", "TX"), complex(0.0, 0.0)),
                    receive_power_tracking_d1=data.excitation_coeffs.get((rx_pol, "D1", "RX"), complex(0.0, 0.0)),
                    receive_power_tracking_d2=data.excitation_coeffs.get((rx_pol, "D2", "RX"), complex(0.0, 0.0)),
                )
                for data, drift_rel_check, drift_abs_check in zip(
                    report.int_cal_sequences, drift_relative_check, drift_absolute_check
                )
            ]
        return cls(sequences=sequences)

    @classmethod
    def from_polarization_list(
        cls, polarization_list: list[Literal["H/H", "H/V", "V/H", "V/V"]], times: list[PreciseDateTime]
    ) -> Self:
        """Create the internal calibration parameters from the polarization list"""
        sequences: dict[Literal["H/H", "H/V", "V/H", "V/V"], list[InternalCalibrationSequence]] = {
            polarization: [InternalCalibrationSequence(time=time) for time in times]
            for polarization in polarization_list
        }
        return cls(sequences=sequences)

    def to_annotations(
        self,
    ) -> dict[common.PolarisationType, list[common_annotation_l1.InternalCalibrationSequenceType]]:
        """Convert the internal calibration parameters to annotations"""
        polarisation_dict = {
            "H/H": common.PolarisationType.HH,
            "H/V": common.PolarisationType.HV,
            "V/H": common.PolarisationType.VH,
            "V/V": common.PolarisationType.VV,
        }

        return {
            polarisation_dict[polarization]: [fill_internal_calibration_sequence(sequence) for sequence in sequences]
            for polarization, sequences in self.sequences.items()
        }
