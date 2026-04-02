# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to parse a l1 pre processor annotation report
-------------------------------------------------------
"""

from dataclasses import dataclass
from enum import Enum, auto

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.common.io.parsing import ParsingError, parse
from bps.transcoder.io.biomass_l1_preproc_annotations import models


class InvalidL1PreProcAnnotations(RuntimeError):
    """Raised when the annotation content is not the correct format"""


class BAQCompressionLevel(Enum):
    """Different baq compression levels"""

    BAQ_4_BIT = auto()
    BAQ_5_BIT = auto()
    BAQ_6_BIT = auto()
    BYPASS = auto()


@dataclass
class RawDataStatistics:
    """Statistics"""

    bias_i: float
    bias_q: float
    std_dev_i: float
    std_dev_q: float
    quadrature_departure: float
    polarization: str


@dataclass
class ChirpReplicaParameters:
    """Chirp replica parameters"""

    bandwidth: float
    pslr: float
    islr: float
    location_error: float
    validity_flag: bool


@dataclass
class IntCalSequence:
    """Internal calibration sequence data"""

    time: PreciseDateTime
    drifts: dict[str, complex]
    excitation_coeffs: dict[tuple[str, str, str], complex]


@dataclass
class NoiseSequence:
    time: PreciseDateTime
    average_noise: float
    num_lines: int


@dataclass
class IOBPRMeasure:
    """In/out-of-band power ratio measure"""

    time: PreciseDateTime
    power_ratios: dict[str, float]


@dataclass
class L1PreProcAnnotations:
    """Annotations"""

    echo_baq_compression: BAQCompressionLevel
    calibration_baq_compression: BAQCompressionLevel
    noise_baq_compression: BAQCompressionLevel

    corrupted_packets: int
    missing_packets: int

    raw_data_statistics: list[RawDataStatistics]

    gain_param_code_h: int
    gain_param_code_v: int

    noise_preamble_present: bool
    noise_postamble_present: bool

    channel_delays: dict[str, float]
    channel_imbalance_tx: complex
    channel_imbalance_rx: complex

    chirp_replica_parameters: dict[str, ChirpReplicaParameters]

    int_cal_sequences: list[IntCalSequence]

    noise_preamble_h: NoiseSequence
    noise_preamble_v: NoiseSequence
    noise_postamble_h: NoiseSequence
    noise_postamble_v: NoiseSequence

    iobpr_measurements: list[IOBPRMeasure]


def translate_compression_level_from_model(
    compression_level: models.CompressionLevelType,
) -> BAQCompressionLevel:
    """Translate from model"""
    if compression_level == models.CompressionLevelType.BAQ_4_BIT:
        return BAQCompressionLevel.BAQ_4_BIT
    elif compression_level == models.CompressionLevelType.BAQ_5_BIT:
        return BAQCompressionLevel.BAQ_5_BIT
    elif compression_level == models.CompressionLevelType.BAQ_6_BIT:
        return BAQCompressionLevel.BAQ_6_BIT
    elif compression_level == models.CompressionLevelType.BYPASS:
        return BAQCompressionLevel.BYPASS
    else:
        raise ValueError("Invalid compression level")


def translate_raw_data_statistics(
    raw_data_stats: models.L1PreProcessorAnnotations.RawDataStatistics,
) -> RawDataStatistics:
    """Translate from model"""
    return RawDataStatistics(
        bias_i=raw_data_stats.mean_i,
        bias_q=raw_data_stats.mean_q,
        std_dev_i=raw_data_stats.std_dev_i,
        std_dev_q=raw_data_stats.std_dev_q,
        quadrature_departure=raw_data_stats.quadrature_departure,
        polarization=raw_data_stats.polarization.value,
    )


def translate_chirp_replica_parameters(
    chirp_replica_pars: models.L1PreProcessorAnnotations.ChirpReplicaParameters,
) -> ChirpReplicaParameters:
    """Translate from model"""
    return ChirpReplicaParameters(
        bandwidth=chirp_replica_pars.bandwidth,
        pslr=chirp_replica_pars.pslr,
        islr=chirp_replica_pars.islr,
        location_error=chirp_replica_pars.location_error,
        validity_flag=chirp_replica_pars.validity_flag,
    )


def translate_int_cal_sequences_from_model(
    model: models.L1PreProcessorAnnotations.InternalCalibrationData,
) -> IntCalSequence:
    """Translate int cal sequences annotations from model"""
    drifts: dict[str, complex] = {}
    for drift in model.drifts.drift:
        pol = drift.polarization.value
        value = complex(drift.real, drift.imag)
        drifts[pol] = value

    coeffs: dict[tuple[str, str, str], complex] = {}
    for coeff in model.excitation_coefficients.power_tracking:
        pol = coeff.polarization.value
        doublet = coeff.doublet.value
        role = coeff.role.value

        value = complex(coeff.real, coeff.imag)
        coeffs[(pol, doublet, role)] = value

    return IntCalSequence(
        time=PreciseDateTime.from_utc_string(model.reference_time), drifts=drifts, excitation_coeffs=coeffs
    )


def translate_noise_sequence_from_model(model: models.NoiseSequenceType) -> NoiseSequence:
    """Translate noise sequence annotations from model"""
    return NoiseSequence(
        time=PreciseDateTime.from_utc_string(model.reference_time),
        average_noise=model.average_noise,
        num_lines=model.num_lines,
    )


def translate_iobpr_measurements_from_model(
    model: models.L1PreProcessorAnnotations.InOutBandPowerRatioData,
) -> IOBPRMeasure:
    """Translate in/out-of-band power ratio measurements annotations from model"""
    power_ratios: dict[str, float] = {}
    for power_ratio in model.power_ratios.power_ratio:
        pol = power_ratio.polarization.value
        value = power_ratio.value
        power_ratios[pol] = value

    return IOBPRMeasure(time=PreciseDateTime.from_utc_string(model.reference_time), power_ratios=power_ratios)


def translate_annotation_from_model(
    model: models.L1PreProcessorAnnotations,
) -> L1PreProcAnnotations:
    """Translate annotation from model"""
    echo_baq_compression = translate_compression_level_from_model(model.ispformat.echo)
    calibration_baq_compression = translate_compression_level_from_model(model.ispformat.calibration)
    noise_baq_compression = translate_compression_level_from_model(model.ispformat.noise)

    corrupted_packets = model.isperrors.num_corrupted_packets
    missing_packets = model.isperrors.num_missing_packets

    raw_data_statistics = [translate_raw_data_statistics(stats) for stats in model.raw_data_statistics]

    noise_preamble_present = model.noise_data.preamble_present
    noise_postamble_present = model.noise_data.postamble_present

    channel_delays = {}
    for delay in model.channel_delays.channel_delay:
        channel_delays[delay.polarization.value] = delay.value

    channel_imbalance_tx = complex(model.channel_imbalance.tx.real, model.channel_imbalance.tx.imag)
    channel_imbalance_rx = complex(model.channel_imbalance.rx.real, model.channel_imbalance.rx.imag)

    chirp_replica_parameters = {
        pars.polarization.value: translate_chirp_replica_parameters(pars) for pars in model.chirp_replica_parameters
    }

    return L1PreProcAnnotations(
        echo_baq_compression=echo_baq_compression,
        calibration_baq_compression=calibration_baq_compression,
        noise_baq_compression=noise_baq_compression,
        corrupted_packets=corrupted_packets,
        missing_packets=missing_packets,
        raw_data_statistics=raw_data_statistics,
        gain_param_code_h=model.gain_param_code_h,
        gain_param_code_v=model.gain_param_code_v,
        noise_preamble_present=noise_preamble_present,
        noise_postamble_present=noise_postamble_present,
        channel_delays=channel_delays,
        channel_imbalance_tx=channel_imbalance_tx,
        channel_imbalance_rx=channel_imbalance_rx,
        chirp_replica_parameters=chirp_replica_parameters,
        int_cal_sequences=[translate_int_cal_sequences_from_model(data) for data in model.internal_calibration_data],
        noise_preamble_h=translate_noise_sequence_from_model(model.noise_preamble_h),
        noise_preamble_v=translate_noise_sequence_from_model(model.noise_preamble_v),
        noise_postamble_h=translate_noise_sequence_from_model(model.noise_postamble_h),
        noise_postamble_v=translate_noise_sequence_from_model(model.noise_postamble_v),
        iobpr_measurements=[
            translate_iobpr_measurements_from_model(data) for data in model.in_out_band_power_ratio_data
        ],
    )


def parse_annotations(
    annotation_content: str,
) -> L1PreProcAnnotations:
    """
    Parse the annotation xml file content and return the parsed L1PreProcAnnotations object.

    Parameters
    ----------
    annotation_content : str
        The content of the annotation xml file.

    Returns
    -------
    L1PreProcAnnotations
        The parsed L1PreProcAnnotations object.

    Raises
    ------
    InvalidL1PreProcAnnotations
        If the annotation content is invalid and cannot be parsed.
    """
    try:
        model = parse(annotation_content, models.L1PreProcessorAnnotations)
    except ParsingError as exc:
        raise InvalidL1PreProcAnnotations from exc

    return translate_annotation_from_model(model)
