# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD BPS l1 preprocesso annotation xml file models
-------------------------------------------------
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CompressionLevelType(Enum):
    BAQ_4_BIT = "BAQ 4 Bit"
    BAQ_5_BIT = "BAQ 5 Bit"
    BAQ_6_BIT = "BAQ 6 Bit"
    BYPASS = "Bypass"


@dataclass
class NoiseSequenceType:
    reference_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "ReferenceTime",
            "type": "Element",
            "required": True,
        },
    )
    average_noise: Optional[float] = field(
        default=None,
        metadata={
            "name": "AverageNoise",
            "type": "Element",
            "required": True,
        },
    )
    num_lines: Optional[int] = field(
        default=None,
        metadata={
            "name": "NumLines",
            "type": "Element",
            "required": True,
        },
    )


class PowerTrackingTypeDoublet(Enum):
    D1 = "D1"
    D2 = "D2"


class PowerTrackingTypePolarization(Enum):
    H = "H"
    V = "V"


class PowerTrackingTypeRole(Enum):
    TX = "TX"
    RX = "RX"


@dataclass
class FcomplexNumberType:
    class Meta:
        name = "FComplexNumberType"
        target_namespace = "biomass_common"

    real: Optional[float] = field(
        default=None,
        metadata={
            "name": "Real",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    imag: Optional[float] = field(
        default=None,
        metadata={
            "name": "Imag",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


class PolarizationType(Enum):
    H_H = "H/H"
    H_V = "H/V"
    V_H = "V/H"
    V_V = "V/V"


class SwathType(Enum):
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"


class UnitType(Enum):
    S = "s"
    UTC = "Utc"


@dataclass
class DriftType:
    real: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )
    imag: Optional[float] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        },
    )
    polarization: Optional[PolarizationType] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class PowerTrackingType(FcomplexNumberType):
    role: Optional[PowerTrackingTypeRole] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    doublet: Optional[PowerTrackingTypeDoublet] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )
    polarization: Optional[PowerTrackingTypePolarization] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class ChannelDelayType:
    class Meta:
        target_namespace = "biomass_common"

    value: Optional[float] = field(
        default=None,
        metadata={
            "required": True,
        },
    )
    swath: Optional[SwathType] = field(
        default=None,
        metadata={
            "name": "Swath",
            "type": "Attribute",
            "required": True,
        },
    )
    polarization: Optional[PolarizationType] = field(
        default=None,
        metadata={
            "name": "Polarization",
            "type": "Attribute",
            "required": True,
        },
    )
    uo_m: UnitType = field(
        init=False,
        default=UnitType.S,
        metadata={
            "name": "UoM",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class ChannelImbalanceType:
    class Meta:
        target_namespace = "biomass_common"

    tx: Optional[FcomplexNumberType] = field(
        default=None,
        metadata={
            "name": "TX",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    rx: Optional[FcomplexNumberType] = field(
        default=None,
        metadata={
            "name": "RX",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class ChannelDelaysType:
    class Meta:
        target_namespace = "biomass_common"

    channel_delay: list[ChannelDelayType] = field(
        default_factory=list,
        metadata={
            "name": "ChannelDelay",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass
class L1PreProcessorAnnotations:
    ispformat: Optional["L1PreProcessorAnnotations.Ispformat"] = field(
        default=None,
        metadata={
            "name": "ISPFormat",
            "type": "Element",
            "required": True,
        },
    )
    isperrors: Optional["L1PreProcessorAnnotations.Isperrors"] = field(
        default=None,
        metadata={
            "name": "ISPErrors",
            "type": "Element",
            "required": True,
        },
    )
    raw_data_statistics: list["L1PreProcessorAnnotations.RawDataStatistics"] = field(
        default_factory=list,
        metadata={
            "name": "RawDataStatistics",
            "type": "Element",
            "max_occurs": 4,
        },
    )
    gain_param_code_h: Optional[int] = field(
        default=None,
        metadata={
            "name": "GainParamCodeH",
            "type": "Element",
            "required": True,
        },
    )
    gain_param_code_v: Optional[int] = field(
        default=None,
        metadata={
            "name": "GainParamCodeV",
            "type": "Element",
            "required": True,
        },
    )
    noise_data: Optional["L1PreProcessorAnnotations.NoiseData"] = field(
        default=None,
        metadata={
            "name": "NoiseData",
            "type": "Element",
            "required": True,
        },
    )
    channel_delays: Optional[ChannelDelaysType] = field(
        default=None,
        metadata={
            "name": "ChannelDelays",
            "type": "Element",
            "required": True,
        },
    )
    channel_imbalance: Optional[ChannelImbalanceType] = field(
        default=None,
        metadata={
            "name": "ChannelImbalance",
            "type": "Element",
            "required": True,
        },
    )
    chirp_replica_parameters: list["L1PreProcessorAnnotations.ChirpReplicaParameters"] = field(
        default_factory=list,
        metadata={
            "name": "ChirpReplicaParameters",
            "type": "Element",
            "max_occurs": 4,
        },
    )
    internal_calibration_data: list["L1PreProcessorAnnotations.InternalCalibrationData"] = field(
        default_factory=list,
        metadata={
            "name": "InternalCalibrationData",
            "type": "Element",
            "min_occurs": 1,
        },
    )
    noise_preamble_h: Optional[NoiseSequenceType] = field(
        default=None,
        metadata={
            "name": "NoisePreambleH",
            "type": "Element",
            "required": True,
        },
    )
    noise_postamble_h: Optional[NoiseSequenceType] = field(
        default=None,
        metadata={
            "name": "NoisePostambleH",
            "type": "Element",
            "required": True,
        },
    )
    noise_preamble_v: Optional[NoiseSequenceType] = field(
        default=None,
        metadata={
            "name": "NoisePreambleV",
            "type": "Element",
            "required": True,
        },
    )
    noise_postamble_v: Optional[NoiseSequenceType] = field(
        default=None,
        metadata={
            "name": "NoisePostambleV",
            "type": "Element",
            "required": True,
        },
    )

    @dataclass
    class Ispformat:
        echo: Optional[CompressionLevelType] = field(
            default=None,
            metadata={
                "name": "Echo",
                "type": "Element",
                "required": True,
            },
        )
        calibration: Optional[CompressionLevelType] = field(
            default=None,
            metadata={
                "name": "Calibration",
                "type": "Element",
                "required": True,
            },
        )
        noise: Optional[CompressionLevelType] = field(
            default=None,
            metadata={
                "name": "Noise",
                "type": "Element",
                "required": True,
            },
        )
        mean_bit_rate: Optional[float] = field(
            default=None,
            metadata={
                "name": "MeanBitRate",
                "type": "Element",
                "required": True,
            },
        )

    @dataclass
    class Isperrors:
        num_missing_packets: Optional[int] = field(
            default=None,
            metadata={
                "name": "NumMissingPackets",
                "type": "Element",
                "required": True,
            },
        )
        num_corrupted_packets: Optional[int] = field(
            default=None,
            metadata={
                "name": "NumCorruptedPackets",
                "type": "Element",
                "required": True,
            },
        )

    @dataclass
    class RawDataStatistics:
        mean_i: Optional[float] = field(
            default=None,
            metadata={
                "name": "Mean_i",
                "type": "Element",
                "required": True,
            },
        )
        mean_q: Optional[float] = field(
            default=None,
            metadata={
                "name": "Mean_q",
                "type": "Element",
                "required": True,
            },
        )
        std_dev_i: Optional[float] = field(
            default=None,
            metadata={
                "name": "StdDev_i",
                "type": "Element",
                "required": True,
            },
        )
        std_dev_q: Optional[float] = field(
            default=None,
            metadata={
                "name": "StdDev_q",
                "type": "Element",
                "required": True,
            },
        )
        quadrature_departure: Optional[float] = field(
            default=None,
            metadata={
                "name": "QuadratureDeparture",
                "type": "Element",
                "required": True,
            },
        )
        polarization: Optional[PolarizationType] = field(
            default=None,
            metadata={
                "name": "Polarization",
                "type": "Element",
                "required": True,
            },
        )

    @dataclass
    class NoiseData:
        preamble_present: Optional[bool] = field(
            default=None,
            metadata={
                "name": "PreamblePresent",
                "type": "Element",
                "required": True,
            },
        )
        postamble_present: Optional[bool] = field(
            default=None,
            metadata={
                "name": "PostamblePresent",
                "type": "Element",
                "required": True,
            },
        )

    @dataclass
    class ChirpReplicaParameters:
        bandwidth: Optional[float] = field(
            default=None,
            metadata={
                "name": "Bandwidth",
                "type": "Element",
                "required": True,
            },
        )
        pslr: Optional[float] = field(
            default=None,
            metadata={
                "name": "PSLR",
                "type": "Element",
                "required": True,
            },
        )
        islr: Optional[float] = field(
            default=None,
            metadata={
                "name": "ISLR",
                "type": "Element",
                "required": True,
            },
        )
        location_error: Optional[float] = field(
            default=None,
            metadata={
                "name": "LocationError",
                "type": "Element",
                "required": True,
            },
        )
        validity_flag: Optional[bool] = field(
            default=None,
            metadata={
                "name": "ValidityFlag",
                "type": "Element",
                "required": True,
            },
        )
        polarization: Optional[PolarizationType] = field(
            default=None,
            metadata={
                "name": "Polarization",
                "type": "Element",
                "required": True,
            },
        )

    @dataclass
    class InternalCalibrationData:
        reference_time: Optional[str] = field(
            default=None,
            metadata={
                "name": "ReferenceTime",
                "type": "Element",
                "required": True,
            },
        )
        excitation_coefficients: Optional[
            "L1PreProcessorAnnotations.InternalCalibrationData.ExcitationCoefficients"
        ] = field(
            default=None,
            metadata={
                "name": "ExcitationCoefficients",
                "type": "Element",
                "required": True,
            },
        )
        drifts: Optional["L1PreProcessorAnnotations.InternalCalibrationData.Drifts"] = field(
            default=None,
            metadata={
                "name": "Drifts",
                "type": "Element",
                "required": True,
            },
        )

        @dataclass
        class ExcitationCoefficients:
            power_tracking: list[PowerTrackingType] = field(
                default_factory=list,
                metadata={
                    "name": "PowerTracking",
                    "type": "Element",
                    "max_occurs": 8,
                },
            )

        @dataclass
        class Drifts:
            drift: list[DriftType] = field(
                default_factory=list,
                metadata={
                    "name": "Drift",
                    "type": "Element",
                    "max_occurs": 4,
                },
            )
