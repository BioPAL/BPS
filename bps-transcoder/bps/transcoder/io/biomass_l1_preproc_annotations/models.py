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

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CompressionLevelType(Enum):
    BAQ_4_BIT = "BAQ 4 Bit"
    BAQ_5_BIT = "BAQ 5 Bit"
    BAQ_6_BIT = "BAQ 6 Bit"
    BYPASS = "Bypass"


@dataclass(kw_only=True)
class NoiseSequenceType:
    reference_time: str = field(
        metadata={
            "name": "ReferenceTime",
            "type": "Element",
        }
    )
    average_noise: float = field(
        metadata={
            "name": "AverageNoise",
            "type": "Element",
        }
    )
    num_lines: int = field(
        metadata={
            "name": "NumLines",
            "type": "Element",
        }
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


@dataclass(kw_only=True)
class FcomplexNumberType:
    class Meta:
        name = "FComplexNumberType"
        target_namespace = "biomass_common"

    real: float = field(
        metadata={
            "name": "Real",
            "type": "Element",
            "namespace": "",
        }
    )
    imag: float = field(
        metadata={
            "name": "Imag",
            "type": "Element",
            "namespace": "",
        }
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


@dataclass(kw_only=True)
class DriftType:
    real: float = field(
        metadata={
            "type": "Element",
        }
    )
    imag: float = field(
        metadata={
            "type": "Element",
        }
    )
    polarization: PolarizationType = field(
        metadata={
            "type": "Attribute",
        }
    )


@dataclass(kw_only=True)
class PowerRatioType:
    value: float = field(
        metadata={
            "type": "Element",
        }
    )
    polarization: PolarizationType = field(
        metadata={
            "type": "Attribute",
        }
    )


@dataclass(kw_only=True)
class PowerTrackingType(FcomplexNumberType):
    role: PowerTrackingTypeRole = field(
        metadata={
            "type": "Attribute",
        }
    )
    doublet: PowerTrackingTypeDoublet = field(
        metadata={
            "type": "Attribute",
        }
    )
    polarization: PowerTrackingTypePolarization = field(
        metadata={
            "type": "Attribute",
        }
    )


@dataclass(kw_only=True)
class ChannelDelayType:
    class Meta:
        target_namespace = "biomass_common"

    value: float = field()
    swath: SwathType = field(
        metadata={
            "name": "Swath",
            "type": "Attribute",
        }
    )
    polarization: PolarizationType = field(
        metadata={
            "name": "Polarization",
            "type": "Attribute",
        }
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


@dataclass(kw_only=True)
class ChannelImbalanceType:
    class Meta:
        target_namespace = "biomass_common"

    tx: FcomplexNumberType = field(
        metadata={
            "name": "TX",
            "type": "Element",
            "namespace": "",
        }
    )
    rx: FcomplexNumberType = field(
        metadata={
            "name": "RX",
            "type": "Element",
            "namespace": "",
        }
    )


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
class L1PreProcessorAnnotations:
    ispformat: L1PreProcessorAnnotations.Ispformat = field(
        metadata={
            "name": "ISPFormat",
            "type": "Element",
        }
    )
    isperrors: L1PreProcessorAnnotations.Isperrors = field(
        metadata={
            "name": "ISPErrors",
            "type": "Element",
        }
    )
    raw_data_statistics: list[L1PreProcessorAnnotations.RawDataStatistics] = field(
        default_factory=list,
        metadata={
            "name": "RawDataStatistics",
            "type": "Element",
            "max_occurs": 4,
        },
    )
    gain_param_code_h: int = field(
        metadata={
            "name": "GainParamCodeH",
            "type": "Element",
        }
    )
    gain_param_code_v: int = field(
        metadata={
            "name": "GainParamCodeV",
            "type": "Element",
        }
    )
    noise_data: L1PreProcessorAnnotations.NoiseData = field(
        metadata={
            "name": "NoiseData",
            "type": "Element",
        }
    )
    channel_delays: ChannelDelaysType = field(
        metadata={
            "name": "ChannelDelays",
            "type": "Element",
        }
    )
    channel_imbalance: ChannelImbalanceType = field(
        metadata={
            "name": "ChannelImbalance",
            "type": "Element",
        }
    )
    chirp_replica_parameters: list[L1PreProcessorAnnotations.ChirpReplicaParameters] = field(
        default_factory=list,
        metadata={
            "name": "ChirpReplicaParameters",
            "type": "Element",
            "max_occurs": 4,
        },
    )
    internal_calibration_data: list[L1PreProcessorAnnotations.InternalCalibrationData] = field(
        default_factory=list,
        metadata={
            "name": "InternalCalibrationData",
            "type": "Element",
            "min_occurs": 1,
        },
    )
    noise_preamble_h: NoiseSequenceType = field(
        metadata={
            "name": "NoisePreambleH",
            "type": "Element",
        }
    )
    noise_postamble_h: NoiseSequenceType = field(
        metadata={
            "name": "NoisePostambleH",
            "type": "Element",
        }
    )
    noise_preamble_v: NoiseSequenceType = field(
        metadata={
            "name": "NoisePreambleV",
            "type": "Element",
        }
    )
    noise_postamble_v: NoiseSequenceType = field(
        metadata={
            "name": "NoisePostambleV",
            "type": "Element",
        }
    )
    in_out_band_power_ratio_data: list[L1PreProcessorAnnotations.InOutBandPowerRatioData] = field(
        default_factory=list,
        metadata={
            "name": "InOutBandPowerRatioData",
            "type": "Element",
        },
    )

    @dataclass(kw_only=True)
    class Ispformat:
        echo: CompressionLevelType = field(
            metadata={
                "name": "Echo",
                "type": "Element",
            }
        )
        calibration: CompressionLevelType = field(
            metadata={
                "name": "Calibration",
                "type": "Element",
            }
        )
        noise: CompressionLevelType = field(
            metadata={
                "name": "Noise",
                "type": "Element",
            }
        )
        mean_bit_rate: float = field(
            metadata={
                "name": "MeanBitRate",
                "type": "Element",
            }
        )

    @dataclass(kw_only=True)
    class Isperrors:
        num_missing_packets: int = field(
            metadata={
                "name": "NumMissingPackets",
                "type": "Element",
            }
        )
        num_corrupted_packets: int = field(
            metadata={
                "name": "NumCorruptedPackets",
                "type": "Element",
            }
        )

    @dataclass(kw_only=True)
    class RawDataStatistics:
        mean_i: float = field(
            metadata={
                "name": "Mean_i",
                "type": "Element",
            }
        )
        mean_q: float = field(
            metadata={
                "name": "Mean_q",
                "type": "Element",
            }
        )
        std_dev_i: float = field(
            metadata={
                "name": "StdDev_i",
                "type": "Element",
            }
        )
        std_dev_q: float = field(
            metadata={
                "name": "StdDev_q",
                "type": "Element",
            }
        )
        quadrature_departure: float = field(
            metadata={
                "name": "QuadratureDeparture",
                "type": "Element",
            }
        )
        polarization: PolarizationType = field(
            metadata={
                "name": "Polarization",
                "type": "Element",
            }
        )

    @dataclass(kw_only=True)
    class NoiseData:
        preamble_present: bool = field(
            metadata={
                "name": "PreamblePresent",
                "type": "Element",
            }
        )
        postamble_present: bool = field(
            metadata={
                "name": "PostamblePresent",
                "type": "Element",
            }
        )

    @dataclass(kw_only=True)
    class ChirpReplicaParameters:
        bandwidth: float = field(
            metadata={
                "name": "Bandwidth",
                "type": "Element",
            }
        )
        pslr: float = field(
            metadata={
                "name": "PSLR",
                "type": "Element",
            }
        )
        islr: float = field(
            metadata={
                "name": "ISLR",
                "type": "Element",
            }
        )
        location_error: float = field(
            metadata={
                "name": "LocationError",
                "type": "Element",
            }
        )
        validity_flag: bool = field(
            metadata={
                "name": "ValidityFlag",
                "type": "Element",
            }
        )
        polarization: PolarizationType = field(
            metadata={
                "name": "Polarization",
                "type": "Element",
            }
        )

    @dataclass(kw_only=True)
    class InternalCalibrationData:
        reference_time: str = field(
            metadata={
                "name": "ReferenceTime",
                "type": "Element",
            }
        )
        excitation_coefficients: L1PreProcessorAnnotations.InternalCalibrationData.ExcitationCoefficients = field(
            metadata={
                "name": "ExcitationCoefficients",
                "type": "Element",
            }
        )
        drifts: L1PreProcessorAnnotations.InternalCalibrationData.Drifts = field(
            metadata={
                "name": "Drifts",
                "type": "Element",
            }
        )

        @dataclass(kw_only=True)
        class ExcitationCoefficients:
            power_tracking: list[PowerTrackingType] = field(
                default_factory=list,
                metadata={
                    "name": "PowerTracking",
                    "type": "Element",
                    "max_occurs": 8,
                },
            )

        @dataclass(kw_only=True)
        class Drifts:
            drift: list[DriftType] = field(
                default_factory=list,
                metadata={
                    "name": "Drift",
                    "type": "Element",
                    "max_occurs": 4,
                },
            )

    @dataclass(kw_only=True)
    class InOutBandPowerRatioData:
        reference_time: str = field(
            metadata={
                "name": "ReferenceTime",
                "type": "Element",
            }
        )
        power_ratios: L1PreProcessorAnnotations.InOutBandPowerRatioData.PowerRatios = field(
            metadata={
                "name": "PowerRatios",
                "type": "Element",
            }
        )

        @dataclass(kw_only=True)
        class PowerRatios:
            power_ratio: list[PowerRatioType] = field(
                default_factory=list,
                metadata={
                    "name": "PowerRatio",
                    "type": "Element",
                    "max_occurs": 4,
                },
            )
