# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD BPS Input file models
-------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DemXyzproductChannelsType(Enum):
    FULL = "FULL"
    XYZ_ONLY = "XYZ_ONLY"


@dataclass(kw_only=True)
class InputBiomassAntennaPattern2Dtype:
    class Meta:
        name = "InputBiomassAntennaPattern2DType"

    input_antenna_pattern_d1_hproduct: str = field(
        metadata={
            "name": "InputAntennaPatternD1HProduct",
            "type": "Element",
        }
    )
    input_antenna_pattern_d2_hproduct: str = field(
        metadata={
            "name": "InputAntennaPatternD2HProduct",
            "type": "Element",
        }
    )
    input_antenna_pattern_d1_vproduct: str = field(
        metadata={
            "name": "InputAntennaPatternD1VProduct",
            "type": "Element",
        }
    )
    input_antenna_pattern_d2_vproduct: str = field(
        metadata={
            "name": "InputAntennaPatternD2VProduct",
            "type": "Element",
        }
    )
    input_txpower_tracking_product: str = field(
        metadata={
            "name": "InputTXPowerTrackingProduct",
            "type": "Element",
        }
    )


@dataclass(kw_only=True)
class OneWayAntennaPatternType:
    tx: str = field(
        metadata={
            "name": "TX",
            "type": "Element",
        }
    )
    rx: str = field(
        metadata={
            "name": "RX",
            "type": "Element",
        }
    )


class OutputSspheadersFileTypeFormat(Enum):
    CSV = "CSV"


@dataclass(kw_only=True)
class PfselectorAzimuthTimeIntervalType:
    """
    Parameters
    ----------
    absolute_start_time
        Absolute start time of the azimuth time interval
    duration
        Duration of the azimuth time interval
    """

    class Meta:
        name = "PFSelectorAzimuthTimeIntervalType"

    absolute_start_time: str = field(
        metadata={
            "name": "AbsoluteStartTime",
            "type": "Element",
        }
    )
    duration: float = field(
        metadata={
            "name": "Duration",
            "type": "Element",
        }
    )


@dataclass(kw_only=True)
class PfselectorIndexIntervalType:
    """
    Parameters
    ----------
    start_index
        Start index of the interval
    length
        Length of the interval
    """

    class Meta:
        name = "PFSelectorIndexIntervalType"

    start_index: int = field(
        metadata={
            "name": "StartIndex",
            "type": "Element",
        }
    )
    length: int = field(
        metadata={
            "name": "Length",
            "type": "Element",
        }
    )


@dataclass(kw_only=True)
class PfselectorRangeTimeIntervalType:
    """
    Parameters
    ----------
    absolute_start_time
        Absolute start time of the range time interval
    duration
        Duration of the range time interval
    """

    class Meta:
        name = "PFSelectorRangeTimeIntervalType"

    absolute_start_time: float = field(
        metadata={
            "name": "AbsoluteStartTime",
            "type": "Element",
        }
    )
    duration: float = field(
        metadata={
            "name": "Duration",
            "type": "Element",
        }
    )


@dataclass(kw_only=True)
class PfselectorSwathNameType:
    class Meta:
        name = "PFSelectorSwathNameType"

    name: str = field(
        metadata={
            "name": "Name",
            "type": "Attribute",
        }
    )


class PolarizationBaseType(Enum):
    HH = "HH"
    HV = "HV"
    VH = "VH"
    VV = "VV"


@dataclass(kw_only=True)
class SarfoctwoWayPatternsType:
    class Meta:
        name = "SARFOCTwoWayPatternsType"

    azimuth_antenna_product: None | str = field(
        default=None,
        metadata={
            "name": "AzimuthAntennaProduct",
            "type": "Element",
        },
    )
    azimuth_elementary_antenna_product: list[str] = field(
        default_factory=list,
        metadata={
            "name": "AzimuthElementaryAntennaProduct",
            "type": "Element",
            "max_occurs": 2,
        },
    )
    elevation_antenna_product: list[str] = field(
        default_factory=list,
        metadata={
            "name": "ElevationAntennaProduct",
            "type": "Element",
            "max_occurs": 2,
        },
    )


@dataclass(kw_only=True)
class TiePointType:
    """
    Parameters
    ----------
    longitude
        Longitude [degree] of the tie point.
    latitude
        Latitude [degree] of the tie point.
    """

    longitude: float = field(
        default=0.0,
        metadata={
            "name": "Longitude",
            "type": "Element",
        },
    )
    latitude: float = field(
        default=0.0,
        metadata={
            "name": "Latitude",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class TimeOfInterestType:
    start: TimeOfInterestType.Start = field(
        metadata={
            "name": "Start",
            "type": "Element",
        }
    )
    stop: TimeOfInterestType.Stop = field(
        metadata={
            "name": "Stop",
            "type": "Element",
        }
    )

    @dataclass(kw_only=True)
    class Start:
        value: str = field(default="")
        unit: str = field(
            init=False,
            default="Utc",
            metadata={
                "name": "Unit",
                "type": "Attribute",
                "required": True,
            },
        )

    @dataclass(kw_only=True)
    class Stop:
        value: str = field(default="")
        unit: str = field(
            init=False,
            default="Utc",
            metadata={
                "name": "Unit",
                "type": "Attribute",
                "required": True,
            },
        )


@dataclass(kw_only=True)
class GenericCoregistratorInputType:
    master_level1_product: str = field(
        metadata={
            "name": "MasterLevel1Product",
            "type": "Element",
        }
    )
    slave_level1_product: str = field(
        metadata={
            "name": "SlaveLevel1Product",
            "type": "Element",
        }
    )
    dem_xyzproduct: None | str = field(
        default=None,
        metadata={
            "name": "DemXYZProduct",
            "type": "Element",
        },
    )
    config_file_name: None | str = field(
        default=None,
        metadata={
            "name": "ConfigFileName",
            "type": "Element",
        },
    )
    output_path: None | str = field(
        default=None,
        metadata={
            "name": "OutputPath",
            "type": "Element",
        },
    )
    dem_xyzproduct_channels: None | DemXyzproductChannelsType = field(
        default=None,
        metadata={
            "name": "DemXYZProductChannels",
            "type": "Element",
        },
    )
    external_shifts_az: None | str = field(
        default=None,
        metadata={
            "name": "ExternalShiftsAz",
            "type": "Element",
        },
    )
    external_shifts_rg: None | str = field(
        default=None,
        metadata={
            "name": "ExternalShiftsRg",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class OutputSspheadersFileType:
    class Meta:
        name = "OutputSSPHeadersFileType"

    value: str = field(default="")
    format: OutputSspheadersFileTypeFormat = field(
        metadata={
            "name": "Format",
            "type": "Attribute",
        }
    )


@dataclass(kw_only=True)
class PfselectorPolarizationsType:
    class Meta:
        name = "PFSelectorPolarizationsType"

    polarization: list[PolarizationBaseType] = field(
        default_factory=list,
        metadata={
            "name": "Polarization",
            "type": "Element",
            "min_occurs": 1,
            "max_occurs": 4,
        },
    )


@dataclass(kw_only=True)
class PfselectorRasterCoordinatesSwathType(PfselectorSwathNameType):
    """
    Parameters
    ----------
    line_interval
        Interval of lines to select for the specified swath (all the lines if not present)
    sample_interval
        Interval of samples to select for the specified swath (all the samples if not present)
    """

    class Meta:
        name = "PFSelectorRasterCoordinatesSwathType"

    line_interval: None | PfselectorIndexIntervalType = field(
        default=None,
        metadata={
            "name": "LineInterval",
            "type": "Element",
        },
    )
    sample_interval: None | PfselectorIndexIntervalType = field(
        default=None,
        metadata={
            "name": "SampleInterval",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class PfselectorSwathNamesType:
    class Meta:
        name = "PFSelectorSwathNamesType"

    swath: list[PfselectorSwathNameType] = field(
        default_factory=list,
        metadata={
            "name": "Swath",
            "type": "Element",
            "min_occurs": 1,
        },
    )


@dataclass(kw_only=True)
class PfselectorSwathsBurstsSwathType(PfselectorSwathNameType):
    """
    Parameters
    ----------
    burst_interval
        Interval of bursts to select for the specified swath (full swath if not present)
    """

    class Meta:
        name = "PFSelectorSwathsBurstsSwathType"

    burst_interval: None | PfselectorIndexIntervalType = field(
        default=None,
        metadata={
            "name": "BurstInterval",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class SarfoconeWayPatternsType:
    class Meta:
        name = "SARFOCOneWayPatternsType"

    azimuth_antenna_product: None | OneWayAntennaPatternType = field(
        default=None,
        metadata={
            "name": "AzimuthAntennaProduct",
            "type": "Element",
        },
    )
    azimuth_elementary_antenna_product: list[OneWayAntennaPatternType] = field(
        default_factory=list,
        metadata={
            "name": "AzimuthElementaryAntennaProduct",
            "type": "Element",
            "max_occurs": 2,
        },
    )
    elevation_antenna_product: list[OneWayAntennaPatternType] = field(
        default_factory=list,
        metadata={
            "name": "ElevationAntennaProduct",
            "type": "Element",
            "max_occurs": 2,
        },
    )


@dataclass(kw_only=True)
class BpsstackProcessorInputType:
    class Meta:
        name = "BPSStackProcessorInputType"

    coregistration: GenericCoregistratorInputType = field(
        metadata={
            "name": "Coregistration",
            "type": "Element",
        }
    )
    bpsconfiguration_file: str = field(
        metadata={
            "name": "BPSConfigurationFile",
            "type": "Element",
        }
    )
    bpslog_file: str = field(
        metadata={
            "name": "BPSLogFile",
            "type": "Element",
        }
    )


@dataclass(kw_only=True)
class BiomassL0ImportPreProcType:
    input_l0_sproduct: str = field(
        metadata={
            "name": "InputL0SProduct",
            "type": "Element",
        }
    )
    input_l0_mproduct: None | str = field(
        default=None,
        metadata={
            "name": "InputL0MProduct",
            "type": "Element",
        },
    )
    input_aux_orb_file: str = field(
        metadata={
            "name": "InputAuxOrbFile",
            "type": "Element",
        }
    )
    input_aux_att_file: str = field(
        metadata={
            "name": "InputAuxAttFile",
            "type": "Element",
        }
    )
    input_aux_ins_file: str = field(
        metadata={
            "name": "InputAuxInsFile",
            "type": "Element",
        }
    )
    input_iersbullettin_file: None | str = field(
        default=None,
        metadata={
            "name": "InputIERSBullettinFile",
            "type": "Element",
        },
    )
    time_of_interest: None | TimeOfInterestType = field(
        default=None,
        metadata={
            "name": "TimeOfInterest",
            "type": "Element",
        },
    )
    configuration_file: str = field(
        metadata={
            "name": "ConfigurationFile",
            "type": "Element",
        }
    )
    bpsconfiguration_file: str = field(
        metadata={
            "name": "BPSConfigurationFile",
            "type": "Element",
        }
    )
    bpslog_file: str = field(
        metadata={
            "name": "BPSLogFile",
            "type": "Element",
        }
    )
    intermediate_dyn_cal_product: None | str = field(
        default=None,
        metadata={
            "name": "IntermediateDynCalProduct",
            "type": "Element",
        },
    )
    intermediate_pgpproduct: None | str = field(
        default=None,
        metadata={
            "name": "IntermediatePGPProduct",
            "type": "Element",
        },
    )
    intermediate_channel_delays_file: None | str = field(
        default=None,
        metadata={
            "name": "IntermediateChannelDelaysFile",
            "type": "Element",
        },
    )
    output_channel_delays_file: None | str = field(
        default=None,
        metadata={
            "name": "OutputChannelDelaysFile",
            "type": "Element",
        },
    )
    output_raw_data_product: str = field(
        metadata={
            "name": "OutputRawDataProduct",
            "type": "Element",
        }
    )
    output_tx_power_tracking_product: None | str = field(
        default=None,
        metadata={
            "name": "OutputTxPowerTrackingProduct",
            "type": "Element",
        },
    )
    output_chirp_replica_product: None | str = field(
        default=None,
        metadata={
            "name": "OutputChirpReplicaProduct",
            "type": "Element",
        },
    )
    output_per_line_correction_factors_product: None | str = field(
        default=None,
        metadata={
            "name": "OutputPerLineCorrectionFactorsProduct",
            "type": "Element",
        },
    )
    output_est_noise_product: None | str = field(
        default=None,
        metadata={
            "name": "OutputEstNoiseProduct",
            "type": "Element",
        },
    )
    output_channel_imbalance_file: None | str = field(
        default=None,
        metadata={
            "name": "OutputChannelImbalanceFile",
            "type": "Element",
        },
    )
    output_sspheaders_file: None | OutputSspheadersFileType = field(
        default=None,
        metadata={
            "name": "OutputSSPHeadersFile",
            "type": "Element",
        },
    )
    output_report_file: None | str = field(
        default=None,
        metadata={
            "name": "OutputReportFile",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class PfselectorGeographicCoordinatesType:
    """
    Parameters
    ----------
    swaths
        Swaths to select (all the swaths if not present)
    tie_point
    """

    class Meta:
        name = "PFSelectorGeographicCoordinatesType"

    swaths: None | PfselectorSwathNamesType = field(
        default=None,
        metadata={
            "name": "Swaths",
            "type": "Element",
        },
    )
    tie_point: list[TiePointType] = field(
        default_factory=list,
        metadata={
            "name": "TiePoint",
            "type": "Element",
            "min_occurs": 2,
        },
    )


@dataclass(kw_only=True)
class PfselectorRasterCoordinatesType:
    class Meta:
        name = "PFSelectorRasterCoordinatesType"

    swath: list[PfselectorRasterCoordinatesSwathType] = field(
        default_factory=list,
        metadata={
            "name": "Swath",
            "type": "Element",
            "min_occurs": 1,
        },
    )


@dataclass(kw_only=True)
class PfselectorSwathsBurstsType:
    class Meta:
        name = "PFSelectorSwathsBurstsType"

    swath: list[PfselectorSwathsBurstsSwathType] = field(
        default_factory=list,
        metadata={
            "name": "Swath",
            "type": "Element",
            "min_occurs": 1,
        },
    )


@dataclass(kw_only=True)
class PfselectorTimeCoordinatesType:
    """
    Parameters
    ----------
    swaths
        Swaths to select (all the swaths if not present)
    azimuth_time_interval
        Azimuth time interval to select (full azimuth time coverage if not present)
    range_time_interval
        Range time interval to select (full range time coverage if not present)
    """

    class Meta:
        name = "PFSelectorTimeCoordinatesType"

    swaths: None | PfselectorSwathNamesType = field(
        default=None,
        metadata={
            "name": "Swaths",
            "type": "Element",
        },
    )
    azimuth_time_interval: None | PfselectorAzimuthTimeIntervalType = field(
        default=None,
        metadata={
            "name": "AzimuthTimeInterval",
            "type": "Element",
        },
    )
    range_time_interval: None | PfselectorRangeTimeIntervalType = field(
        default=None,
        metadata={
            "name": "RangeTimeInterval",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class SarfocinputPatternsType:
    class Meta:
        name = "SARFOCInputPatternsType"

    two_way: None | SarfoctwoWayPatternsType = field(
        default=None,
        metadata={
            "name": "TwoWay",
            "type": "Element",
        },
    )
    one_way: None | SarfoconeWayPatternsType = field(
        default=None,
        metadata={
            "name": "OneWay",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class PfselectorAreaType:
    """
    Parameters
    ----------
    swaths_bursts
        Area selection by swaths/bursts
    raster_coordinates
        Area selection by raster coordinates
    geographic_coordinates
        Area selection by geographic coordinates
    time_coordinates
        Area selection by time coordinates
    """

    class Meta:
        name = "PFSelectorAreaType"

    swaths_bursts: None | PfselectorSwathsBurstsType = field(
        default=None,
        metadata={
            "name": "SwathsBursts",
            "type": "Element",
        },
    )
    raster_coordinates: None | PfselectorRasterCoordinatesType = field(
        default=None,
        metadata={
            "name": "RasterCoordinates",
            "type": "Element",
        },
    )
    geographic_coordinates: None | PfselectorGeographicCoordinatesType = field(
        default=None,
        metadata={
            "name": "GeographicCoordinates",
            "type": "Element",
        },
    )
    time_coordinates: None | PfselectorTimeCoordinatesType = field(
        default=None,
        metadata={
            "name": "TimeCoordinates",
            "type": "Element",
        },
    )


@dataclass(kw_only=True)
class Sarfocinput:
    class Meta:
        name = "SARFOCInput"

    input_level0_product: str = field(
        metadata={
            "name": "InputLevel0Product",
            "type": "Element",
        }
    )
    input_antenna_patterns: None | SarfocinputPatternsType = field(
        default=None,
        metadata={
            "name": "InputAntennaPatterns",
            "type": "Element",
        },
    )
    input_chirp_replica_product: None | str = field(
        default=None,
        metadata={
            "name": "InputChirpReplicaProduct",
            "type": "Element",
        },
    )
    input_per_line_dechirping_reference_times_product: None | str = field(
        default=None,
        metadata={
            "name": "InputPerLineDechirpingReferenceTimesProduct",
            "type": "Element",
        },
    )
    input_per_line_correction_factors_product: None | str = field(
        default=None,
        metadata={
            "name": "InputPerLineCorrectionFactorsProduct",
            "type": "Element",
        },
    )
    input_noise_product: None | str = field(
        default=None,
        metadata={
            "name": "InputNoiseProduct",
            "type": "Element",
        },
    )
    input_processing_dcpoly_file_name: None | str = field(
        default=None,
        metadata={
            "name": "InputProcessingDCPolyFileName",
            "type": "Element",
        },
    )
    processing_options_file: str = field(
        metadata={
            "name": "ProcessingOptionsFile",
            "type": "Element",
        }
    )
    processing_parameters_file: str = field(
        metadata={
            "name": "ProcessingParametersFile",
            "type": "Element",
        }
    )
    polarization_to_process: None | PfselectorPolarizationsType = field(
        default=None,
        metadata={
            "name": "PolarizationToProcess",
            "type": "Element",
        },
    )
    area_to_process: None | PfselectorAreaType = field(
        default=None,
        metadata={
            "name": "AreaToProcess",
            "type": "Element",
        },
    )
    output_path: str = field(
        metadata={
            "name": "OutputPath",
            "type": "Element",
        }
    )


@dataclass(kw_only=True)
class Bpsl1CoreProcessorInputType:
    """
    Parameters
    ----------
    core_processor
    input_biomass_antenna_pattern2_d
    input_geomagnetic_field_model_product
    input_tec_map_product
    input_climatological_model_file
    input_faraday_rotation_product
        Faraday rotation product folder
    input_phase_screen_product
        Phase screen product folder
    bpsconfiguration_file
    bpslog_file
    """

    class Meta:
        name = "BPSL1CoreProcessorInputType"

    core_processor: Sarfocinput = field(
        metadata={
            "name": "CoreProcessor",
            "type": "Element",
        }
    )
    input_biomass_antenna_pattern2_d: None | InputBiomassAntennaPattern2Dtype = field(
        default=None,
        metadata={
            "name": "InputBiomassAntennaPattern2D",
            "type": "Element",
        },
    )
    input_geomagnetic_field_model_product: None | str = field(
        default=None,
        metadata={
            "name": "InputGeomagneticFieldModelProduct",
            "type": "Element",
        },
    )
    input_tec_map_product: None | str = field(
        default=None,
        metadata={
            "name": "InputTecMapProduct",
            "type": "Element",
        },
    )
    input_climatological_model_file: None | str = field(
        default=None,
        metadata={
            "name": "InputClimatologicalModelFile",
            "type": "Element",
        },
    )
    input_faraday_rotation_product: None | str = field(
        default=None,
        metadata={
            "name": "InputFaradayRotationProduct",
            "type": "Element",
        },
    )
    input_phase_screen_product: None | str = field(
        default=None,
        metadata={
            "name": "InputPhaseScreenProduct",
            "type": "Element",
        },
    )
    bpsconfiguration_file: str = field(
        metadata={
            "name": "BPSConfigurationFile",
            "type": "Element",
        }
    )
    bpslog_file: str = field(
        metadata={
            "name": "BPSLogFile",
            "type": "Element",
        }
    )


@dataclass(kw_only=True)
class AresysXmlInputType:
    step: list[AresysXmlInputType.Step] = field(
        default_factory=list,
        metadata={
            "name": "Step",
            "type": "Element",
        },
    )

    @dataclass(kw_only=True)
    class Step:
        sarfoc: None | Sarfocinput = field(
            default=None,
            metadata={
                "name": "SARFOC",
                "type": "Element",
            },
        )
        bpsl1_core_processor: None | Bpsl1CoreProcessorInputType = field(
            default=None,
            metadata={
                "name": "BPSL1CoreProcessor",
                "type": "Element",
            },
        )
        bpsstack_processor: None | BpsstackProcessorInputType = field(
            default=None,
            metadata={
                "name": "BPSStackProcessor",
                "type": "Element",
            },
        )
        generic_coregistrator_input: None | GenericCoregistratorInputType = field(
            default=None,
            metadata={
                "name": "GenericCoregistratorInput",
                "type": "Element",
            },
        )
        biomass_l0_import_pre_proc: None | BiomassL0ImportPreProcType = field(
            default=None,
            metadata={
                "name": "BiomassL0ImportPreProc",
                "type": "Element",
            },
        )
        number: int = field(
            metadata={
                "name": "Number",
                "type": "Attribute",
            }
        )
        total: int = field(
            metadata={
                "name": "Total",
                "type": "Attribute",
            }
        )


@dataclass(kw_only=True)
class AresysXmlInput(AresysXmlInputType):
    pass
