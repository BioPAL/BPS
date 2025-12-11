# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD BPS Input file models
-------------------------
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DemXyzproductChannelsType(Enum):
    FULL = "FULL"
    XYZ_ONLY = "XYZ_ONLY"


@dataclass
class InputBiomassAntennaPattern2Dtype:
    class Meta:
        name = "InputBiomassAntennaPattern2DType"

    input_antenna_pattern_d1_hproduct: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputAntennaPatternD1HProduct",
            "type": "Element",
            "required": True,
        },
    )
    input_antenna_pattern_d2_hproduct: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputAntennaPatternD2HProduct",
            "type": "Element",
            "required": True,
        },
    )
    input_antenna_pattern_d1_vproduct: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputAntennaPatternD1VProduct",
            "type": "Element",
            "required": True,
        },
    )
    input_antenna_pattern_d2_vproduct: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputAntennaPatternD2VProduct",
            "type": "Element",
            "required": True,
        },
    )
    input_txpower_tracking_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputTXPowerTrackingProduct",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class OneWayAntennaPatternType:
    tx: Optional[str] = field(
        default=None,
        metadata={
            "name": "TX",
            "type": "Element",
            "required": True,
        },
    )
    rx: Optional[str] = field(
        default=None,
        metadata={
            "name": "RX",
            "type": "Element",
            "required": True,
        },
    )


class OutputSspheadersFileTypeFormat(Enum):
    CSV = "CSV"


@dataclass
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

    absolute_start_time: Optional[str] = field(
        default=None,
        metadata={
            "name": "AbsoluteStartTime",
            "type": "Element",
            "required": True,
        },
    )
    duration: Optional[float] = field(
        default=None,
        metadata={
            "name": "Duration",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
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

    start_index: Optional[int] = field(
        default=None,
        metadata={
            "name": "StartIndex",
            "type": "Element",
            "required": True,
        },
    )
    length: Optional[int] = field(
        default=None,
        metadata={
            "name": "Length",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
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

    absolute_start_time: Optional[float] = field(
        default=None,
        metadata={
            "name": "AbsoluteStartTime",
            "type": "Element",
            "required": True,
        },
    )
    duration: Optional[float] = field(
        default=None,
        metadata={
            "name": "Duration",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class PfselectorSwathNameType:
    class Meta:
        name = "PFSelectorSwathNameType"

    name: Optional[str] = field(
        default=None,
        metadata={
            "name": "Name",
            "type": "Attribute",
            "required": True,
        },
    )


class PolarizationBaseType(Enum):
    HH = "HH"
    HV = "HV"
    VH = "VH"
    VV = "VV"


@dataclass
class SarfoctwoWayPatternsType:
    class Meta:
        name = "SARFOCTwoWayPatternsType"

    azimuth_antenna_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "AzimuthAntennaProduct",
            "type": "Element",
            "required": True,
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
            "max_occurs": 3,
        },
    )


@dataclass
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
            "required": True,
        },
    )
    latitude: float = field(
        default=0.0,
        metadata={
            "name": "Latitude",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class TimeOfInterestType:
    start: Optional["TimeOfInterestType.Start"] = field(
        default=None,
        metadata={
            "name": "Start",
            "type": "Element",
            "required": True,
        },
    )
    stop: Optional["TimeOfInterestType.Stop"] = field(
        default=None,
        metadata={
            "name": "Stop",
            "type": "Element",
            "required": True,
        },
    )

    @dataclass
    class Start:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: str = field(
            init=False,
            default="Utc",
            metadata={
                "name": "Unit",
                "type": "Attribute",
                "required": True,
            },
        )

    @dataclass
    class Stop:
        value: str = field(
            default="",
            metadata={
                "required": True,
            },
        )
        unit: str = field(
            init=False,
            default="Utc",
            metadata={
                "name": "Unit",
                "type": "Attribute",
                "required": True,
            },
        )


@dataclass
class GenericCoregistratorInputType:
    master_level1_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "MasterLevel1Product",
            "type": "Element",
            "required": True,
        },
    )
    slave_level1_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "SlaveLevel1Product",
            "type": "Element",
            "required": True,
        },
    )
    dem_xyzproduct: Optional[str] = field(
        default=None,
        metadata={
            "name": "DemXYZProduct",
            "type": "Element",
        },
    )
    config_file_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "ConfigFileName",
            "type": "Element",
        },
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={
            "name": "OutputPath",
            "type": "Element",
        },
    )
    dem_xyzproduct_channels: Optional[DemXyzproductChannelsType] = field(
        default=None,
        metadata={
            "name": "DemXYZProductChannels",
            "type": "Element",
        },
    )
    external_shifts_az: Optional[str] = field(
        default=None,
        metadata={
            "name": "ExternalShiftsAz",
            "type": "Element",
        },
    )
    external_shifts_rg: Optional[str] = field(
        default=None,
        metadata={
            "name": "ExternalShiftsRg",
            "type": "Element",
        },
    )


@dataclass
class OutputSspheadersFileType:
    class Meta:
        name = "OutputSSPHeadersFileType"

    value: str = field(
        default="",
        metadata={
            "required": True,
        },
    )
    format: Optional[OutputSspheadersFileTypeFormat] = field(
        default=None,
        metadata={
            "name": "Format",
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
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


@dataclass
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

    line_interval: Optional[PfselectorIndexIntervalType] = field(
        default=None,
        metadata={
            "name": "LineInterval",
            "type": "Element",
            "required": True,
        },
    )
    sample_interval: list[PfselectorIndexIntervalType] = field(
        default_factory=list,
        metadata={
            "name": "SampleInterval",
            "type": "Element",
            "min_occurs": 1,
            "max_occurs": 2,
            "sequence": 1,
        },
    )


@dataclass
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


@dataclass
class PfselectorSwathsBurstsSwathType(PfselectorSwathNameType):
    """
    Parameters
    ----------
    burst_interval
        Interval of bursts to select for the specified swath (full swath if not present)
    """

    class Meta:
        name = "PFSelectorSwathsBurstsSwathType"

    burst_interval: Optional[PfselectorIndexIntervalType] = field(
        default=None,
        metadata={
            "name": "BurstInterval",
            "type": "Element",
        },
    )


@dataclass
class SarfoconeWayPatternsType:
    class Meta:
        name = "SARFOCOneWayPatternsType"

    azimuth_antenna_product: Optional[OneWayAntennaPatternType] = field(
        default=None,
        metadata={
            "name": "AzimuthAntennaProduct",
            "type": "Element",
            "required": True,
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
            "max_occurs": 3,
        },
    )


@dataclass
class BpsstackProcessorInputType:
    class Meta:
        name = "BPSStackProcessorInputType"

    coregistration: Optional[GenericCoregistratorInputType] = field(
        default=None,
        metadata={
            "name": "Coregistration",
            "type": "Element",
            "required": True,
        },
    )
    bpsconfiguration_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "BPSConfigurationFile",
            "type": "Element",
            "required": True,
        },
    )
    bpslog_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "BPSLogFile",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class BiomassL0ImportPreProcType:
    input_l0_sproduct: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputL0SProduct",
            "type": "Element",
            "required": True,
        },
    )
    input_l0_mproduct: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputL0MProduct",
            "type": "Element",
        },
    )
    input_aux_orb_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputAuxOrbFile",
            "type": "Element",
            "required": True,
        },
    )
    input_aux_att_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputAuxAttFile",
            "type": "Element",
            "required": True,
        },
    )
    input_aux_ins_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputAuxInsFile",
            "type": "Element",
            "required": True,
        },
    )
    input_iersbullettin_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputIERSBullettinFile",
            "type": "Element",
        },
    )
    time_of_interest: Optional[TimeOfInterestType] = field(
        default=None,
        metadata={
            "name": "TimeOfInterest",
            "type": "Element",
        },
    )
    configuration_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "ConfigurationFile",
            "type": "Element",
            "required": True,
        },
    )
    bpsconfiguration_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "BPSConfigurationFile",
            "type": "Element",
            "required": True,
        },
    )
    bpslog_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "BPSLogFile",
            "type": "Element",
            "required": True,
        },
    )
    intermediate_dyn_cal_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "IntermediateDynCalProduct",
            "type": "Element",
        },
    )
    intermediate_pgpproduct: Optional[str] = field(
        default=None,
        metadata={
            "name": "IntermediatePGPProduct",
            "type": "Element",
        },
    )
    intermediate_channel_delays_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "IntermediateChannelDelaysFile",
            "type": "Element",
        },
    )
    output_channel_delays_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "OutputChannelDelaysFile",
            "type": "Element",
        },
    )
    output_raw_data_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "OutputRawDataProduct",
            "type": "Element",
            "required": True,
        },
    )
    output_tx_power_tracking_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "OutputTxPowerTrackingProduct",
            "type": "Element",
        },
    )
    output_chirp_replica_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "OutputChirpReplicaProduct",
            "type": "Element",
        },
    )
    output_per_line_correction_factors_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "OutputPerLineCorrectionFactorsProduct",
            "type": "Element",
        },
    )
    output_est_noise_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "OutputEstNoiseProduct",
            "type": "Element",
        },
    )
    output_channel_imbalance_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "OutputChannelImbalanceFile",
            "type": "Element",
        },
    )
    output_sspheaders_file: Optional[OutputSspheadersFileType] = field(
        default=None,
        metadata={
            "name": "OutputSSPHeadersFile",
            "type": "Element",
        },
    )
    output_report_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "OutputReportFile",
            "type": "Element",
        },
    )


@dataclass
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

    swaths: Optional[PfselectorSwathNamesType] = field(
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


@dataclass
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


@dataclass
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


@dataclass
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

    swaths: Optional[PfselectorSwathNamesType] = field(
        default=None,
        metadata={
            "name": "Swaths",
            "type": "Element",
        },
    )
    azimuth_time_interval: Optional[PfselectorAzimuthTimeIntervalType] = field(
        default=None,
        metadata={
            "name": "AzimuthTimeInterval",
            "type": "Element",
            "required": True,
        },
    )
    range_time_interval: list[PfselectorRangeTimeIntervalType] = field(
        default_factory=list,
        metadata={
            "name": "RangeTimeInterval",
            "type": "Element",
            "min_occurs": 1,
            "max_occurs": 2,
            "sequence": 1,
        },
    )


@dataclass
class SarfocinputPatternsType:
    class Meta:
        name = "SARFOCInputPatternsType"

    two_way: Optional[SarfoctwoWayPatternsType] = field(
        default=None,
        metadata={
            "name": "TwoWay",
            "type": "Element",
        },
    )
    one_way: Optional[SarfoconeWayPatternsType] = field(
        default=None,
        metadata={
            "name": "OneWay",
            "type": "Element",
        },
    )


@dataclass
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

    swaths_bursts: Optional[PfselectorSwathsBurstsType] = field(
        default=None,
        metadata={
            "name": "SwathsBursts",
            "type": "Element",
        },
    )
    raster_coordinates: Optional[PfselectorRasterCoordinatesType] = field(
        default=None,
        metadata={
            "name": "RasterCoordinates",
            "type": "Element",
        },
    )
    geographic_coordinates: Optional[PfselectorGeographicCoordinatesType] = field(
        default=None,
        metadata={
            "name": "GeographicCoordinates",
            "type": "Element",
        },
    )
    time_coordinates: Optional[PfselectorTimeCoordinatesType] = field(
        default=None,
        metadata={
            "name": "TimeCoordinates",
            "type": "Element",
        },
    )


@dataclass
class Sarfocinput:
    class Meta:
        name = "SARFOCInput"

    input_level0_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputLevel0Product",
            "type": "Element",
            "required": True,
        },
    )
    input_antenna_patterns: Optional[SarfocinputPatternsType] = field(
        default=None,
        metadata={
            "name": "InputAntennaPatterns",
            "type": "Element",
        },
    )
    input_chirp_replica_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputChirpReplicaProduct",
            "type": "Element",
        },
    )
    input_per_line_dechirping_reference_times_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputPerLineDechirpingReferenceTimesProduct",
            "type": "Element",
        },
    )
    input_per_line_correction_factors_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputPerLineCorrectionFactorsProduct",
            "type": "Element",
        },
    )
    input_noise_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputNoiseProduct",
            "type": "Element",
        },
    )
    input_processing_dcpoly_file_name: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputProcessingDCPolyFileName",
            "type": "Element",
        },
    )
    processing_options_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "ProcessingOptionsFile",
            "type": "Element",
            "required": True,
        },
    )
    processing_parameters_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "ProcessingParametersFile",
            "type": "Element",
            "required": True,
        },
    )
    polarization_to_process: Optional[PfselectorPolarizationsType] = field(
        default=None,
        metadata={
            "name": "PolarizationToProcess",
            "type": "Element",
        },
    )
    area_to_process: Optional[PfselectorAreaType] = field(
        default=None,
        metadata={
            "name": "AreaToProcess",
            "type": "Element",
        },
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={
            "name": "OutputPath",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
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

    core_processor: Optional[Sarfocinput] = field(
        default=None,
        metadata={
            "name": "CoreProcessor",
            "type": "Element",
            "required": True,
        },
    )
    input_biomass_antenna_pattern2_d: Optional[InputBiomassAntennaPattern2Dtype] = field(
        default=None,
        metadata={
            "name": "InputBiomassAntennaPattern2D",
            "type": "Element",
        },
    )
    input_geomagnetic_field_model_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputGeomagneticFieldModelProduct",
            "type": "Element",
        },
    )
    input_tec_map_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputTecMapProduct",
            "type": "Element",
        },
    )
    input_climatological_model_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputClimatologicalModelFile",
            "type": "Element",
        },
    )
    input_faraday_rotation_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputFaradayRotationProduct",
            "type": "Element",
        },
    )
    input_phase_screen_product: Optional[str] = field(
        default=None,
        metadata={
            "name": "InputPhaseScreenProduct",
            "type": "Element",
        },
    )
    bpsconfiguration_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "BPSConfigurationFile",
            "type": "Element",
            "required": True,
        },
    )
    bpslog_file: Optional[str] = field(
        default=None,
        metadata={
            "name": "BPSLogFile",
            "type": "Element",
            "required": True,
        },
    )


@dataclass
class AresysXmlInputType:
    step: list["AresysXmlInputType.Step"] = field(
        default_factory=list,
        metadata={
            "name": "Step",
            "type": "Element",
        },
    )

    @dataclass
    class Step:
        sarfoc: Optional[Sarfocinput] = field(
            default=None,
            metadata={
                "name": "SARFOC",
                "type": "Element",
            },
        )
        bpsl1_core_processor: Optional[Bpsl1CoreProcessorInputType] = field(
            default=None,
            metadata={
                "name": "BPSL1CoreProcessor",
                "type": "Element",
            },
        )
        bpsstack_processor: Optional[BpsstackProcessorInputType] = field(
            default=None,
            metadata={
                "name": "BPSStackProcessor",
                "type": "Element",
            },
        )
        generic_coregistrator_input: Optional[GenericCoregistratorInputType] = field(
            default=None,
            metadata={
                "name": "GenericCoregistratorInput",
                "type": "Element",
            },
        )
        biomass_l0_import_pre_proc: Optional[BiomassL0ImportPreProcType] = field(
            default=None,
            metadata={
                "name": "BiomassL0ImportPreProc",
                "type": "Element",
            },
        )
        number: Optional[int] = field(
            default=None,
            metadata={
                "name": "Number",
                "type": "Attribute",
                "required": True,
            },
        )
        total: Optional[int] = field(
            default=None,
            metadata={
                "name": "Total",
                "type": "Attribute",
                "required": True,
            },
        )


@dataclass
class AresysXmlInput(AresysXmlInputType):
    pass
