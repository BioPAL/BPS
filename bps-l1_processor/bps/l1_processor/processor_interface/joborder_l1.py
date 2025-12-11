# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 processor job order
----------------------
"""

from dataclasses import dataclass
from pathlib import Path

from bps.common.joborder import DeviceResources, ProcessorConfiguration


@dataclass
class L1StripmapInputProducts:
    """Relevant Input products when processing stripmap products"""

    input_standard: Path
    """Sx_RAW__0S, Stripmap Level-0 - Standard product"""

    input_monitoring: Path | None
    """Sx_RAW__0M, Stripmap Level-0 - Monitoring product"""


@dataclass
class L1StripmapOutputProducts:
    """Relevant Output products when processing stripmap products"""

    output_directory: Path
    """Output products directory"""

    output_baseline: int
    """Output products baseline"""

    scs_standard_required: bool
    """Sx_SCS__1S, Stripmap L1 SCS - Standard product"""

    scs_monitoring_required: bool
    """Sx_SCS__1M, Stripmap L1 SCS - Monitoring product"""

    dgm_standard_required: bool = False
    """Sx_DGM__1S, Stripmap L1 DGM - Standard product"""


@dataclass
class L1StripmapProducts:
    """Relevant I/O products when processing stripmap products"""

    input: L1StripmapInputProducts
    """input products"""

    output: L1StripmapOutputProducts
    """output products"""


@dataclass
class L1RXOnlyInputProducts:
    """Relevant Input products when processing RO products"""

    input_standard: Path
    """RO_RAW__0S, RX Only Mode Level-0 - Standard product"""


@dataclass
class L1RXOnlyOutputProducts:
    """Relevant Output products when processing RO products"""

    output_directory: Path
    """Output products directory"""

    output_baseline: int
    """Output products baseline"""

    scs_standard_required: bool
    """RO_SCS__1S, RX Only Mode L1 SCS - Standard product"""


@dataclass
class L1RXOnlyProducts:
    """Relevant I/O products when processing RO products"""

    input: L1RXOnlyInputProducts
    """input products"""

    output: L1RXOnlyOutputProducts
    """output products"""


@dataclass
class L1AuxiliaryProducts:
    """Job order auxiliary files"""

    orbit: Path
    """AUX_ORB, Auxiliary Orbit"""

    attitude: Path
    """AUX_ATT, Auxiliary Attitude"""

    tec_maps: list[Path]
    """AUX_TEC, Auxiliary TEC Map(s), can be empty or with 1 or 2 values"""

    instrument_parameters: Path
    """AUX_INS, Auxiliary Instrument Parameters"""

    l1_processing_parameters: Path
    """AUX_PP1, Auxiliary L1 Processing Parameters"""

    calibration_site_information: Path | None = None
    """PARC_INFO, Calibration Site Information"""

    def get_all_aux_products_paths(self) -> list[Path]:
        """Get all the paths to the aux products"""
        aux_products_paths = [
            self.orbit,
            self.attitude,
            self.instrument_parameters,
            self.l1_processing_parameters,
        ]
        if self.calibration_site_information is not None:
            aux_products_paths.append(self.calibration_site_information)
        aux_products_paths.extend(self.tec_maps)
        return aux_products_paths


@dataclass
class L1ProcessingParameters:
    """High level processing parameters"""

    frame_id: int = 0
    """Absolute frame index of the frame to be generated"""

    frame_status: str = "NOMINAL"
    """Status of the frame to be generated"""

    range_interval: tuple[float, float] | None = None
    """Interval in range times - start, stop [s]"""

    rfi_mitigation_enabled: bool | None = None
    """Whether RFI mitigation should be performed during L1Processing (overrides AUX_PP1 conf)"""


@dataclass
class L1JobOrder:
    """Job order data for L1 processing."""

    io_products: L1StripmapProducts | L1RXOnlyProducts
    """Input/Output products depending on the required processing"""

    auxiliary_files: L1AuxiliaryProducts
    """The auxiliary files - orbit, attitude, ins, pp1, ..."""

    processing_parameters: L1ProcessingParameters
    """High level processing parameters"""

    dem_database_entry_point: Path | None
    """DEM database entry point location"""

    geomagnetic_field: Path | None
    """GMF, Geomagnetic Field"""

    iri_data_folder: Path | None
    """IRI data folder"""

    device_resources: DeviceResources
    """Available device resources"""

    processor_configuration: ProcessorConfiguration
    """Processor configuration"""

    intermediate_data_dir: Path | None
    """Intermediate data dir"""

    def keep_intermediate(self) -> bool:
        """Wether intermediates should be kept"""
        return self.processor_configuration.keep_intermediate and self.intermediate_data_dir is not None
