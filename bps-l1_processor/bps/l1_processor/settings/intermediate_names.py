# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Intermediate products names
---------------------------
"""

from enum import Enum, auto

from bps.l1_processor.parc.parc_info import ScatteringResponse

# Files in base dir
BPS_CONF_FILE_NAME = "BPSConf.xml"
"""Name of the executable bps configuration file"""
BPS_L1_CORE_PROCESSOR_STATUS_FILE_NAME = "BPSL1CoreProcessorStatusFile.xml"
"""Name of the l1 core processor status file"""
BPS_L1_PROCESSOR_STATUS_FILE_NAME = "BPSL1ProcessorStatusFile.json"
"""Name of the l1 processor status file"""

# Main folders
L1_PRE_PROC_OUTPUT_FOLDER = "l1_pre_processor_output"
"""Name of l1 pre-processor output folder"""
L1_CORE_PROC_OUTPUT_FOLDER = "l1_core_processor_output"
"""Name of l1 core processor output folder"""
L1_PARC_PROC_OUTPUT_FOLDERS = {
    ScatteringResponse.GT1: "parc_Gt1",
    ScatteringResponse.GT2: "parc_Gt2",
    ScatteringResponse.X: "parc_X",
    ScatteringResponse.Y: "parc_Y",
}
"""Name of l1 parc processor output folders"""

# Pre processor intermediates
EXTRACTED_RAW_PRODUCT = "iRAW"
"""Name of extracted raw data product"""
EXTRACTED_DYNCAL_PRODUCT = "iRAWDynCAL"
"""Name of extracted dynamic calibration data product"""
PGP_PRODUCT = "iPGP"
"""Name of extracted PGP product"""
AMP_PHASE_DRIFT_PRODUCT = "iDrift"
"""Name of extracted amplitude and phase drifts product"""
CHIRP_REPLICA_PRODUCT = "iChirp"
"""Name of extracted chirp replica product"""
INTERNAL_DELAYS_FILE = "iDelays.xml"
"""Name of extracted internal delays file"""
CHANNEL_IMBALANCE_FILE = "iChannelImbalance.xml"
"""Name of extracted channel imbalance file"""
TX_POWER_TRACKING_PRODUCT = "iTxPT"
"""Name of extracted transmit power tracking product"""
EST_NOISE_PRODUCT = "iEstNoise"
"""Name of extracted estimated noise product"""
REPAIRED_ATTITUDE = "attitude_repaired.xml"
"""Name of the repaired attitude xml file"""
SSP_HEADERS_FILE = "ssp_headers.csv"
"""Name of ssp headers csv file"""
L1_PREPROC_REPORT = "iL1PreProcReport.xml"
"""Name of l1 preproc report xml file"""

ANT_D1_H_PRODUCT = "iPatt2D_D1H"
"""Name of the antenna pattern product for doublet 1 and polarization H"""
ANT_D1_V_PRODUCT = "iPatt2D_D1V"
"""Name of the antenna pattern product for doublet 1 and polarization V"""
ANT_D2_H_PRODUCT = "iPatt2D_D2H"
"""Name of the antenna pattern product for doublet 2 and polarization H"""
ANT_D2_V_PRODUCT = "iPatt2D_D2V"
"""Name of the antenna pattern product for doublet 2 and polarization V"""

FOOTPRINT_FILE_NAME = "footprint.xml"
"""Name of the footprint file inside intermediates"""


class IntermediateProductID(Enum):
    """Output of the BPSL1CoreProcessor, intermediates of the bps l1 processor"""

    RAW_MITIGATED = auto()
    RFI_TIME_MASK = auto()
    RFI_FREQ_MASK = auto()
    RGC_DC_FR_ESTIMATOR = auto()
    DOPPLER_CENTROID_ESTIMATOR_GRID = auto()
    SLANT_DEM = auto()
    SLC = auto()
    SLC_IONO_CORRECTED = auto()
    IONO_CAL_REPORT = auto()
    FR = auto()
    FR_PLANE = auto()
    PHASE_SCREEN_BB = auto()
    SLC_AF_CORRECTED = auto()
    PHASE_SCREEN_AF = auto()
    SRD_MULTILOOKED = auto()
    SLC_NESZ_MAP = auto()
    SRD_DENOISED = auto()
    GRD = auto()

    def to_name(self) -> str:
        """Retrieve product name"""

        name_dict = {
            self.RFI_TIME_MASK: "iRFI_time_domain_mask",  # Required for LUTs
            self.RFI_FREQ_MASK: "iRFI_freq_domain_mask",  # Required for LUTs
            self.SLANT_DEM: "iSlantDEM",  # Required for LUTs
            self.IONO_CAL_REPORT: "iIonosphericCalibrationReport.xml",  # Required for annotations
            self.FR: "iFR",  # Required for LUTs
            self.FR_PLANE: "iFRPlane",  # Required for LUTs
            self.PHASE_SCREEN_BB: "iPhaseScreenBB",  # Required for LUTs
            self.PHASE_SCREEN_AF: "iPhaseScreenAF",  # Required for LUTs
            self.SLC_NESZ_MAP: "iSLC_nesz_map",  # Required for LUTs
            self.RAW_MITIGATED: "iRAW_rfi_corrected",
            self.RGC_DC_FR_ESTIMATOR: "iRGC",
            self.SLC: "iSLC",  # Main output
            self.SLC_IONO_CORRECTED: "iSLC_iono_corrected",  # Main output
            self.SLC_AF_CORRECTED: "iSLC_af_corrected",  # Main output
            self.SRD_MULTILOOKED: "iSRD",
            self.SRD_DENOISED: "iSRD_denoised",
            self.GRD: "iGRD",  # Main output
            self.DOPPLER_CENTROID_ESTIMATOR_GRID: "iDCGrid",  # Required for combined DC annotations
        }

        name = name_dict.get(self)
        if not name:
            raise RuntimeError(f"{self.name} label not supported")

        return name


IONOSPHERIC_HEIGHT_MODEL_FILE = "iIonosphericHeightModel.xml"
"""Name of the file containing the height of the ionosphere"""
RESAMPLING_FILTER_PRODUCT = "iResamplingFilter"
"""Resampling filter product, used for azimuth resampling"""
GEOMETRIC_DC_PRODUCT = "iGeometricDC"
"""Geometric dc product, available for annotations when geometry is not the dc est method used in processing"""
EXTRACTED_RAW_ANNOTATION = "iReferenceExtractedRAWAnnotation.xml"
"""Reference metadata with RAW annotations"""
