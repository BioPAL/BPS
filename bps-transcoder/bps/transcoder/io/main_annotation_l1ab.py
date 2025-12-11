# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Main annotation L1ab"""

from dataclasses import dataclass

from bps.common.io import common
from bps.transcoder.io import common_annotation_l1


@dataclass
class MainAnnotationL1ab:
    """L1ab main annotation"""

    acquisition_information: common_annotation_l1.AcquisitionInformationType
    sar_image: common_annotation_l1.SarImageType
    instrument_parameters: common_annotation_l1.InstrumentParametersType
    raw_data_analysis: common_annotation_l1.RawDataAnalysisType
    processing_parameters: common_annotation_l1.ProcessingParameters
    internal_calibration: common_annotation_l1.InternalCalibrationType
    rfi_mitigation: common_annotation_l1.RfiMitigationType
    doppler_parameters: common_annotation_l1.DopplerParametersType
    radiometric_calibration: common_annotation_l1.RadiometricCalibrationType
    polarimetric_distortion: common_annotation_l1.PolarimetricDistortionType
    ionosphere_correction: common_annotation_l1.IonosphereCorrection
    geometry: common_annotation_l1.GeometryType
    quality: common_annotation_l1.QualityType
    annotation_lut: list[common.LayerType]
