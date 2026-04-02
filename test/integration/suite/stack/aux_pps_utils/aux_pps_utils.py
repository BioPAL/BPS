# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to serialize AUX-PPS for tests
----------------------------------------
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from bps.stack_processor.interface.external.aux_pps import (
    AuxPPSInvalidSchema,
    validate_aux_pps_xsd_schema,
)

# NOTE: This must be updated at each AUX_FMT release.
AUX_FMT_VERSION = "3.6.0"


# NOTE: This must be updated at each AUX_FTM update. This runs the *Nominal*
# (i.e. w/ IOS) stack execution w/ default parameters. Note that
# polarizationCombinationMethod cannot be None otherwise the L2 processor
# stops.
__AUX_PPS_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<auxiliarySTAProcessingParameters  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="../support/bio-aux-pps.xsd">
  <staProductList count="1">
    <staProduct>
      <productID>Sx_STA__1S</productID>
      <general>
        <polarisationCombinationMethod>Average</polarisationCombinationMethod>
        <outerParallelizationMaxCores>0</outerParallelizationMaxCores>
        <allowDuplicateImagesFlag>true</allowDuplicateImagesFlag>
        <flatteningPhaseBiasCompensationFlag>false</flatteningPhaseBiasCompensationFlag>
      </general>
      <primaryImageSelection>
        <primaryImageSelectionInformation>Geometry</primaryImageSelectionInformation>
        <rfiDecorrelationThreshold>1</rfiDecorrelationThreshold>
        <faradayDecorrelationThreshold>1</faradayDecorrelationThreshold>
      </primaryImageSelection>
      <coregistration>
        <coregistrationMethod>Geometry and Data</coregistrationMethod>
        <residualShiftsQualityThreshold>0.0</residualShiftsQualityThreshold>
        <polarisationUsed>HH</polarisationUsed>
        <blockQualityThreshold>0.0</blockQualityThreshold>
        <fittingQualityThreshold>0.0</fittingQualityThreshold>
        <minValidBlocks>-1</minValidBlocks>
        <azimuthMaxShift>5</azimuthMaxShift>
        <azimuthBlockSize>101</azimuthBlockSize>
        <azimuthMinOverlap>0</azimuthMinOverlap>
        <rangeMaxShift>5</rangeMaxShift>
        <rangeBlockSize>51</rangeBlockSize>
        <rangeMinOverlap>0</rangeMinOverlap>
        <modelBasedFitFlag>true</modelBasedFitFlag>
        <lowPassFilterType>Average</lowPassFilterType>
        <lowPassFilterOrder>9</lowPassFilterOrder>
        <lowPassFilterStdDev>0.84089642</lowPassFilterStdDev>
        <exportDebugProductsFlag>false</exportDebugProductsFlag>
        <coregistrationExecutionPolicy>Nominal</coregistrationExecutionPolicy>
      </coregistration>
      <rfiDegradationEstimation>
        <rfiDegradationEstimationFlag>true</rfiDegradationEstimationFlag>
      </rfiDegradationEstimation>
      <azimuthSpectralFiltering>
        <azimuthSpectralFilteringFlag>true</azimuthSpectralFilteringFlag>
        <usePrimaryWeightingWindowFlag>true</usePrimaryWeightingWindowFlag>
        <spectralWeightingWindow>Hamming</spectralWeightingWindow>
        <spectralWeightingWindowParameter>0.54</spectralWeightingWindowParameter>
        <use32bitFlag>true</use32bitFlag>
      </azimuthSpectralFiltering>
      <slowIonosphereRemoval>
        <slowIonosphereRemovalFlag>false</slowIonosphereRemovalFlag>
        <primaryImageFlag>true</primaryImageFlag>
        <polarisationUsed>HH</polarisationUsed>
        <compensateL1IonoPhaseScreenFlag>true</compensateL1IonoPhaseScreenFlag>
        <rangeLookBandwidth>0.3</rangeLookBandwidth>
        <rangeLookFrequency>0.3</rangeLookFrequency>
        <phaseUnwrappingFlag>true</phaseUnwrappingFlag>
        <latitudeThreshold units="deg">90.0</latitudeThreshold>
        <baselineMethod>Multi-Baseline</baselineMethod>
        <unweightedMultiBaselineEstimation>false</unweightedMultiBaselineEstimation>
        <slowIonosphereQualityThreshold>0.0</slowIonosphereQualityThreshold>
        <sublookWindowAzimuthSize>501</sublookWindowAzimuthSize>
        <sublookWindowRangeSize>41</sublookWindowRangeSize>
        <multiBaselineCriticalBaselineThreshold>1.0</multiBaselineCriticalBaselineThreshold>
        <minCoherenceThreshold>0.0</minCoherenceThreshold>
        <minUsablePixelRatio>0.0</minUsablePixelRatio>
        <maxDeltaPhaseUnwrapTest units="rad">-1.0</maxDeltaPhaseUnwrapTest>
        <use32bitFlag>true</use32bitFlag>
      </slowIonosphereRemoval>
      <multiSquintCalibration>
        <multiSquintCalibrationFlag>true</multiSquintCalibrationFlag>
        <polarisationUsed>HH</polarisationUsed>
        <seedRangeCoherenceThreshold>0.3</seedRangeCoherenceThreshold>
        <edgeGuard>11</edgeGuard>
        <validAzimuthThreshold>0.1</validAzimuthThreshold>
        <propagationStepInversionMethod>Projection</propagationStepInversionMethod>
        <minIonoRangeCandidate units="m">200000</minIonoRangeCandidate>
        <maxIonoRangeCandidate units="m">700000</maxIonoRangeCandidate>
        <numIonoRangeCandidates>25</numIonoRangeCandidates>
        <robustLayerEstimationHighResNumSlices>5</robustLayerEstimationHighResNumSlices>
        <robustLayerEstimationLowResNumSlices>3</robustLayerEstimationLowResNumSlices>
        <layerAdditionThreshold>0.005</layerAdditionThreshold>
        <maxNumIonosphereLayers>5</maxNumIonosphereLayers>
        <maxIonoRangeOffset units="m">50000</maxIonoRangeOffset>
        <ionoRangeEstimationMaxIterationsDerivative>10</ionoRangeEstimationMaxIterationsDerivative>
        <ionoRangeEstimationMaxIterationsProjection>10</ionoRangeEstimationMaxIterationsProjection>
        <ionoRangeEstimationConvergenceThresholdDerivative>0.0</ionoRangeEstimationConvergenceThresholdDerivative>
        <ionoRangeEstimationConvergenceThresholdProjection>0.0</ionoRangeEstimationConvergenceThresholdProjection>
        <ionoRangeSeedingMaxIterationsDerivative>10</ionoRangeSeedingMaxIterationsDerivative>
        <ionoRangeSeedingMaxIterationsProjection>10</ionoRangeSeedingMaxIterationsProjection>
        <ionoRangeSeedingConvergenceThresholdDerivative>0.0</ionoRangeSeedingConvergenceThresholdDerivative>
        <ionoRangeSeedingConvergenceThresholdProjection>0.0</ionoRangeSeedingConvergenceThresholdProjection>
        <ionoRangePropagationMaxIterationsDerivative>10</ionoRangePropagationMaxIterationsDerivative>
        <ionoRangePropagationMaxIterationsProjection>10</ionoRangePropagationMaxIterationsProjection>
        <ionoRangePropagationConvergenceThresholdDerivative>0.0</ionoRangePropagationConvergenceThresholdDerivative>
        <ionoRangePropagationConvergenceThresholdProjection>0.0</ionoRangePropagationConvergenceThresholdProjection>
        <forceMultiSquintIonoCorrectionFlag>false</forceMultiSquintIonoCorrectionFlag>
        <disableMultiSquintIonoCorrectionFlag>true</disableMultiSquintIonoCorrectionFlag>
        <coherenceAzimuthWindowSize units="m">100.0</coherenceAzimuthWindowSize>
        <coherenceRangeWindowSize units="m">100.0</coherenceRangeWindowSize>
        <multiSquintCoherenceSubbandResolution units="m">300.0</multiSquintCoherenceSubbandResolution>
        <multiSquintCoherenceHighResAzimuthWindowSize units="m">0.0</multiSquintCoherenceHighResAzimuthWindowSize>
        <multiSquintCoherenceHighResRangeWindowSize units="m">250.0</multiSquintCoherenceHighResRangeWindowSize>
        <multiSquintCoherenceLowResAzimuthWindowSize units="m">250.0</multiSquintCoherenceLowResAzimuthWindowSize>
        <multiSquintCoherenceLowResRangeWindowSize units="m">300.0</multiSquintCoherenceLowResRangeWindowSize>
        <multiSquintCoherenceValidityThreshold>0.3</multiSquintCoherenceValidityThreshold>
        <multiSquintCoherenceImprovementThreshold>0.3</multiSquintCoherenceImprovementThreshold>
        <multiSquintCoherenceLowResThreshold>0.1</multiSquintCoherenceLowResThreshold>
        <use32bitEstimationFlag>true</use32bitEstimationFlag>
        <use32bitCorrectionFlag>false</use32bitCorrectionFlag>
      </multiSquintCalibration>
      <skpPhaseCalibration>
        <skpPhaseEstimationFlag>true</skpPhaseEstimationFlag>
        <phaseCorrection>None</phaseCorrection>
        <estimationWindowSize units="m">500</estimationWindowSize>
        <skpCalibrationPhaseScreenQualityThreshold>0.0</skpCalibrationPhaseScreenQualityThreshold>
        <overallProductQualityThreshold>0.0</overallProductQualityThreshold>
        <calibrationPhasePostprocessing>None</calibrationPhasePostprocessing>
        <postprocessingFilterWindowSize units="m">1000.0</postprocessingFilterWindowSize>
        <goldsteinFilterWindowSize>5</goldsteinFilterWindowSize>
        <goldsteinFilterAlpha>0.5</goldsteinFilterAlpha>
        <excludeMPMBPolarizationCrossCovarianceFlag>false</excludeMPMBPolarizationCrossCovarianceFlag>
        <crossPolMergingFlag>false</crossPolMergingFlag>
        <use32bitFlag>true</use32bitFlag>
      </skpPhaseCalibration>
      <l1cProductExport>
        <l1ProductDOI>DOI</l1ProductDOI>
        <pixelRepresentation>Abs Phase</pixelRepresentation>
        <pixelQuantity>Beta-Nought</pixelQuantity>
        <absCompressionMethod>LERC_ZSTD</absCompressionMethod>
        <absMaxZError>0.0001</absMaxZError>
        <absMaxZErrorPercentile>0.0</absMaxZErrorPercentile>
        <phaseCompressionMethod>LERC_ZSTD</phaseCompressionMethod>
        <phaseMaxZError units="rad">0.001</phaseMaxZError>
        <phaseMaxZErrorPercentile>0.0</phaseMaxZErrorPercentile>
        <noPixelValue>-9999.0</noPixelValue>
        <l1cQlColourCodingMethod>Pauli</l1cQlColourCodingMethod>
        <qlRangeDecimationFactor>2</qlRangeDecimationFactor>
        <qlRangeAveragingFactor>3</qlRangeAveragingFactor>
        <qlAzimuthDecimationFactor>12</qlAzimuthDecimationFactor>
        <qlAzimuthAveragingFactor>13</qlAzimuthAveragingFactor>
        <qlAbsoluteScalingFactor>1</qlAbsoluteScalingFactor>
        <qlExportCoherenceFlag>true</qlExportCoherenceFlag>
        <qlExportInterferogramFlag>true</qlExportInterferogramFlag>
      </l1cProductExport>
    </staProduct>
  </staProductList>
</auxiliarySTAProcessingParameters>
"""


def initialize_aux_pps(aux_pps_path: str | Path):
    """Write the AUX_PPS file to path."""
    # Write to disk.
    with Path(aux_pps_path).open(mode="w", encoding="utf-8") as f:
        f.write(__AUX_PPS_CONTENT)

    # Validate the template.
    try:
        validate_aux_pps_xsd_schema(aux_pps_path)
    except AuxPPSInvalidSchema as exc:
        print(exc)
        print(f"\n\nInvalid AUX-PPS template (AUX_FMT v{AUX_FMT_VERSION}). Update the template in {__file__}\n\n")
        sys.exit(1)


if __name__ == "__main__":
    with TemporaryDirectory() as tmp_dir:
        aux_pps_path = Path(tmp_dir) / "aux_pps.xml"
        initialize_aux_pps(aux_pps_path)
