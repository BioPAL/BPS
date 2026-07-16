# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Restrict XSD types to BPS
-------------------------

This script trims the XSD types from our general code base that are not
relevant to the BPS. Usage:

1) Download/clone the XSD's from the BrickSAR repo
2) Make sure to checkout the correct branch
3) Create the folder /path/to/bps/xsd/bricksar-xsd
4) Copy all Biomass XSD to /path/to/bps/xsd/bricksar-xsd/
5) Run the following 3 commands

   $ python restrict_xsd_to_bps.py                                               \
         --input_xsd /path/to/BrickSAR/xsd/aresys_generic_inputfile.xsd          \
         --output_xsd /path/to/bps/xsd/bricksar-xsd/aresys_generic_inputfile.xsd \
         --xsd_type input_file

   $ python restrict_xsd_to_bps.py                                          \
         --input_xsd /path/to/BrickSAR/xsd/aresys_generic_conf.xsd          \
         --output_xsd /path/to/bps/xsd/bricksar-xsd/aresys_generic_conf.xsd \
         --xsd_type generic_conf

   $ python restrict_xsd_to_bps.py                                \
         --input_xsd /path/to/BrickSAR/xsd/ConfTypes.xsd          \
         --output_xsd /path/to/bps/xsd/bricksar-xsd/ConfTypes.xsd \
         --xsd_type conf_types

"""

import xml.etree.ElementTree as Et
from pathlib import Path

import click

bps_inputfile_relevant_tags = [
    "AresysXmlInput",
    "AresysXmlInputType",
    "BPSL1CoreProcessorInputType",
    "BPSStackProcessorInputType",
    "BiomassL0ImportPreProcType",
    "DemXYZProductChannelsType",
    "EarthGeometryType",
    "GenericCoregistratorInputType",
    "InputBiomassAntennaPattern2DType",
    "InputDirectoryType",
    "InputFileType",
    "InputProductType",
    "OneWayAntennaPatternType",
    "OutputDirectoryType",
    "OutputFileType",
    "OutputProductType",
    "OutputSSPHeadersFileType",
    "PFSelectorAreaType",
    "PFSelectorAzimuthTimeIntervalType",
    "PFSelectorGeographicCoordinatesType",
    "PFSelectorIndexIntervalType",
    "PFSelectorPolarizationsType",
    "PFSelectorRangeTimeIntervalType",
    "PFSelectorRasterCoordinatesSwathType",
    "PFSelectorRasterCoordinatesType",
    "PFSelectorSwathNameType",
    "PFSelectorSwathNamesType",
    "PFSelectorSwathsBurstsSwathType",
    "PFSelectorSwathsBurstsType",
    "PFSelectorTimeCoordinatesType",
    "PolarizationBaseType",
    "SAFEPolarizationType",
    "SARFOCInput",
    "SARFOCInputPatternsType",
    "SARFOCOneWayPatternsType",
    "SARFOCTwoWayPatternsType",
    "SubSceneTimeCornersType",
    "TiePointType",
    "TimeOfInterestType",
]

bps_conf_relevant_tags = [
    "AresysXmlDoc",
    "AresysXmlDocType",
    "AzimuthConf",
    "BPSConf",
    "BPSL1CoreProcessorConf",
    "BiomassL0ImportPreProcConf",
    "CalibrationConstantsConf",
    "ChannelType",
    "DCEstConfStripmap",
    "InterferometerConf",
    "IonosphericCalibrationConf",
    "MultiProcessorConf",
    "NoiseMapGeneratorConf",
    "PolarimetricProcessorConf",
    "RFIMitigationConf",
    "RangeCompensatorConf",
    "RangeConf",
    "STAProcessorConf",
    "Slant2GroundConf",
    "SynthGeometryConf",
]

bps_conf_types_relevant_tags = [
    "ATTITUDE_FITTING_TYPES",
    "AntennaShiftCompensationModeType",
    "AzimuthConfType",
    "BISTATIC_DELAY_CORRECTION_TYPES",
    "BPSAntennaPatternCompensationType",
    "BPSConfType",
    "BPSL1CoreProcessingSettingsType",
    "BPSL1CoreProcessorConfType",
    "BPSL1CoreProcessorInterfaceSettingsType",
    "BPSLOGLEVELS",
    "BPSLoggerConfType",
    "BaseConfType",
    "BiomassIntCalConfType",
    "BiomassL0ImportConfType",
    "BiomassL0ImportPreProcConfType",
    "BiomassRawDataCorrectionsConfType",
    "COMPLEX_ALG",
    "COMPLEX_POL",
    "COREG_MODE",
    "CalibrationConstantsConfType",
    "ComplexNumberType",
    "CoregConfType",
    "CoregistrationConfType",
    "CoregistrationOutputProductsConfType",
    "DataRefinementConfType",
    "DATA_REFINEMENT_MODE",
    "DCCoreAlgorithmType",
    "DC_ESTIMATION_METHODS_TYPES",
    "DEM_TYPES",
    "DigitalElevationModelType",
    "EARTH_GEOMETRY",
    "FCOMPLEX",
    "FILTER_MASK",
    "FLAG",
    "FOCUSING_METHOD_TYPES",
    "FinefitConfType",
    "FormatType",
    "FullAccuracyConfType",
    "FullAccuracyPostProcessingConfType",
    "FullAccuracyPreProcessingConfType",
    "GenericInterferometerConfType",
    "INTERP_TYPE",
    "IonosphericCalibrationConfType",
    "LOGLEVELS",
    "LoggerConfType",
    "MemorySizeType",
    "MemorySizeType",
    "MemorySizeType",
    "MultilookConfType",
    "MultilookDirectionConfType",
    "NORMALIZED_DOUBLE",
    "NoiseMapGeneratorConfType",
    "NonStationaryCoregConfType",
    "OutputBorderPolicyType",
    "OutputQuantityType",
    "POLY_ESTIMATION_CONSTRAINT_TYPES",
    "PolarimetricProcessorConfType",
    "PolarizationType",
    "ReinterpolationConfType",
    "RFIFrequencyDomainRemovalConfType",
    "RFIFrequencyDomainRemovalConfType",
    "RFIMaskCompositionMethodsType",
    "RFIMitigationConfType",
    "RFIMitigationMethodsType",
    "RFITimeDomainRemovalConfType",
    "RangeCompensatorConfType",
    "RangeConfType",
    "RangeFocusingMethodType",
    "RealValueDegrees",
    "RealValueMeters",
    "RefineFitConfType",
    "STAProcessorConfType",
    "SarfocDigitalElevationModelType",
    "SarfocExternalResourcesType",
    "SarfocOutputProductsType",
    "SarfocProcessingSettingsType",
    "SarfocProcessingStepType",
    "SarfocProcessingStepsType",
    "SarfocProductType",
    "Slant2GroundConfType",
    "StripmapDcConfType",
    "SynthGeometryConfType",
    "UNIT_TYPES",
    "WINDOWS",
    "WindowConfType",
]


def type_filter(relevant_types: list[str]):
    def irrelevant_tags_by_type(elem: Et.Element):
        try:
            name = elem.attrib["type"]
        except KeyError:
            return False
        else:
            return name not in relevant_types

    return irrelevant_tags_by_type


def name_filter(relevant_names: list[str]):
    def irrelevant_tags_by_name(elem: Et.Element):
        try:
            name = elem.attrib["name"]
        except KeyError:
            return False
        else:
            return name not in relevant_names

    return irrelevant_tags_by_name


def find_first_child_by_name(parent: Et.Element, name: str) -> Et.Element | None:
    return next((child for child in parent if child.get("name", default="") == name))


def restrict_generic_input_file(root: Et.Element):
    irrelevant_children = list(filter(name_filter(bps_inputfile_relevant_tags), root))
    for child in irrelevant_children:
        root.remove(child)

    input_base_elem = find_first_child_by_name(root, "AresysXmlInputType")
    assert input_base_elem is not None
    step_elem = input_base_elem[0][0]
    assert step_elem.get("name", default="") == "Step"

    choice_elem = step_elem[0][0]
    irrelevant_options = list(filter(type_filter(bps_inputfile_relevant_tags), choice_elem))
    for option in irrelevant_options:
        choice_elem.remove(option)


def restrict_generic_conf_file(root: Et.Element):
    irrelevant_children = list(filter(name_filter(bps_conf_relevant_tags), root))
    for child in irrelevant_children:
        root.remove(child)

    conf_base_elem = find_first_child_by_name(root, "AresysXmlDocType")
    assert conf_base_elem is not None

    channel_elem = conf_base_elem[0][3][0]
    assert channel_elem.get("name", default="") == "Channel"

    choice_elem = channel_elem[0][0][0][0]
    irrelevant_options = list(filter(name_filter(bps_conf_relevant_tags), choice_elem))
    for option in irrelevant_options:
        choice_elem.remove(option)


def restrict_generic_conf_types_file(root: Et.Element):
    irrelevant_children = list(filter(name_filter(bps_conf_types_relevant_tags), root))
    for child in irrelevant_children:
        root.remove(child)


def restrict_xsd_to_biomass(xsd_type, xsd_file_in: Path, xsd_file_out: Path):
    tree = Et.parse(xsd_file_in)
    root = tree.getroot()

    if xsd_type == "input_file":
        restrict_generic_input_file(root)
    elif xsd_type == "generic_conf":
        restrict_generic_conf_file(root)
        root.set("xmlns:cf", "aresysConfTypes")
    elif xsd_type == "conf_types":
        restrict_generic_conf_types_file(root)
        root.set("xmlns:act", "aresysConfTypes")
    else:
        raise RuntimeError

    xsd_file_out.parent.mkdir(exist_ok=True, parents=True)
    tree.write(
        xsd_file_out,
        encoding="utf-8",
        xml_declaration=True,
        default_namespace=None,
        method="xml",
    )


@click.command()
@click.option(
    "--xsd_type",
    nargs=1,
    type=click.Choice(["input_file", "generic_conf", "conf_types"], case_sensitive=False),
    required=True,
)
@click.option("--input_xsd", nargs=1, type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output_xsd", nargs=1, type=click.Path(dir_okay=False), required=True)
def main(xsd_type: str, input_xsd: str, output_xsd: str):
    click.echo(f"Type xsd: {xsd_type}")
    click.echo(f"Input xsd: {input_xsd}")
    click.echo(f"Output xsd: {output_xsd}")
    restrict_xsd_to_biomass(xsd_type, Path(input_xsd), Path(output_xsd))


if __name__ == "__main__":
    main()  # type: ignore
