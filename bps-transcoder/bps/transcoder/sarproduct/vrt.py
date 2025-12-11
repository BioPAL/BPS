# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""VRT utilities"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from arepytools.geometry.conversions import xyz2llh
from bps.common.io.parsing import serialize
from bps.transcoder.io import vrt
from bps.transcoder.utils.gdal_utils import GeotiffConf, GeotiffMetadata, get_wgs84_wkt


@dataclass
class VRTInfo:
    """Info required to fill vrt file"""

    raster_xsize: int
    raster_ysize: int
    ground_corner_points_ecef: list[list[float]]

    geotiff_metadata: GeotiffMetadata

    abs_measurement_file: Path
    phase_measurement_file: Path

    abs_geotiff_conf: GeotiffConf
    phase_geotiff_conf: GeotiffConf


def translate_metadata(metadata: dict[str, str]) -> vrt.MetadataType:
    """Translate metadata to corresponding xsdata object"""
    return vrt.MetadataType(any_element=[vrt.Mditype(value=value, key=key) for key, value in metadata.items()])


def translate_vrt_file(vrt_info: VRTInfo) -> vrt.Vrtdataset:
    """Write VRT files"""

    reference_geotiff_conf = vrt_info.abs_geotiff_conf
    measurement_files = [vrt_info.abs_measurement_file, vrt_info.phase_measurement_file]

    def gcp_to_gcp_type(count: int, gcp: list) -> vrt.Gcptype:
        """Convert gcp into vrt models type"""
        llh = xyz2llh([gcp[0], gcp[1], gcp[2]]).squeeze()
        return vrt.Gcptype(
            id=str(count + 1),
            info="GCP" + str(count + 1),
            pixel=gcp[3],
            line=gcp[4],
            x=float(np.rad2deg(llh[1])),  # X -->longitude
            y=float(np.rad2deg(llh[0])),  # Y -->latitude
            z=float(llh[2]),
        )

    gcp_list = [gcp_to_gcp_type(gcp_count, gcp) for gcp_count, gcp in enumerate(vrt_info.ground_corner_points_ecef)]
    longitudes = [gcp.x for gcp in gcp_list]
    if max(longitudes) - min(longitudes) > 180:
        for gcp in gcp_list:
            if gcp.x < 0:
                gcp.x += 360.0
    gcp_vrt_list = [vrt.GcplistType(gcp=gcp_list, projection=get_wgs84_wkt())]

    data_type = vrt.DataTypeType.CFLOAT32
    pixel_function_type_list = ["polar"]
    pixel_function_arguments_list = [
        vrt.VrtrasterBandType.PixelFunctionArguments(any_attributes={"amplitude_type": "AMPLITUDE"})
    ]

    vrtraster_band_list = []
    assert vrt_info.geotiff_metadata.polarizations is not None
    for channel_id, polarization in enumerate(vrt_info.geotiff_metadata.polarizations):
        band = channel_id + 1
        sub_class = vrt.VrtrasterBandSubTypeType.VRTDERIVED_RASTER_BAND

        description_list = [polarization]

        source_transfer_type_list = [vrt.DataTypeType.FLOAT32]
        no_data_value_element_list = [reference_geotiff_conf.nodata_value]
        color_interp_list = [vrt.ColorInterpType.GRAY]

        simple_source_list = []
        for file in measurement_files:
            if file.exists():
                source_filename = [vrt.SourceFilenameType(value=file.name, relative_to_vrt=vrt.ZeroOrOne.VALUE_1)]
                source_band = [str(band)]
                source_properties = [
                    vrt.SourcePropertiesType(
                        raster_xsize=vrt_info.raster_xsize,
                        raster_ysize=vrt_info.raster_ysize,
                    )
                ]
                simple_source = vrt.SimpleSourceType(
                    source_filename=source_filename,
                    source_band=source_band,
                    source_properties=source_properties,
                )
                simple_source_list.append(simple_source)

        vrtraster_band = vrt.VrtrasterBandType(
            description=description_list,
            metadata=[translate_metadata({"POLARIMETRIC_INTERP": polarization})],
            pixel_function_type=pixel_function_type_list,
            pixel_function_arguments=pixel_function_arguments_list,
            source_transfer_type=source_transfer_type_list,
            no_data_value_element=no_data_value_element_list,
            color_interp=color_interp_list,
            simple_source=simple_source_list,
            data_type=data_type,
            band=band,
            sub_class=sub_class,
        )
        vrtraster_band_list.append(vrtraster_band)

    metadata = {
        "AREA_OR_POINT": "Area",
        "MATRIX_REPRESENTATION": vrt_info.geotiff_metadata.matrix_representation,
        "ABS_MAX_Z_ERROR": str(vrt_info.abs_geotiff_conf.max_z_error),
        "PHASE_MAX_Z_ERROR": str(vrt_info.phase_geotiff_conf.max_z_error),
        "PolarisationsSequence": " ".join(vrt_info.geotiff_metadata.polarizations),
        "Swath": vrt_info.geotiff_metadata.swath,
        "DATETIME": vrt_info.geotiff_metadata.creation_date,
        "IMAGEDESCRIPTION": vrt_info.geotiff_metadata.description,
        "SOFTWARE": vrt_info.geotiff_metadata.software,
    }
    return vrt.Vrtdataset(
        gcplist=gcp_vrt_list,
        metadata=[translate_metadata(metadata)],
        vrtraster_band=vrtraster_band_list,
        raster_xsize=vrt_info.raster_xsize,
        raster_ysize=vrt_info.raster_ysize,
    )


def write_vrt_file(vrt_file: Path, vrt_info: VRTInfo):
    """Write vrt file"""

    vrt_model = translate_vrt_file(vrt_info)

    vrt_text = serialize(vrt_model)
    vrt_file.write_text(vrt_text, encoding="utf-8")
