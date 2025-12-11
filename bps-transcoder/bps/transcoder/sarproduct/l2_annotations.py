# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""L2 common annotations utils"""

import numpy as np

COORDINATE_REFERENCE_SYSTEM = "GEOGCRS[WGS 84,DATUM[World Geodetic System 1984,ELLIPSOID[WGS 84,6378137,298.257223563,LENGTHUNIT[metre,1]]],PRIMEM[Greenwich,0,ANGLEUNIT[degree,0.0174532925199433]],CS[ellipsoidal,2],AXIS[geodetic latitude (Lat),north,ORDER[1],ANGLEUNIT[degree,0.0174532925199433]],AXIS[geodetic longitude (Lon),east,ORDER[2],ANGLEUNIT[degree,0.0174532925199433]],ID[EPSG,4326]]"


def ground_corner_points(dgg_latitude_axis: np.ndarray, dgg_longitude_axis: np.ndarray):
    # [latitude, longitude, 0.0, latitude pixel, longitude pixel]
    corner_nw = [
        float(np.max(dgg_latitude_axis)),
        float(np.min(dgg_longitude_axis)),
        float(0.0),
        int(len(dgg_latitude_axis)),
        int(0),
    ]
    corner_ne = [
        float(np.max(dgg_latitude_axis)),
        float(np.max(dgg_longitude_axis)),
        float(0.0),
        int(len(dgg_latitude_axis)),
        int(len(dgg_longitude_axis)),
    ]
    corner_se = [
        float(np.min(dgg_latitude_axis)),
        float(np.max(dgg_longitude_axis)),
        float(0.0),
        int(0),
        int(len(dgg_longitude_axis)),
    ]
    corner_sw = [
        float(np.min(dgg_latitude_axis)),
        float(np.min(dgg_longitude_axis)),
        float(0.0),
        int(0),
        int(0),
    ]

    return [corner_nw, corner_ne, corner_se, corner_sw]
