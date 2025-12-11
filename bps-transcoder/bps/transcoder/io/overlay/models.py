# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Overlay models
------------------
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataType:
    """
    Parameters
    ----------
    value
        Metadata value.
    name
    """

    value: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        },
    )


@dataclass
class IconType:
    """
    Parameters
    ----------
    href
        A local file specification or URL used to load the desired image and overlay it on the map.
    """

    href: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class LatLonQuadType:
    """
    Parameters
    ----------
    coordinates
        A string of 5 lon,lat coordinate pairs which describe the corners of the image. The string is of the form:
        lon,lat lon,lat lon,lat lon,lat lon,lat. The coordinates must appear in the following order: last line first
        pixel, last line last pixel, first line last pixel, first line first pixel, last line first pixel.
    """

    coordinates: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class LineStyleType:
    """
    Parameters
    ----------
    color
        Line color in 32 bit hex format (AABBGGRR order).
    width
        Line width.
    """

    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    width: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class LinearRingType:
    """
    Parameters
    ----------
    coordinates
        A string of 5 lon,lat,height coordinate tuples which describe the corners of the image. The string is of the
        form: lon,lat,height lon,lat,height lon,lat,height lon,lat,height lon,lat,height. The coordinates must
        appear in the following order: last line first pixel, last line last pixel, first line last pixel, first
        line first pixel, last line first pixel.
    """

    coordinates: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class PolyStyleType:
    """
    Parameters
    ----------
    color
        Line style.
    """

    color: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class TimeStampType:
    """
    Parameters
    ----------
    when
        Overlay time stamp in ISO 8601 format (YYYY-MM-DDThh:mm:ss).
    """

    when: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class LatLonQuad:
    class Meta:
        namespace = "http://www.google.com/kml/ext/2.2"

    coordinates: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class ExtendedDataType:
    """
    Parameters
    ----------
    data
        Metadata.
    """

    data: list[DataType] = field(
        default_factory=list,
        metadata={
            "name": "Data",
            "type": "Element",
            "namespace": "",
            "min_occurs": 1,
        },
    )


@dataclass
class GroundOverlayType:
    """
    Parameters
    ----------
    name
        Name of the ground overlay.
    visibility
        Specifies whether the feature is drawn in the 3D viewer when it is initially loaded (0: not visible, 1:
        visible).
    time_stamp
        Structure containing the time stamp of the overlay that can be used for dynamic visualizations over time.
    icon
        Structure describing the image file used on the map overlay.
    lat_lon_quad
        Structure containing the latitude and longitude coordinates used to position the image overlay on the map.
    """

    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    visibility: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    time_stamp: Optional[TimeStampType] = field(
        default=None,
        metadata={
            "name": "TimeStamp",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    icon: Optional[IconType] = field(
        default=None,
        metadata={
            "name": "Icon",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    lat_lon_quad: Optional[LatLonQuad] = field(
        default=None,
        metadata={
            "name": "LatLonQuad",
            "type": "Element",
            "namespace": "http://www.google.com/kml/ext/2.2",
            "required": True,
        },
    )


@dataclass
class StyleType:
    """
    Parameters
    ----------
    line_style
        Line style.
    poly_style
        Polygon style.
    """

    line_style: Optional[LineStyleType] = field(
        default=None,
        metadata={
            "name": "LineStyle",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    poly_style: Optional[PolyStyleType] = field(
        default=None,
        metadata={
            "name": "PolyStyle",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class OuterBoundaryIsType:
    """
    Parameters
    ----------
    linear_ring
        Linear ring description.
    """

    class Meta:
        name = "outerBoundaryIsType"

    linear_ring: Optional[LinearRingType] = field(
        default=None,
        metadata={
            "name": "LinearRing",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class PolygonType:
    """
    Parameters
    ----------
    tessellate
        Tessellate option.
    altitude_mode
        Altitude mode option.
    outer_boundary_is
        Polygon outer boundary description.
    """

    tessellate: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    altitude_mode: Optional[str] = field(
        default=None,
        metadata={
            "name": "altitudeMode",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    outer_boundary_is: Optional[OuterBoundaryIsType] = field(
        default=None,
        metadata={
            "name": "outerBoundaryIs",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class PlacemarkType:
    """
    Parameters
    ----------
    name
        Name of the placemark.
    visibility
        Specifies whether the feature is drawn in the 3D viewer when it is initially loaded (0: not visible, 1:
        visible).
    time_stamp
        Structure containing the time stamp of the placemark that can be used for dynamic visualizations over time.
    style
        Structure describing the stype used to display the placemark.
    extended_data
        Structure containing the metadata associated to the overlay.
    polygon
        Structure containing the latitude and longitude coordinates used to position the image overlay on the map.
    """

    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    visibility: Optional[int] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    time_stamp: Optional[TimeStampType] = field(
        default=None,
        metadata={
            "name": "TimeStamp",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    style: Optional[StyleType] = field(
        default=None,
        metadata={
            "name": "Style",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    extended_data: Optional[ExtendedDataType] = field(
        default=None,
        metadata={
            "name": "ExtendedData",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    polygon: Optional[PolygonType] = field(
        default=None,
        metadata={
            "name": "Polygon",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class DocumentType:
    """
    Parameters
    ----------
    name
        Name of the document.
    description
        Description of the document.
    ground_overlay
        Contains the parameters required to specify the footprint of the image and overlay the quicklook image on a
        map.
    placemark
        Contains the metadata associated to the overlay.
    """

    name: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    description: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    ground_overlay: Optional[GroundOverlayType] = field(
        default=None,
        metadata={
            "name": "GroundOverlay",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )
    placemark: Optional[PlacemarkType] = field(
        default=None,
        metadata={
            "name": "Placemark",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class KmlType:
    """
    Parameters
    ----------
    document
        Document container for KML components.
    """

    class Meta:
        name = "kmlType"

    document: Optional[DocumentType] = field(
        default=None,
        metadata={
            "name": "Document",
            "type": "Element",
            "namespace": "",
            "required": True,
        },
    )


@dataclass
class Kml(KmlType):
    """
    BIOMASS L1 products overlay element.
    """

    class Meta:
        name = "kml"
