# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
XSD Overlay models
------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(kw_only=True)
class DataType:
    """
    Parameters
    ----------
    value
        Metadata value.
    name
    """

    value: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    name: str = field(
        metadata={
            "type": "Attribute",
        },
    )


@dataclass(kw_only=True)
class IconType:
    """
    Parameters
    ----------
    href
        A local file specification or URL used to load the desired image and overlay it on the map.
    """

    href: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class LatLonQuadType:
    """
    Parameters
    ----------
    coordinates
        A string of 5 lon,lat coordinate pairs which describe the corners of the image. The string is of the form:
        lon,lat lon,lat lon,lat lon,lat lon,lat. The coordinates must appear in the following order: last line first
        pixel, last line last pixel, first line last pixel, first line first pixel, last line first pixel.
    """

    coordinates: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class LineStyleType:
    """
    Parameters
    ----------
    color
        Line color in 32 bit hex format (AABBGGRR order).
    width
        Line width.
    """

    color: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    width: int = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    coordinates: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class PolyStyleType:
    """
    Parameters
    ----------
    color
        Line style.
    """

    color: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class TimeStampType:
    """
    Parameters
    ----------
    when
        Overlay time stamp in ISO 8601 format (YYYY-MM-DDThh:mm:ss).
    """

    when: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class LatLonQuad:
    class Meta:
        namespace = "http://www.google.com/kml/ext/2.2"

    coordinates: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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


@dataclass(kw_only=True)
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

    name: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    visibility: int = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    time_stamp: TimeStampType = field(
        metadata={
            "name": "TimeStamp",
            "type": "Element",
            "namespace": "",
        },
    )
    icon: IconType = field(
        metadata={
            "name": "Icon",
            "type": "Element",
            "namespace": "",
        },
    )
    lat_lon_quad: LatLonQuad = field(
        metadata={"name": "LatLonQuad", "type": "Element", "namespace": "http://www.google.com/kml/ext/2.2"}
    )


@dataclass(kw_only=True)
class StyleType:
    """
    Parameters
    ----------
    line_style
        Line style.
    poly_style
        Polygon style.
    """

    line_style: LineStyleType = field(
        metadata={
            "name": "LineStyle",
            "type": "Element",
            "namespace": "",
        },
    )
    poly_style: PolyStyleType = field(
        metadata={
            "name": "PolyStyle",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class OuterBoundaryIsType:
    """
    Parameters
    ----------
    linear_ring
        Linear ring description.
    """

    class Meta:
        name = "outerBoundaryIsType"

    linear_ring: LinearRingType = field(
        metadata={
            "name": "LinearRing",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    tessellate: int = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    altitude_mode: str = field(
        metadata={
            "name": "altitudeMode",
            "type": "Element",
            "namespace": "",
        },
    )
    outer_boundary_is: OuterBoundaryIsType = field(
        metadata={
            "name": "outerBoundaryIs",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    name: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    visibility: int = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    time_stamp: TimeStampType = field(
        metadata={
            "name": "TimeStamp",
            "type": "Element",
            "namespace": "",
        },
    )
    style: StyleType = field(
        metadata={
            "name": "Style",
            "type": "Element",
            "namespace": "",
        },
    )
    extended_data: ExtendedDataType = field(
        metadata={
            "name": "ExtendedData",
            "type": "Element",
            "namespace": "",
        },
    )
    polygon: PolygonType = field(
        metadata={
            "name": "Polygon",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
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

    name: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    description: str = field(
        metadata={
            "type": "Element",
            "namespace": "",
        },
    )
    ground_overlay: GroundOverlayType = field(
        metadata={
            "name": "GroundOverlay",
            "type": "Element",
            "namespace": "",
        },
    )
    placemark: PlacemarkType = field(
        metadata={
            "name": "Placemark",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class KmlType:
    """
    Parameters
    ----------
    document
        Document container for KML components.
    """

    class Meta:
        name = "kmlType"

    document: DocumentType = field(
        metadata={
            "name": "Document",
            "type": "Element",
            "namespace": "",
        },
    )


@dataclass(kw_only=True)
class Kml(KmlType):
    """
    BIOMASS L1 products overlay element.
    """

    class Meta:
        name = "kml"
