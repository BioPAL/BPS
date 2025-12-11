# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Parsing
-------
"""

from pathlib import Path
from typing import Any

import xmlschema
import xsdata.exceptions
from xsdata.formats.dataclass.context import XmlContext
from xsdata.formats.dataclass.parsers import XmlParser
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig

_CONTEXT = XmlContext()
_PARSER = XmlParser(context=_CONTEXT)
_SERIALIZER_CONFIGURATION = SerializerConfig(pretty_print=True, encoding="utf-8")
_SERIALIZER = XmlSerializer(context=_CONTEXT, config=_SERIALIZER_CONFIGURATION)


class ParsingError(RuntimeError):
    """Raise when the XML parsing fails"""


def parse(xml_string: str, model: type) -> Any:
    """Parse a string according to an XSD model

    Parameters
    ----------
    xml_string : str
        input xml string to parse
    model : Type
        xsd model type

    Returns
    -------
    Any
        The content as a structure of type model

    Raises
    ------
    ParsingError
        in case the xml_string is incompatible with the XSD model
    """
    try:
        model = _PARSER.from_string(xml_string, model)
    except xsdata.exceptions.ParserError as exc:
        raise ParsingError(exc) from exc

    return model


def serialize(model: Any, **kwargs) -> str:
    """Serial an XSD object to a string

    Parameters
    ----------
    model :
        Object to serialize

    kwargs are forwarded to serialized render method

    Returns
    -------
    str
        XML string
    """
    return _SERIALIZER.render(model, **kwargs)


def validate(*, xml_file: Path, schema: Path):
    """Validate xml file with respect to xsd"""

    try:
        xmlschema.validate(xml_file, schema=schema)
    except xmlschema.XMLSchemaValidationError as validation_error:
        raise ParsingError(f"XML file {xml_file} does not validate: {validation_error.reason}") from validation_error
