# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 Virtual Frame product module
"""

import warnings
from pathlib import Path

import numpy as np
from bps.l1_framing_processor import BPS_L1_FRAMING_PROCESSOR_ID
from bps.l1_framing_processor import __version__ as VERSION
from bps.l1_framing_processor.io.l1_vfra_models import models
from bps.l1_framing_processor.l1_framer.l1_framer import Frame
from bps.l1_framing_processor.utils.time_conversions import (
    datetime64_to_compact_date,
    datetime64_to_compact_string,
    datetime64_to_string,
)
from xsdata.exceptions import ConverterWarning
from xsdata.formats.dataclass.context import XmlContext
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig

_CONTEXT = XmlContext()
_SERIALIZER_CONFIGURATION = SerializerConfig(pretty_print=True, encoding="utf-8")
_SERIALIZER = XmlSerializer(context=_CONTEXT, config=_SERIALIZER_CONFIGURATION)

warnings.filterwarnings("ignore", category=ConverterWarning)


class L1VFRAProduct:
    """L1VFRAProduct class"""

    def __init__(self, frame: Frame, file_class="OPER", product_baseline=1):
        """Initialise L1VFRAProduct object

        Parameters
        ----------
        frame : Frame
            Frame object
        file_class : str, optional
            File class, by default OPER
        product_baseline : int, optional
            Product baseline, by default 1
        """
        self.index = frame.index
        self.start_time = frame.start_time
        self.stop_time = frame.stop_time
        self.start_angle = frame.start_angle
        self.stop_angle = frame.stop_angle
        self.status = frame.status.value
        self.duration = frame.duration

        self.file_class = file_class
        self.product_baseline = product_baseline

        self.creation_time = np.datetime64("now")

        self.__set_product_name()
        self.product_model = None
        self.l1vfra_file = None

        self.l0s_name = None
        self.l0m_name = None
        self.orbit_name = None

    def __set_product_name(self):
        """Set CPF_L1VFRA product name

        Returns
        -------
        bool
            Status (True for success, False for unsuccess)
        """
        file_class = self.file_class
        start_time = datetime64_to_compact_string(self.start_time)
        stop_time = datetime64_to_compact_string(self.stop_time)
        product_baseline = str(self.product_baseline).zfill(2)
        creation_time = datetime64_to_compact_date(self.creation_time)
        self.name = (
            "BIO_"
            + file_class
            + "_CPF_L1VFRA_"
            + start_time
            + "_"
            + stop_time
            + "_"
            + product_baseline
            + "_"
            + creation_time
            + ".EOF"
        )
        return True

    def __fill_product_model(self):
        """Fill CPF_L1VFRA product model

        Returns
        -------
        bool
            Status (True for success, False for unsuccess)
        """
        file_name = self.name[:-4]
        file_description = "L1 Virtual Frame"
        notes = ""
        mission = "BIOMASS"
        file_class = self.file_class
        file_type = "CPF_L1VFRA"
        validity_period = models.ValidityPeriodType(
            validity_start="UTC=" + datetime64_to_string(self.start_time),
            validity_stop="UTC=" + datetime64_to_string(self.stop_time),
        )
        file_version = str(self.product_baseline).zfill(4)
        source = models.SourceType(
            system="PDGS",
            creator=BPS_L1_FRAMING_PROCESSOR_ID,
            creator_version=VERSION,
            creation_date="UTC=" + datetime64_to_string(self.creation_time),
        )
        fixed_header = models.FixedHeaderType(
            file_name,
            file_description,
            notes,
            mission,
            file_class,
            file_type,
            validity_period,
            file_version,
            source,
        )

        variable_header = ""

        earth_explorer_header = models.L1VirtualFrameHeaderType(fixed_header, variable_header)

        source_l0_s = self.l0s_name
        source_l0_m = self.l0m_name
        source_aux_orb = self.orbit_name
        frame_id = str(self.index).zfill(3)
        frame_start_time = "UTC=" + datetime64_to_string(self.start_time, "us")
        frame_stop_time = "UTC=" + datetime64_to_string(self.stop_time, "us")
        frame_status = self.status
        ops_angle_start = models.AngleType(self.start_angle)
        ops_angle_start.unit = "deg"
        ops_angle_stop = models.AngleType(self.stop_angle)
        ops_angle_stop.unit = "deg"
        data_block = models.L1VirtualFrameDataBlockType(
            source_l0_s,
            source_l0_m,
            source_aux_orb,
            frame_id,
            frame_start_time,
            frame_stop_time,
            frame_status,
            ops_angle_start,
            ops_angle_stop,
        )

        schema_version = "1.0"

        self.product_model = models.EarthExplorerFile(earth_explorer_header, data_block, schema_version)

        return True

    def write_product(self, output_path, l0s_name, l0m_name, orbit_name):
        """Write CPF_L1VFRA product to disk

        Parameters
        ----------
        output_path : str
            Path to output folder
        l0s_name : str
            L0S product name
        l0m_name : str
            L0M product name
        orbit_name : str
            Orbit file name

        Returns
        -------
        bool
            Status (True for success, False for unsuccess)
        """
        self.l0s_name = l0s_name
        self.l0m_name = l0m_name
        self.orbit_name = orbit_name

        self.__fill_product_model()

        product_text = _SERIALIZER.render(self.product_model, ns_map={None: "http://eop-cfi.esa.int/CFI"})
        self.l1vfra_file = Path(output_path, self.name)
        self.l1vfra_file.write_text(product_text, encoding="utf-8")

        return True
