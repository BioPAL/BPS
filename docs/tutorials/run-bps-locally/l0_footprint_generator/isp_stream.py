# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""ISP Stream Module"""

import datetime
import logging
import os

import pandas as pd
from generic_isp import PACKET_HEADER_LENGTH, GenericIsp, PacketHeader
from sar_isp import BiomassSarIsp


class BiomassIspStream:
    """Class to read and decode a BIOMASS ISP stream from a file."""

    def __init__(self, isp_stream_file, *, limit: int | None = None):
        self.source = isp_stream_file
        self.fh = open(isp_stream_file, "rb")

        self.fh.seek(0, os.SEEK_END)
        self._file_size = self.fh.tell()
        self.fh.seek(0, os.SEEK_SET)

        self.df = None
        self.isp = None
        self._limit = limit or 20 * 60 * 4000  # 20 min * prf=4000Hz

    def __del__(self):
        self.fh.close()

    def pop(self) -> BiomassSarIsp:
        """Method to extract an ISP from the opened file cursor position."""
        data = self.fh.read(PACKET_HEADER_LENGTH)

        packet_header = PacketHeader.frombytes(data)

        packet_len = packet_header.packet_data_field_length + 1

        packet = self.fh.read(packet_len)
        self.isp = GenericIsp(packet_header=packet_header, packet_data_field=packet)

        if int(self.isp.pid) < 25:
            error_msg = f"Unknown service type {self.isp.pid}"
            raise RuntimeError(error_msg)

        self.isp = BiomassSarIsp.from_generic_isp(self.isp)

        return self.isp

    def read(self):
        """Sequentially decode packet headers and store them into a pandas data-frame."""
        packet_counter = 0
        log = logging.getLogger(__name__)
        log.info(f'start decoding: "{self.source}"')

        records = []
        t0 = datetime.datetime.now()
        while (self.fh.tell() < self._file_size) and packet_counter < self._limit:
            pos = self.fh.tell()
            self.isp = self.pop()
            # Update counter and progress bar
            packet_counter += 1
            dd = self.isp.decode_header()
            dd["byte_offset"] = pos
            dd["datation"] = self.isp.datation
            records.append(dd)

        dt = datetime.datetime.now() - t0
        log.info(f"Decoding {packet_counter} ISPs in {dt.total_seconds()} [s]")
        log.info(f"Decoding speed {packet_counter / dt.total_seconds()} [isp/s]")

        self.df = pd.DataFrame(records)
