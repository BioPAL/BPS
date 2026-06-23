# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""BIOMASS ISP"""

from dataclasses import dataclass, field

import numpy as np
from generic_isp import GenericIsp, PacketHeader
from isp_structure import SYNC_MARKER, SarPacketDataField

PEC_LENGTH = 2  # bytes
DATA_FIELD_HEADER_LENGTH = 10  # bytes
SAR_HEADER_LENGTH = 76
F_USO = 37.826087
ASM_L = 13
DEC_FACTOR = 5


def mjd2000_to_datetime64(mjd2000, ref: np.datetime64 | None = None):
    """Convert elaspsed seconds since the reference to datetime64"""
    ref = ref or np.datetime64("2000-01-01T00:00:00")
    mjd2000 = np.atleast_1d(mjd2000)
    dt = (mjd2000 * 1e9).astype(np.int64)
    dt = dt.astype("timedelta64[ns]")
    dt = dt.squeeze()
    return ref + dt


@dataclass
class BiomassSarIsp:
    """Top class containing a BIOMASS SAR Source Packet

    |-------------------------------Source-Packet-------------------------------------|
    |--Packet-Header--|--------Packet-Data-Field--------------------------------------|
           (6)        |--data field header---|---------------source-data--------|-pec-|
                               (10)          |--SAR-header--|---user--data------|
                                                    (76)
    """

    packet_header: PacketHeader = field(default_factory=PacketHeader)
    packet_data_field: SarPacketDataField = field(default_factory=SarPacketDataField)

    @classmethod
    def from_generic_isp(cls, isp: GenericIsp):
        """This method extracts from a generic isp a Biomass SAR ISP."""
        packet_header = isp.packet_header
        user_data = isp.packet_data_field[DATA_FIELD_HEADER_LENGTH + SAR_HEADER_LENGTH : -PEC_LENGTH]

        packet_data_field = SarPacketDataField.frombytes(isp.packet_data_field)
        biomass_isp = cls(packet_header=packet_header, packet_data_field=packet_data_field)

        biomass_isp.user_data = user_data

        if biomass_isp.sync_marker != SYNC_MARKER:
            error_msg = f"Sync marker mismatch: expected {SYNC_MARKER}, got {biomass_isp.sync_marker}. "
            raise RuntimeError(error_msg)
        return biomass_isp

    @property
    def user_data(self):
        """Return the user data of the packet."""
        return self.packet_data_field.source_data.user_data

    @user_data.setter
    def user_data(self, ud):
        assert (
            len(ud)
            == self.packet_header.packet_data_field_length
            + 1
            - DATA_FIELD_HEADER_LENGTH
            - SAR_HEADER_LENGTH
            - PEC_LENGTH
        ), "Wrong User Data Length"
        self.packet_data_field.source_data.user_data = ud

    def decode_header(self):
        """Decode the header of the packet and return it as a dictionary."""
        r = self.packet_header.asdict()
        r.update(self.packet_data_field.data_field_header.asdict())
        r.update(self.packet_data_field.source_data.sar_header.asdict())
        return r

    @property
    def datation(self) -> float:
        """Perform the calculation of the ISP GPS time"""
        ct = self.packet_data_field.data_field_header.coarse_time
        ft = self.packet_data_field.data_field_header.fine_time << 8
        ft2 = self.packet_data_field.source_data.sar_header.datation_enhancement_service.fine_time_byte2
        ft += ft2
        # TODO: using 1e6 in the code below is to cope for an error with the data. to be removed later
        fuso = F_USO * 1e6
        ft_sec = ft * 4 / fuso
        return mjd2000_to_datetime64(ct + ft_sec)

    @property
    def sync_marker(self):
        """Return the synchronization marker of the packet."""
        return self.packet_data_field.source_data.sar_header.synchronization_service
