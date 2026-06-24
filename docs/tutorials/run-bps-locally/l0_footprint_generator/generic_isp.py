# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""Generic ISP module"""

from dataclasses import dataclass, field
from typing import Any

import bpack
import bpack.bs
from bpack import T

PACKET_HEADER_LENGTH = 6


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
class PacketHeader:
    """Packet Header class for a generic ISP."""

    version: T["u3"] = 0  # FIXED 0 means space packet
    packet_type: T["u1"] = 0  # FIXED 0 means telemetry packet
    data_field_header_flag: bool = True
    pid: T["u7"] = 0
    pcat: T["u4"] = 0
    grouping_flags: T["u2"] = 0
    sequence_count: T["u14"] = 0
    packet_data_field_length: T["u16"] = 0

    @property
    def size(self):
        """Return the size of the packet header in bytes."""
        return bpack.calcsize(self, bpack.EBaseUnits.BYTES)

    @property
    def packet_lenght(self):
        """Return the total packet length."""
        return self.packet_data_field_length

    def asdict(self):
        """Return the packet header as a dictionary."""
        return bpack.asdict(self)


@dataclass
class GenericIsp:
    """Generic ISP
    |-----------------Source-Packet-----------------------|
    |--Packet-Header--|--------Packet-Data-Field----------|
    """

    packet_header: PacketHeader = field(default_factory=PacketHeader)
    packet_data_field: Any = field(default=None, repr=False)

    @property
    def pid(self):
        """Return the Packet Identifier (PID) of the packet header."""
        return self.packet_header.pid

    @pid.setter
    def pid(self, value):
        """Set the Packet Identifier (PID) of the packet header."""
        self.packet_header.pid = value

    @property
    def pcat(self):
        """Return the Packet Category (PCAT) of the packet header."""
        return self.packet_header.pcat

    @pcat.setter
    def pcat(self, value):
        """Set the Packet Category (PCAT) of the packet header."""
        self.packet_header.pcat = value

    @property
    def sequence_count(self):
        """Return the sequence count of the packet header."""
        return self.packet_header.sequence_count

    @sequence_count.setter
    def sequence_count(self, value):
        """Set the sequence count of the packet header."""
        self.packet_header.sequence_count = value % 2e14

    @property
    def packet_header_size(self):
        """Return the size of the packet header."""
        return self.packet_header.size

    @classmethod
    def frombytes(cls, data_bytes):
        """Create a GenericIsp from bytes."""
        data = data_bytes[0:PACKET_HEADER_LENGTH]
        packet_header = PacketHeader.frombytes(data)  # pylint: disable=E1101

        packet_len = packet_header.packet_data_field_length + 1
        packet = data_bytes[PACKET_HEADER_LENGTH : PACKET_HEADER_LENGTH + packet_len + 1]

        return cls(packet_header=packet_header, packet_data_field=packet)
