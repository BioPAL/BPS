# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""ISP structure header"""

import enum
from dataclasses import dataclass, field

import bpack
import bpack.bs
from bpack import T

## Constants
# Synchronization marker fixed value
SYNC_MARKER = 0x352EF853


class BAQBLcode(enum.IntEnum):
    """BAQ Block Length Code"""

    COMPLEX_64 = 0  # 64 complex samples
    COMPLEX_128 = 1  # 128 complex samples


class GSTLcode(enum.IntEnum):
    """GSTL Code"""

    S1_INT = 0  # Round orbit for swath 1 on INT measurement
    S2_INT = 1  # Round orbit for swath 2 on INT measurement
    S3_INT = 2  # Round orbit for swath 3 on INT measurement
    S1_TOM = 3  # Round orbit for swath 1 on TOM measurement
    S2_TOM = 4  # Round orbit for swath 2 on TOM measurement
    S3_TOM = 5  # Round orbit for swath 3 on TOM measurement
    RO = 6  # Round orbit for receive only mode
    EXT_CAL = 16  # External calibration sequence
    SIMULATION = 47  # Short timeline for simulation mode
    # all the not booked codes 7-15 and 17-46 are free for future uses


class GpreCode(enum.IntEnum):
    """GPreamble Code"""

    ODD_RANK_PREAMBLE = 0  # Preamble sequence for odd rank used in measurement mode
    EVEN_RANK_PREAMBLE = 1  # Preamble sequence for even rank used in measurement mode
    # 2 and 3 are free for future use
    DUMMY2 = 2
    DUMMY3 = 3


class GposCode(enum.IntEnum):
    """GPostamble Code"""

    ODD_RANK_POSTAMBLE = 0  # Preamble sequence for odd rank used in measurement mode
    EVEN_RANK_POSTAMBLE = 1  # Preamble sequence for even rank used in measurement mode
    # 2 and 3 are free for future use
    DUMMY2 = 2
    DUMMY3 = 3


class GcalCode(enum.IntEnum):
    """GCalibration Code"""

    ODD_RANK_CALIBRATION = 0  # Interleaved calibration sequence for odd rank used in measurement mode
    EVEN_RANK_CALIBRATION = 1  # Interleaved calibration sequence for even rank used in measurement mode
    # 2 and 3 are free for future use
    DUMMY2 = 2
    DUMMY3 = 3


class BAQMODcode(enum.IntEnum):
    """BAQ Mode Code"""

    BAQ_4 = 0  # BAQ 4 bit
    BAQ_5 = 1  # BAQ 5 bit
    BAQ_6 = 2  # BAQ 6 bit
    BYPASS = 3  # BAQ bypassed
    # all other values are invalid


class STYPcode(enum.IntEnum):
    """STYP Code"""

    TXV_RX = 0  # Tx in V - Pol, Receive in V & H
    TXH_RX = 1  # Tx in H - Pol, Receive in V & H
    RX_ONLY = 2  # Receive Only in V & H
    TXV_ONLY = 3  # Transmit Only in V - Pol
    TXH_ONLY = 4  # Transmit Only in H - Pol
    TXCAL_V = 5  # Tx Calibration of PAS - V
    TXCAL_H = 6  # Tx Calibration of PAS - H
    TXCAL_V_1 = 7  # Tx Calibration of SSPA1 in PAS - V
    TXCAL_V_2 = 8  # Tx Calibration of SSPA2 in PAS - V
    TXCAL_H_1 = 9  # Tx Calibration of SSPA1 in PAS - H
    TXCAL_H_2 = 10  # Tx Calibration of SSPA2 in PAS - H
    RXCAL = 11  # Rx Calibration
    SHCAL = 12  # Short Loop Calibration in CDN
    IDLE = 13  # No Transmit and no Receive
    TXV_RXS = 14  # Same settings as TxV&Rx but needed in ICAL to identify stabilisation prior TxCal
    TXH_RXS = 15  # Same settings as TxH&Rx but needed in ICAL to identify stabilisation prior TxCal
    NOISE = 16  # Same settings as RxOnly but useful to clearly identify noise measurement
    # 17 - 31 not used


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
class DataFieldHeader:
    """Data Field Header of the BIOMASS SAR ISP"""

    filler_1: T["u1"] = bpack.field(default=0, repr=False)
    version_number: T["u3"] = 1  # FIXED 1 corresponds to the ECSS-E-70-41A
    filler_2: T["u4"] = bpack.field(default=0, repr=False)
    service_type: T["u8"] = 211  # FIXED 211 means SAR data management service
    service_subtype: T["u8"] = 1  # FIXED 1 means SAR data TM
    destination_id: T["u8"] = 0  # FIXED 0 means Ground
    coarse_time: T["u32"] = 0
    fine_time: T["u16"] = 0

    @property
    def size(self):
        """Return the size of the data field header in bytes."""
        return bpack.calcsize(self, bpack.EBaseUnits.BYTES)

    def asdict(self):
        """Return the data field header as a dictionary."""
        return bpack.asdict(self)


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
class OBTSyncStatus:
    """OBT Sync Status"""

    npe: T["u1"] = 0  # No PPS Error. 0 means no error
    nle: T["u1"] = 0  # No LOBT Error. 0 means no error
    mss: T["u1"] = 0  # Master Sync Status. 0 means not synchronized
    tty: T["u1"] = 0  # Time Type. 0 means no update since reset
    tmr: T["u1"] = 0  # Time Message Received. 0 means TM Message not received
    smt: T["u1"] = 0  # Sync Method. 0/1 means synchronized with SpW/PPS
    sst: T["u1"] = 0  # Sync Status. 0 means not synchronized
    sen: T["u1"] = 0  # Sync Enabled. 0 means disabled

    @property
    def size(self):
        """Return the size of the OBTSyncStatus in bytes."""
        return bpack.calcsize(self, bpack.EBaseUnits.BYTES)

    def asdict(self):
        """Return the OBTSyncStatus as a dictionary."""
        return bpack.asdict(self)


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
@dataclass
class DatationEnhancementService:
    """Datation Enhancement Service"""

    fine_time_byte2: T["u8"] = 0  # 3rd byte of fine time
    obt_sync_status: OBTSyncStatus = field(default_factory=OBTSyncStatus)

    @property
    def size(self):
        """Return the size of the datation enhancement service in bytes."""
        return bpack.calcsize(self, bpack.EBaseUnits.BYTES)

    def asdict(self):
        """Return the datation enhancement service as a dictionary."""
        d = {"fine_time_byte2": self.fine_time_byte2}
        d.update(self.obt_sync_status.asdict())
        return d


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
@dataclass
class IndexService:
    """Index Service"""

    sar_packet_count: T["u32"] = 0
    pri_count: T["u32"] = 0

    @property
    def size(self):
        """Return the size of the index service in bytes."""
        return bpack.calcsize(self, bpack.EBaseUnits.BYTES)

    def asdict(self):
        """Return the index service as a dictionary."""
        return bpack.asdict(self)


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
class DataTakeInformationService:
    """Data Take Information Service"""

    baq_block_length: BAQBLcode = bpack.field(size=1, default=BAQBLcode.COMPLEX_128)
    decimation_bypassed: bool = False  # 0/False means Decimation filtering performed
    instrument_configuration_id: T["u30"] = 0
    data_take_id: T["u32"] = 0  # TODO consolidate definition
    ground_user_annotation: T["u32"] = 0  # TODO consolidate definition
    gstl_index: GSTLcode = bpack.field(size=6, default=GSTLcode.S1_INT)
    warmup_flag: bool = False  # 0/False means no warmup sequence used
    preamble_enabled: bool = False  # 0/False means preamble sequence used
    # TODO check with Airbus since semantically it should be in the opposite way
    gpre_index: GpreCode = bpack.field(size=2, default=GpreCode.ODD_RANK_PREAMBLE)
    postamble_enabled: bool = False  # 0/False means postamble sequence used
    # TODO check with Airbus since semantically it should be in the opposite way
    gpos_index: GposCode = bpack.field(size=2, default=GposCode.ODD_RANK_POSTAMBLE)
    calibration_enabled: bool = False  # 0/False means interleaved calibration sequences used
    # TODO check with Airbus since semantically it should be in the opposite way
    gcal_index: GcalCode = bpack.field(size=2, default=GcalCode.ODD_RANK_CALIBRATION)

    @property
    def size(self):
        """Return the size of the data take information service in bytes."""
        return bpack.calcsize(self, bpack.EBaseUnits.BYTES)

    def asdict(self):
        """Return the data take information service as a dictionary."""
        return bpack.asdict(self)


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
class BITSELcode:
    """Bit Selection Code"""

    bitsel_length: T["u4"] = 0
    bitsel_start: T["u4"] = 0


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
class SSPAControlMessage:
    """SSPA Control Message"""

    iv_sign: T["u1"] = 0
    iv_code: T["u11"] = 0  # In-phase Voltage IV=iv_sign*5*iv_code/4096
    qv_sign: T["u1"] = 0
    qv_code: T["u11"] = 0  # Quadrature Voltage QV=qv_sign*5*qv_code/4096
    epcv_code: T["u6"] = 0  # auxiliary drain voltage EPCV=12*epcv_code/64
    tx_enable: bool = 0  # This flag is combined with the TxGate signal. It is set for Tx on.
    parity: T["u1"] = 0

    @property
    def size(self):
        """Return the size of the SSPA Control Message in bytes."""
        return bpack.calcsize(self, bpack.EBaseUnits.BYTES)


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
class SSPAMessageIndex:
    """SSPA Message Index"""

    sidx_fill: T["u3"] = 0  # unused bits
    sidx_code: T["u5"] = (
        0  # Each SSPA message index has a value range 0 - 31 and is located in the 5 lsbs of each byte.
    )

    @property
    def size(self):
        """Return the size of the SSPA Message Index in bytes."""
        return bpack.calcsize(self, bpack.EBaseUnits.BYTES)


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
@dataclass
class RealTimeParameterService:
    """Real Time Parameter Service"""

    pulse_repetition_interval: T["u16"] = 0
    sampling_window_position: T["u16"] = 0
    sampling_window_length: T["u16"] = 0
    bit_selection: BITSELcode = field(default_factory=BITSELcode)
    baq_mode: BAQMODcode = bpack.field(size=3, default=BAQMODcode.BAQ_4)
    signal_type: STYPcode = bpack.field(size=5, default=STYPcode.TXV_RX)
    tx_pulse_index: T["u6"] = 0  # TODO TXPIDXcode class definition after TBD resolution in SAR data ICD
    rank: T["u5"] = 0
    temperature_compensation: bool = False  # 0/False means PAS temperature compensation disabled
    spares_1: T["u4"] = bpack.field(default=0, repr=False)
    tx_pulse_length: T["u16"] = 0
    sspa_contol_message_v_n1: SSPAControlMessage = field(default_factory=SSPAControlMessage)
    sspa_contol_message_v_n2: SSPAControlMessage = field(default_factory=SSPAControlMessage)
    sspa_contol_message_v_r: SSPAControlMessage = field(default_factory=SSPAControlMessage)
    sspa_contol_message_h_n1: SSPAControlMessage = field(default_factory=SSPAControlMessage)
    sspa_contol_message_h_n2: SSPAControlMessage = field(default_factory=SSPAControlMessage)
    sspa_contol_message_h_r: SSPAControlMessage = field(default_factory=SSPAControlMessage)
    ras_control_message_v: T["u16"] = 0  # TODO RASControlMessage class definition after TBD resolution in SAR data ICD
    ras_control_message_h: T["u16"] = 0  # TODO RASControlMessage class definition after TBD resolution in SAR data ICD
    sspa_message_index_0: SSPAMessageIndex = field(default_factory=SSPAMessageIndex)
    sspa_message_index_1: SSPAMessageIndex = field(default_factory=SSPAMessageIndex)
    sspa_message_index_2: SSPAMessageIndex = field(default_factory=SSPAMessageIndex)
    sspa_message_index_3: SSPAMessageIndex = field(default_factory=SSPAMessageIndex)
    sspa_message_index_4: SSPAMessageIndex = field(default_factory=SSPAMessageIndex)
    sspa_message_index_5: SSPAMessageIndex = field(default_factory=SSPAMessageIndex)
    spares_2: T["u16"] = bpack.field(default=0, repr=False)

    @property
    def size(self):
        """Return the size of the real time parameter service in bytes."""
        return bpack.calcsize(self, bpack.EBaseUnits.BYTES)

    @property
    def pri(self):
        """Return the pulse repetition interval."""
        return self.pulse_repetition_interval

    @property
    def swp(self):
        """Return the sampling window position."""
        return self.sampling_window_position

    @property
    def swl(self):
        """Return the sampling window length."""
        return self.sampling_window_length

    def asdict(self):
        """Return the real time parameter service as a dictionary."""
        return {
            "pri": self.pulse_repetition_interval,
            "swp": self.sampling_window_position,
            "swl": self.sampling_window_length,
            "rank": self.rank,
            "bitsel_length": self.bit_selection.bitsel_length,
            "bitsel_start": self.bit_selection.bitsel_start,
            "baq_mode": self.baq_mode,
            "sigtyp": self.signal_type,
            "txpl_index": self.tx_pulse_index,
            "txpl": self.tx_pulse_length,
        }

    @pri.setter
    def pri(self, value):
        """Set the pulse repetition interval."""
        self.pulse_repetition_interval = value

    @swp.setter
    def swp(self, value):
        """Set the sampling window position."""
        self.sampling_window_position = value


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
@dataclass
class SarHeader:
    """BIOMASS SAR Header"""

    datation_enhancement_service: DatationEnhancementService = field(default_factory=DatationEnhancementService)
    synchronization_service: T["u32"] = SYNC_MARKER  # FIXED to 0x352EF853
    index_service: IndexService = field(default_factory=IndexService)
    data_take_information_service: DataTakeInformationService = field(default_factory=DataTakeInformationService)
    real_time_parameter_service: RealTimeParameterService = field(default_factory=RealTimeParameterService)

    @property
    def size(self):
        """Return the size of the SAR header in bytes."""
        return bpack.calcsize(self, bpack.EBaseUnits.BYTES)

    def asdict(self):
        """Return the SAR header as a dictionary."""
        d = self.datation_enhancement_service.asdict()
        d.update({"sync_marker": self.synchronization_service})
        d.update(self.index_service.asdict())
        d.update(self.data_take_information_service.asdict())
        d.update(self.real_time_parameter_service.asdict())
        return d


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
@dataclass
class PacketErrorCount:
    """Packet Error Count"""

    pec: T["u16"] = 0

    @property
    def size(self):
        """Return the size of the packet error count in bytes."""
        return bpack.calcsize(self, bpack.EBaseUnits.BYTES)


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
@dataclass
class SarSourceData:
    """dataclass for the source data field
                 source data field
    |-------------------------------Source-Packet-------------------------------------|
    |--Packet-Header--|--------Packet-Data-Field--------------------------------------|
           (6)        |--data field header---|---------------source-data--------|-pec-|
    >>>>>>>>                   (10)          |--SAR-header--|---user--data------|
                                                    (76)
    """

    sar_header: SarHeader = field(default_factory=SarHeader)

    def __post_init__(self):
        self.user_data = field(default=None, repr=False)


@bpack.bs.decoder
@bpack.descriptor(baseunits=bpack.EBaseUnits.BITS, byteorder=bpack.EByteOrder.BE)
@dataclass
class SarPacketDataField:
    """dataclass for the packet data field
     |-----------------Source-Packet---------------------------------------|
     |--Packet-Header--|--------Packet-Data-Field--------------------------|
    >>>>>>             |--data field header---|-------source-data----|-pec-|
    """

    data_field_header: DataFieldHeader = field(default_factory=DataFieldHeader)
    source_data: SarSourceData = field(default_factory=SarSourceData)

    def __post_init__(self):
        self.pec = field(default=None, repr=False)
