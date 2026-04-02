# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
PARC info parsing test
----------------------
"""

import unittest

import numpy.testing as npt
from arepytools.timing.precisedatetime import PreciseDateTime
from bps.l1_processor.parc.parc_info import ScatteringResponse
from bps.l1_processor.parc.parc_info_utils import parse_parc_info

PARC_INFO_XML = """<?xml version="1.0" encoding="UTF-8"?>
<auxiliaryCalSiteInformation xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="../support/bio-parc-info.xsd">
    <parcList count="1">
        <parc>
            <parcID>PARC_0001</parcID>
            <validityStart>2022-01-01T00:00:00.000000</validityStart>
            <validityStop>2099-12-31T00:00:00.000000</validityStop>
            <positionX>6223856.166886711</positionX>
            <positionY>-558608.5425139616</positionY>
            <positionZ>1273293.3350749211</positionZ>
            <delayList>
                <delayGt1 units="s">0.00000000000000000000000</delayGt1>
                <delayGt2 units="s">0.00001000000000000000082</delayGt2>
                <delayX units="s">0.00002000000000000000164</delayX>
                <delayY units="s">0.00003000000000000000076</delayY>
            </delayList>
            <rcsList>
                <rcsGt1 units="dB">71.0</rcsGt1>
                <rcsGt2 units="dB">72.0</rcsGt2>
                <rcsX units="dB">73.0</rcsX>
                <rcsY units="dB">74.0</rcsY>
            </rcsList>
        </parc>
    </parcList>
</auxiliaryCalSiteInformation>
"""


class PARCInfoParsing(unittest.TestCase):
    """Tests of aux ins content fill"""

    def test_parsing(self):
        """Parse string with parc info"""
        parc_info_list = parse_parc_info(PARC_INFO_XML)
        self.assertEqual(len(parc_info_list), 1)
        parc_info = parc_info_list[0]
        self.assertEqual(parc_info.parc_id, "PARC_0001")
        self.assertEqual(
            parc_info.validity_interval,
            (
                PreciseDateTime.from_numeric_datetime(2022, 1, 1),
                PreciseDateTime.from_numeric_datetime(2099, 12, 31),
            ),
        )
        npt.assert_array_equal(
            parc_info.position,
            [6223856.166886711, -558608.5425139616, 1273293.3350749211],
        )

        self.assertEqual(parc_info.delays[ScatteringResponse.GT1], 0.00000000000000000000000)
        self.assertEqual(parc_info.delays[ScatteringResponse.GT2], 0.00001000000000000000082)
        self.assertEqual(parc_info.delays[ScatteringResponse.X], 0.00002000000000000000164)
        self.assertEqual(parc_info.delays[ScatteringResponse.Y], 0.00003000000000000000076)

        self.assertEqual(parc_info.rcs[ScatteringResponse.GT1], 71.0)
        self.assertEqual(parc_info.rcs[ScatteringResponse.GT2], 72.0)
        self.assertEqual(parc_info.rcs[ScatteringResponse.X], 73.0)
        self.assertEqual(parc_info.rcs[ScatteringResponse.Y], 74.0)


if __name__ == "__main__":
    unittest.main()
