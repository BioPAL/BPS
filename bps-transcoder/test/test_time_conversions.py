# SPDX-FileCopyrightText: 2026 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: Apache-2.0

import unittest

from arepytools.timing.precisedatetime import PreciseDateTime
from bps.transcoder.utils.time_conversions import (
    pdt_to_compact_date,
    pdt_to_compact_string,
    round_precise_datetime,
)


class TimeConversionsTest(unittest.TestCase):
    def test_pdt_to_compact_string(self):
        time = PreciseDateTime.from_sec85(1000)
        pdt_compact_string = pdt_to_compact_string(time)
        self.assertEqual(pdt_compact_string, "19850101T001640Z")

    def test_pdt_to_compact_date(self):
        time = PreciseDateTime.fromisoformat("2000-01-04T00:00:00Z")
        compact_date = pdt_to_compact_date(time)
        self.assertEqual(compact_date, "5K00")

    def test_tound_precise_datetime_ms(self):
        time = PreciseDateTime.from_utc_string("2000-01-04T00:00:00.1234567896")
        time_ms = round_precise_datetime(time, timespec="milliseconds")
        self.assertEqual(time.year, time_ms.year)
        self.assertEqual(time.month, time_ms.month)
        self.assertEqual(time.day_of_the_month, time_ms.day_of_the_month)
        self.assertEqual(time.hour_of_day, time_ms.hour_of_day)
        self.assertEqual(time.minute_of_hour, time_ms.minute_of_hour)
        self.assertEqual(time.second_of_minute, time_ms.second_of_minute)
        self.assertEqual(round(time_ms.picosecond_of_second), int(123 * 1e9))

    def test_tound_precise_datetime_us(self):
        time = PreciseDateTime.from_utc_string("2000-01-04T00:00:00.1234567896")
        time_us = round_precise_datetime(time, timespec="microseconds")
        self.assertEqual(time.year, time_us.year)
        self.assertEqual(time.month, time_us.month)
        self.assertEqual(time.day_of_the_month, time_us.day_of_the_month)
        self.assertEqual(time.hour_of_day, time_us.hour_of_day)
        self.assertEqual(time.minute_of_hour, time_us.minute_of_hour)
        self.assertEqual(time.second_of_minute, time_us.second_of_minute)
        self.assertEqual(round(time_us.picosecond_of_second), int(123457 * 1e6))

    def test_tound_precise_datetime_ns(self):
        time = PreciseDateTime.from_utc_string("2000-01-04T00:00:00.1234567892")
        time_ns = round_precise_datetime(time, timespec="nanoseconds")
        self.assertEqual(time.year, time_ns.year)
        self.assertEqual(time.month, time_ns.month)
        self.assertEqual(time.day_of_the_month, time_ns.day_of_the_month)
        self.assertEqual(time.hour_of_day, time_ns.hour_of_day)
        self.assertEqual(time.minute_of_hour, time_ns.minute_of_hour)
        self.assertEqual(time.second_of_minute, time_ns.second_of_minute)
        self.assertEqual(round(time_ns.picosecond_of_second), int(123456789 * 1e3))


if __name__ == "__main__":
    unittest.main()
