# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to read a ionex file
------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from arepytools.timing.precisedatetime import PreciseDateTime


def _flatten(nested_list: list[list[str]]) -> list[str]:
    return [elem for sub_list in nested_list for elem in sub_list]


def _split_maps(maps_lines: list[str], map_kind: str) -> list[list[str]]:
    output_maps = []
    end_tag = f"END OF {map_kind} MAP"
    assert maps_lines[-1].strip().endswith(end_tag)
    while maps_lines:
        end_of_map = find_first_line_index(maps_lines, end_tag)
        output_maps.append(maps_lines[0 : end_of_map + 1])
        maps_lines = maps_lines[end_of_map + 1 :]
    return output_maps


@dataclass
class IonexFileSections:
    """Ionex file sections"""

    header: list[str]
    tec_maps: list[list[str]]
    rms_maps: list[list[str]] | None
    footer: str

    @classmethod
    def from_content(cls, content: str) -> IonexFileSections:
        """Analyse content to split lines in different sections"""

        lines = content.splitlines()

        last_header_line_index = find_first_line_index(lines, "END OF HEADER")
        header = lines[0 : last_header_line_index + 1]

        values = lines[last_header_line_index + 1 :]

        first_rms_map_line = find_first_line_index_no_raise(values, "START OF RMS MAP")
        if first_rms_map_line is None:
            first_rms_map_line = find_first_line_index_no_raise(values, "END OF FILE")
            tec_maps = _split_maps(values[0:first_rms_map_line], "TEC")
            rms_maps = None
        else:
            tec_maps = _split_maps(values[0:first_rms_map_line], "TEC")
            rms_maps = _split_maps(values[first_rms_map_line:-1], "RMS")

        footer = values[-1]

        return cls(header, tec_maps, rms_maps, footer)

    def to_content(self) -> str:
        """Join all the sections back togheter"""
        if self.rms_maps is None:
            output_lines = self.header + _flatten(self.tec_maps) + [self.footer]
        else:
            output_lines = self.header + _flatten(self.tec_maps) + _flatten(self.rms_maps) + [self.footer]

        return "\n".join(output_lines)


def find_first_line_index(lines: list[str], content: str):
    """Find the first line index that contains content"""
    for line_index, line in enumerate(lines):
        if content in line:
            return line_index
    raise RuntimeError(f"None of the input lines contains: {content}")


def find_first_line_index_no_raise(lines: list[str], content: str) -> int | None:
    """Find the first line index that contains content"""
    for line_index, line in enumerate(lines):
        if content in line:
            return line_index
    return None


def find_last_line_from_content(lines: list[str], content: str):
    """Find the last line index that contains content"""
    for line_index, line in enumerate(reversed(lines)):
        if content in line:
            return len(lines) - 1 - line_index
    raise RuntimeError(f"None of the input lines contains: {content}")


def retrieve_time_from_line(line: str) -> PreciseDateTime:
    """Retrieve time from line"""
    time_tuple = list(map(int, line.split()[0:6]))
    return PreciseDateTime.from_numeric_datetime(*time_tuple)


def retrieve_first_map_epoch_time(lines) -> PreciseDateTime:
    """Retrieve first map time"""
    line_index = find_first_line_index(lines, "EPOCH OF FIRST MAP")
    return retrieve_time_from_line(lines[line_index])


def retrieve_last_map_epoch_time(lines) -> PreciseDateTime:
    """Retrieve lassts map time"""
    line_index = find_first_line_index(lines, "EPOCH OF LAST MAP")
    return retrieve_time_from_line(lines[line_index])
