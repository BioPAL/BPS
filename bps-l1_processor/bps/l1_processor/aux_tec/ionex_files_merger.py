# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Tools to merge two ionex files
------------------------------
"""

import copy
from pathlib import Path

from bps.l1_processor.aux_tec.ionex_files_reader import (
    IonexFileSections,
    find_first_line_index,
    retrieve_first_map_epoch_time,
    retrieve_last_map_epoch_time,
)


def _update_header(header: list[str], last_map_line: str, num_maps_line: str) -> list[str]:
    """Update header information"""
    header_out_tmp = copy.copy(header)
    # Update last map time
    last_map_line_index_1 = find_first_line_index(header, "LAST MAP")
    header_out_tmp[last_map_line_index_1] = last_map_line

    # Update number of maps
    num_maps_line_index_1 = find_first_line_index(header, "OF MAPS IN FILE")
    header_out_tmp[num_maps_line_index_1] = num_maps_line

    # Drop aux data instead of merging them
    start_aux_index = find_first_line_index(header, "START OF AUX DATA")
    end_aux_index = find_first_line_index(header, "END OF AUX DATA")
    header_out = header_out_tmp[:start_aux_index]
    header_out = header_out + header_out_tmp[end_aux_index + 1 :]

    return header_out


def _update_maps_indexes(map_list: list[list[str]], offset: int):
    """Update indexes"""
    out_map = copy.copy(map_list)
    for index, curr_map in enumerate(out_map):
        index_in = str(index + 1)
        index_out = str(offset + index + 1)
        curr_map[0] = curr_map[0].replace(index_in, index_out)
        curr_map[-1] = curr_map[-1].replace(index_in, index_out)
    return out_map


def _sort_content(
    ionex_files_sections: tuple[IonexFileSections, IonexFileSections],
) -> tuple[IonexFileSections, IonexFileSections]:
    "Reorder ionex files"

    ionex_1, ionex_2 = ionex_files_sections
    start_1 = retrieve_first_map_epoch_time(ionex_1.header)
    start_2 = retrieve_first_map_epoch_time(ionex_2.header)

    if start_1 > start_2:
        ionex_1, ionex_2 = ionex_2, ionex_1

    stop_1 = retrieve_last_map_epoch_time(ionex_1.header)
    start_2 = retrieve_first_map_epoch_time(ionex_2.header)

    if stop_1 != start_2:
        raise RuntimeError("Input ionex files are not consecutive: cannot merge them")

    return ionex_1, ionex_2


def merge_ionex_files_sections(ionex_1: IonexFileSections, ionex_2: IonexFileSections) -> IonexFileSections:
    """Merge ionex files sections"""

    ionex_1, ionex_2 = _sort_content((ionex_1, ionex_2))

    tec_maps_out = ionex_1.tec_maps[0:24] + _update_maps_indexes(ionex_2.tec_maps, len(ionex_1.tec_maps) - 1)

    rms_maps_out = None
    if ionex_1.rms_maps is not None and ionex_2.rms_maps is not None:
        rms_maps_out = ionex_1.rms_maps[0:24] + _update_maps_indexes(ionex_2.rms_maps, len(ionex_1.rms_maps) - 1)

    updated_last_map_line = ionex_2.header[find_first_line_index(ionex_2.header, "EPOCH OF LAST MAP")]

    original_num_maps = len(ionex_1.tec_maps)
    updated_num_maps = len(tec_maps_out)
    original_num_maps_line = ionex_1.header[find_first_line_index(ionex_1.header, "OF MAPS IN FILE")]
    updated_num_maps_line = original_num_maps_line.replace(str(original_num_maps), str(updated_num_maps))

    header_out = _update_header(ionex_1.header, updated_last_map_line, updated_num_maps_line)

    footer_out = ionex_1.footer

    return IonexFileSections(header_out, tec_maps_out, rms_maps_out, footer_out)


def merge_ionex_contents(content_1: str, content_2: str) -> str:
    """Merge ionex files"""

    ionex_1 = IonexFileSections.from_content(content_1)
    ionex_2 = IonexFileSections.from_content(content_2)
    ionex_out = merge_ionex_files_sections(ionex_1, ionex_2)
    return ionex_out.to_content()


def merge_ionex_files(ionex_files: tuple[Path, Path], merged_ionex_file: Path):
    """Merge ionex files"""
    ionex_1, ionex_2 = ionex_files
    content_1 = ionex_1.read_text(encoding="utf-8")
    content_2 = ionex_2.read_text(encoding="utf-8")

    merged_content = merge_ionex_contents(content_1, content_2)
    merged_ionex_file.write_text(merged_content, encoding="utf-8")
