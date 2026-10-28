# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""Reader core for annotations."""

from pathlib import Path
from typing import Type, TypeVar

from bps.common import bps_logger
from bps.common.io.parsing import parse
from bps.transcoder import ANNOTATION_LEVEL, ANNOTATION_VERSION, LATEST_VERSION_STR, PFD_VERSION_MAPPING

AnnotationModel = TypeVar("AnnotationModel")


def reader(
    file: Path,
    level_tag: ANNOTATION_LEVEL,
    models: list[tuple[ANNOTATION_VERSION, Type[AnnotationModel]]],
) -> AnnotationModel:
    """Read main annotation of products

    The order of the models list matters: first models in the list will be tried before later ones.
    """
    content = file.read_text(encoding="utf-8")

    all_pfd_versions = {version: PFD_VERSION_MAPPING[version][level_tag] for version, _ in models}

    model: AnnotationModel | None = None
    for version, model_cls in models:
        latest_pfd_version = all_pfd_versions[LATEST_VERSION_STR]
        pfd_version = all_pfd_versions[version]
        try:
            model = parse(content, model_cls)
        except Exception as e:
            bps_logger.warning(f"Cannot parse {file} assuming PFD version {pfd_version} for {level_tag}: {e}")
            continue
        else:
            if version != LATEST_VERSION_STR:
                bps_logger.info(
                    f"File {file} was parsed assuming PFD version {pfd_version} for {level_tag}, but the latest version is {latest_pfd_version}",
                )
            break

    if model is None:
        raise ValueError(f"Failed to parse {file} assuming PFD versions {all_pfd_versions} for {level_tag}")

    return model
