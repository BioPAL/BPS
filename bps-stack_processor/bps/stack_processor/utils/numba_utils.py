# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Utilities to force numba to precompile code
-------------------------------------------
"""

from threading import Thread

import numpy as np
from bps.common import bps_logger
from bps.stack_cal_processor.core.azf.utils import colwise_roll
from bps.stack_cal_processor.core.azf.windowing import (
    hamming_window_bank,
    kaiser_window_bank,
    none_window_bank,
)
from bps.stack_cal_processor.core.floating_precision import EstimationDType
from bps.stack_cal_processor.core.skp.skpdecomposition import skp_processing
from bps.stack_cal_processor.core.utils import query_grid_mask
from bps.stack_processor.interface.external.aux_pps import (
    AuxiliaryStaprocessingParameters,
)


# NOTE: Numba requires some preliminary compilation to make sure we don't
# run into problems with multithreaded compilation.
def precompile_numba(aux_pps: AuxiliaryStaprocessingParameters) -> Thread:
    """
    Force numba to precompile code.

    Return
    ------
    Thread
        An handler to the thread that runs the compilation job.

    """
    bps_logger.info("Precompiling the numba code")
    thread = Thread(target=lambda: _precompilation_task(aux_pps))
    thread.start()
    return thread


def _precompilation_task(aux_pps: AuxiliaryStaprocessingParameters):
    """The numba precompilation task."""
    _precompile_numba_azf(
        EstimationDType.from_32bit_flag(use_32bit_flag=aux_pps.azimuth_spectral_filtering.use_32bit_flag),
    )
    _precompile_numba_skp(
        EstimationDType.from_32bit_flag(use_32bit_flag=aux_pps.skp_phase_calibration.use_32bit_flag),
    )
    _precompile_utils()


def _precompile_numba_azf(dtypes: EstimationDType):
    """Force numba to precompile the AZF utils."""
    for win_fn in [
        hamming_window_bank,
        kaiser_window_bank,
        none_window_bank,
    ]:
        win_fn(
            centers=np.ones(5, dtype=dtypes.float_dtype),
            frequency_bandwidths=np.ones(5, dtype=dtypes.float_dtype),
            sampling_frequency=1.0,
            window_param=0.5,
            nsamples=np.int32(2),
            inverse=False,
            dtype=dtypes.float_dtype,
        )
    colwise_roll(
        data=np.ones((5, 5), dtype=dtypes.float_dtype),
        shift=np.arange(5),
    )


def _precompile_numba_skp(dtypes: EstimationDType):
    """Force numba to precompile the SKP code."""
    mpmb_coh = np.zeros((9, 9, 61, 61), dtype=dtypes.complex_dtype)
    mpmb_coh[..., 0, 0] = np.identity(9, dtype=dtypes.complex_dtype)
    skp_processing(
        mpmb_coherences=mpmb_coh,
        subsampled_vertical_wavenumbers=np.zeros((3, 61, 61), dtype=dtypes.float_dtype),
        azm=0,
        rng=0,
        coreg_primary_image_index=0,
        num_images=3,
        num_polarizations=3,
        spectra_z=np.arange(-60.0, 60.0, 61, dtype=dtypes.float_dtype),
        volumetric_noise=np.random.rand(3, 3).astype(dtypes.complex_dtype),
        float_dtype=dtypes.float_dtype,
        complex_dtype=dtypes.complex_dtype,
    )


def _precompile_utils():
    """Force precompiling the utils code."""
    query_grid_mask(
        mask=np.random.rand(2, 2).astype(np.uint8),
        x_axis=np.arange(2, dtype=np.float64),
        y_axis=np.arange(2, dtype=np.float64),
        xs=np.arange(2, dtype=np.float64),
        ys=np.arange(2, dtype=np.float64),
        nodata_fill_value=0.5,
        nodata_value=np.nan,
    )
