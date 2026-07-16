# SPDX-FileCopyrightText: 2025 ARESYS - European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import unittest

import numpy as np
from bps.stack_cal_processor.core.skp.skpdecomposition import (
    fallback_skp_scattering_coherences,
)
from bps.stack_cal_processor.core.skp.utils import normalize_coherence


class SkpDecompositionTest(unittest.TestCase):
    def test_fallback_skp_scattering_coherence_pol3(self):
        np.random.seed(seed=1)

        nimg, npol = (3, 3)
        mpmb_coherence = np.random.rand(npol * nimg, npol * nimg)
        volume_noise = np.random.randn(nimg, nimg)
        dut = fallback_skp_scattering_coherences(
            mpmb_coherence,
            volume_noise,
            npol,
            nimg,
            np.complex128,
        )

        self.assertTupleEqual(dut.shape, (4, nimg, nimg))
        for i in range(npol):
            np.testing.assert_almost_equal(
                dut[i],
                normalize_coherence(
                    mpmb_coherence[
                        i * nimg : (i + 1) * nimg,
                        i * nimg : (i + 1) * nimg,
                    ],
                    np.complex128,
                ),
            )
        np.testing.assert_almost_equal(dut[3], volume_noise)

    def test_fallback_skp_scattering_coherence_pol4(self):
        np.random.seed(seed=2)

        nimg, npol = (3, 4)
        mpmb_coherence = np.random.rand(npol * nimg, npol * nimg)
        volume_noise = np.random.randn(nimg, nimg)
        dut = fallback_skp_scattering_coherences(
            mpmb_coherence,
            volume_noise,
            nimg,
            npol,
            np.complex128,
        )

        self.assertTupleEqual(dut.shape, (4, nimg, nimg))
        np.testing.assert_almost_equal(
            dut[0],
            normalize_coherence(
                mpmb_coherence[0:nimg, 0:nimg],
                np.complex128,
            ),
        )
        np.testing.assert_almost_equal(
            dut[1],
            normalize_coherence(
                (
                    mpmb_coherence[nimg : 2 * nimg :, nimg : 2 * nimg]
                    + mpmb_coherence[2 * nimg : 3 * nimg :, 2 * nimg : 3 * nimg]
                )
                / 2,
                np.complex128,
            ),
        )
        np.testing.assert_almost_equal(
            dut[2],
            normalize_coherence(
                mpmb_coherence[-nimg:, -nimg:],
                np.complex128,
            ),
        )
        np.testing.assert_almost_equal(dut[3], volume_noise)


if __name__ == "__main__":
    unittest.main()
