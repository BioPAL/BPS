# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l., DLR, Deimos Space
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

import itertools
from copy import deepcopy
from pathlib import Path
from time import time

import numpy as np
import scipy.optimize
import scipy.sparse as sp
from arepytools import io
from bps.common import bps_logger
from bps.l1_processor.autofocus.save_aresys import save_ph_acc_aresys, save_SLC_aresys
from bps.l1_processor.autofocus.signalproclib import Regular2DInterp, smooth
from scipy import constants as cons
from scipy import linalg
from scipy.constants import speed_of_light as c0
from scipy.interpolate import interp1d
from scipy.signal import detrend
from scipy.stats import entropy

R_EQ_EARTH = np.float64(6378137.0)  # WGS-84 Equatorial Radius
R_PL_EARTH = np.float64(6356752.31425)  # WGS-84 Polar Radius

re = cons.physical_constants["classical electron radius"][0]
me = cons.physical_constants["electron mass"][0]
qe = cons.physical_constants["elementary charge"][0]
e0 = cons.physical_constants["electric constant"][0]


# ===================================================================================================
# Generic functions
def LMSW_LU(ddy, sigmas2, LL, detrend_order=1):
    """
    Returns the Weighted Least Squares (WLS) estimation of a 2D matrix given repeated observations observations, accuracy and linear transformation.

    Parameters
    ----------
    ddy: ndarray
        Matrix of second derivatives. 3rd dimension is for repeated observations.
    sigmas2: ndarray
        Matrix of corresponding accuracies (3D)
    LL: ndarray
        Matrix that involves linear transformation.
    detrend: bool
        Flag if want to detrend solution. Default True.

    Returns
    -------
    yy: ndarray
        Results (2D)
    """
    Ny, Nx, _ = ddy.shape

    L_csr = sp.csr_matrix(LL)

    b = LL.T * (sigmas2.transpose(2, 0, 1).flatten())[np.newaxis, :]
    A = b @ L_csr
    lu, piv = linalg.lu_factor(A)

    # Estimate
    identity_matrix = np.eye((Nx * Ny))
    A_inv = linalg.lu_solve((lu, piv), identity_matrix)
    yy = (A_inv @ b @ (ddy.transpose(2, 0, 1)).flatten()).reshape(Ny, Nx)

    if detrend_order != 0:
        _, yy = detrend_2d(np.arange(Ny), yy, detrend_order)

    return yy, A_inv


def LMSW(ddy, sigmas2, LL, detrend=True):
    Ny, Nx, _ = ddy.shape

    b = LL.T * (sigmas2.transpose(2, 0, 1).flatten())[np.newaxis, :]

    A_inv = linalg.inv(b @ LL)

    yy = (A_inv @ b @ (ddy.transpose(2, 0, 1)).flatten()).reshape(Ny, Nx)

    if detrend:
        _, yy = detrend_2d(np.arange(Ny), yy, 2)

    return yy, A_inv


def detrend_2d(t, y, order):
    Na, Nr = y.shape

    fits = np.flipud(np.polynomial.polynomial.polyfit(t, y, order))

    trend = np.empty((Na, Nr), np.double)

    for xx in range(Nr):
        trend[:, xx] = np.polyval(fits[:, xx], t)

    return trend, y - trend


def interpolate_to_grid(data, r0in, t0in, r0out, t0out):
    drin = r0in[1] - r0in[0]
    dtin = t0in[1] - t0in[0]
    nrInt = ((r0out - r0in[0]) / drin)[None, :]
    naInt = ((t0out - t0in[0]) / dtin)[:, None]
    return Regular2DInterp(data, naInt, nrInt)  # Regular 2D interpolation


def calc_entropy(img):
    hist, _ = np.histogram(abs(img).flatten(), bins=100)
    prob_dist = hist / hist.sum()
    image_entropy = entropy(prob_dist, base=2)
    return image_entropy


# ===================================================================================================
# Faraday Rotation-related functions
def bickel_and_bates2(data_vv, data_vh, data_hv, data_hh, window=256, normalize=False):
    """coming from Paus script"""
    sym = data_hv - data_vh
    ant = data_hh + data_vv
    o12 = -sym + 1j * ant  # RL image
    o21 = sym + 1j * ant  # LR image
    fr = o12 * np.conj(o21)

    coh = (
        smooth(fr, window_len=11)
        / np.sqrt(smooth(abs(o12) ** 2, window_len=11))
        / np.sqrt(smooth(abs(o21) ** 2, window_len=11))
    )  # smooth = boxcar if side window_len

    coh[np.where(coh > 1)] = 1.0

    if normalize:
        # normalizing fr
        fr = np.exp(1j * np.angle(fr))

    # smoothing result and retrieving Faraday rotation
    if window > 0:
        fr = smooth(fr, window_len=window)  # smooth = boxcar if side window_len
    omega = np.angle(fr) / 4

    return omega, coh


def read_data(product: Path, channel: int = 1) -> np.ndarray:
    """Read data from a product folder and return it as a numpy array.

    Parameters
    ----------
    product : Path
        Path to the product folder.

    Returns
    -------
    np.ndarray
        Data read from the product folder.
    """
    pf = io.open_product_folder(product)
    meta = io.read_metadata(pf.get_channel_metadata(1))
    return io.read_raster_with_raster_info(pf.get_channel_data(1), meta.get_raster_info())


class ImportFaraday:
    def __init__(self, folder2operate, folder_bb, folder_bb_fullres, folder_plane, folder_bcos, folder_frcoh):
        pf_in = io.open_product_folder(folder2operate)
        channel = io.read_metadata(pf_in.get_channel_metadata(1))
        sc = channel.get_sampling_constants()
        ri = channel.get_raster_info()
        dsi = channel.get_dataset_info()

        # Extract Doppler rate vector for one channel
        drv = channel.get_doppler_rate()

        NaRaw = ri.lines
        NrSLC = ri.samples

        PRF = 1 / ri.lines_step
        RSF = 1 / ri.samples_step
        azBw = sc.baz_hz

        SA = synth_ap_samples(drv, NrSLC // 1 / RSF, PRF, azBw) - 1000

        NaSLCStart = int(SA // 2)
        NaSLCEnd = int(ri.lines - NaSLCStart)

        # First updample to NaRaw, NrSLC and then crop halp a chirp at both ends in azimuth
        aux1 = np.linspace(0, 1, num=NrSLC, endpoint=True)
        aux2 = np.linspace(0, 1, num=NaRaw, endpoint=True)

        # Imort Bcos from GPP
        bcos = read_data(folder_bcos)

        bcos = interpolate_to_grid(
            bcos,
            np.linspace(0, 1, num=bcos.shape[1], endpoint=True),
            np.linspace(0, 1, num=bcos.shape[0], endpoint=True),
            aux1,
            aux2,
        )[NaSLCStart:NaSLCEnd, :]

        fr2phase = (
            4 * np.pi * me * dsi.fc_hz / qe / (bcos * 1e-9)
        )  # Transform from 1-way fr imported from GPP to 2-way phase I need for autofocus

        # # This is if I want to use the multilooked version that I upsampled (i worked well at first)
        # self.data_ = fr2phase * omega

        # This is if I wanto to directly use the fullres phase from GPP
        self.data = read_data(folder_bb_fullres)  # [rad]

        # # This is in case I want to remove plane from total above.
        self.plane = read_data(folder_plane)[NaSLCStart:NaSLCEnd, :]
        self.plane_phase = read_data(folder_plane)[NaSLCStart:NaSLCEnd, :] * fr2phase

        # save_fr_ph_screen_aresys(folderFRPhaseScreen, self.data)
        self.data = self.data[NaSLCStart:NaSLCEnd, :]

        self.data = self.data - self.plane_phase

        # Import coherence matrix from GPP
        frcoh = read_data(folder_frcoh)
        # Calculate number of looks for calculation of coherence
        self.MLa = int(NaRaw / frcoh.shape[0] * azBw / sc.faz_hz)  # multilook in azimuth
        self.MLr = int(NrSLC / frcoh.shape[1] * sc.brg_hz / sc.frg_hz)  # Mutilook in range

        frcoh = interpolate_to_grid(
            frcoh,
            np.linspace(0, 1, num=frcoh.shape[1], endpoint=True),
            np.linspace(0, 1, num=frcoh.shape[0], endpoint=True),
            aux1,
            aux2,
        )[NaSLCStart:NaSLCEnd, :]

        self.sigma2_fr = 1 / 16 * (1 - frcoh**2) / (2 * frcoh**2 * (self.MLa * self.MLr)) * abs(fr2phase) ** 2


# ===================================================================================================
# Master Autofocus function
class masterAutofocus:
    def __init__(
        self,
        folder2operate,
        pols_use,
        folderPhaseScreen,
        folderOutputSLC,
        h_iono,
        delta_aa,
        delta_rr,
        olfa,
        olfr,
        Niter,
        returnCorrectedSLC=True,
        saveAllScreens=True,
        phase_fr=None,
        weighted=False,
        constrained=False,
    ):
        self.folder2operate = folder2operate
        self.pols_use = pols_use
        self.npols_use = len(self.pols_use)

        self.folderPhaseScreen = folderPhaseScreen
        self.folderOutputSLC = folderOutputSLC

        self.h_iono = h_iono

        # MDA block parameters
        self.delta_aa = delta_aa
        self.delta_rr = delta_rr
        self.olfa = olfa
        self.olfr = olfr
        self.Niter = Niter

        # Inputs for integration
        self.phase_fr = phase_fr  # Phase screen imported from Faraday Rotation

        # Scriptic parameters
        self.returnCorrectedSLC = returnCorrectedSLC
        self.saveAllScreens = saveAllScreens
        self.weighted = weighted
        self.constrained = constrained

    def run(self):
        self.__import_images_geometry__()

        self.__compress_at_hiono__()

        bps_logger.info("Autofocus - Initializing shift calculator...")
        self.shift_calculator = ShiftCalculator_C(
            self.centers, [self.delta_aa, self.delta_rr], coh_flag=0, mode=1, usf=1024, Ca=8, Cr=8, sldist=1
        )

        if self.phase_fr is not None:  # If phase_fr is available, resample it to the block centers
            self.__resampleFaraday__()

        self.integrator = Integrator2D(
            self.dt, self.N_az_blocks, self.N_rg_blocks, self.npols_use, self.weighted, self.constrained, self.phase_fr
        )
        self.__run_iterations__()

        if not self.abort:
            self.__save_results__()

        return

    def __resampleFaraday__(self):
        # Resample phase from FR and sigma2 to the block centers
        pha_fr_zp = np.zeros((self.Nax, self.Nrx), self.phase_fr.data.dtype)
        pha_fr_zp[
            self.delta_aa // 2 : self.NaSLC + self.delta_aa // 2, self.delta_rr // 2 : self.NrSLC + self.delta_rr // 2
        ] = self.phase_fr.data
        self.phase_fr.data = pha_fr_zp[self.iterables[0], :][:, self.iterables[1]]

        pha_fr_zp[
            self.delta_aa // 2 : self.NaSLC + self.delta_aa // 2, self.delta_rr // 2 : self.NrSLC + self.delta_rr // 2
        ] = self.phase_fr.sigma2_fr
        self.phase_fr.sigma2_fr = pha_fr_zp[self.iterables[0], :][:, self.iterables[1]]

        self.phase_fr.sigma2_fr[np.where(self.phase_fr.sigma2_fr == 0.0)] = np.max(self.phase_fr.sigma2_fr)

    def __save_results__(self):
        bps_logger.info("Autofocus - Saving results...")
        self.acc_int = np.sqrt(self.acc_int)

        save_ph_acc_aresys(
            self.folder2operate,
            self.folderPhaseScreen,
            np.pad(self.ph_int, [(self.NaSLCStart, self.NaSLCStart), (0, 0), (0, 0)], "edge"),
            np.pad(self.acc_int, [(self.NaSLCStart, self.NaSLCStart), (0, 0), (0, 0)], "edge"),
            self.saveAllScreens,
        )

        if self.returnCorrectedSLC:
            data_exp = np.zeros([self.NaRaw, self.NrSLC, self.nPol], np.complex64)

            data_exp[self.NaSLCStart : self.NaSLCEnd, :, :] = self.data_SLC_zp[
                self.delta_aa // 2 : self.delta_aa // 2 + self.NaSLC,
                self.delta_rr // 2 : self.delta_rr // 2 + self.NrSLC,
                :,
            ]

            save_SLC_aresys(self.folder2operate, self.folderOutputSLC, data_exp)

    def __run_iterations__(self):
        bps_logger.info("Autofocus - Image to be corrected: {}".format(self.folder2operate))

        self.abort = False

        if self.Niter < 0:  # then stop by itself based on entropy
            bps_logger.info(
                "Autofocus - You selected the AF to stop by itself there is no improve in entropy. Just take into account that AF can do many iterations before stopping."
            )
            entropy_prev = self.entr
            entropy_actual = self.entr - 0.001
            self.iteration = 0

            # Loop so that the algorithm stops iterating when entropy has reached its minimum
            shifts_a = None
            while entropy_actual < entropy_prev:
                bps_logger.info("Autofocus - Entered iteration {}".format(self.iteration + 1))
                entropy_prev = entropy_actual

                if self.iteration > 0:
                    shifts_a = self.__routine_phase_extract__(shifts_a)
                else:
                    shifts_a = self.__routine_phase_extract__()

                if not self.abort:
                    self.__apply_correction_2D_zp_iono__(self.ph_int[..., self.iteration])
                    entropy_actual = self.entr[-1]
                    self.iteration += 1
                else:
                    pass
            if not self.abort:
                bps_logger.info("Autofocus - Reached minimum entropy, now correcting again with second last phase...")

                if self.iteration == 1:
                    bps_logger.info("Autofocus - First iteration was already bad...")
                    self.__abort_af__()
                else:
                    self.__apply_correction_2D_zp_iono__(self.ph_int[..., self.iteration - 1])
            else:
                pass

        else:
            for self.iteration in range(self.Niter):
                if not self.abort:
                    bps_logger.info("Autofocus - Entered iteration {}".format(self.iteration + 1))

                    if self.iteration > 0:
                        shifts_a = self.__routine_phase_extract__(shifts_a)
                    else:
                        shifts_a = self.__routine_phase_extract__()

                    if not self.abort:
                        self.__apply_correction_2D_zp_iono__(self.ph_int[..., self.iteration])
                    else:
                        pass
                else:
                    self.__abort_af__()
            if not self.abort:
                self.iteration += 1
            else:
                pass

    def __init_comp_vars_corr__(self):
        bps_logger.info("Autofocus - Initializing match filter for hiono compression ...")
        hl = time()

        self.fa = get_Doppler_axis(self.PRF, self.NaRaw, self.dopplerCentroid)

        # Extract Doppler rate for every range position at the middle of the image
        Ndr = len(self.drv._poly_list)
        self.Kt_g, self.ve_g = drv2ve(
            self.drv, Ndr // 2, 1 / self.RSF * np.arange(self.NrSLC), self.wl, self.r0SLC
        )  # ground velocity in middle

        # Solve iono geometry and effective velocity
        self.rvec_iono = get_range_vector_iono(self.pSat, self.r0SLC, self.h_iono)
        # self.rvec_iono = rvec_iono_beeps # AQUI

        npoints = 10
        r_sub = self.rvec_iono[np.arange(npoints) * int(self.NrSLC / npoints)]

        ve_iono = get_effective_velocity_iono(self.pSat, self.vSat, self.ta, r_sub, self.h_iono, self.xyzRef, self.azBw)
        interpfunc = interp1d(r_sub, ve_iono, kind="cubic", bounds_error=False, fill_value="extrapolate")
        self.ve_iono = interpfunc(self.rvec_iono)

        self.phAC_dec_iono = (
            4
            * np.pi
            / self.wl
            * (self.rvec_iono[np.newaxis, :])
            * (np.sqrt(1 - (self.wl * self.fa[:, np.newaxis] / 2 / self.ve_iono[np.newaxis, :]) ** 2) - 1)
        ) + (
            4
            * np.pi
            / self.wl
            * (-self.r0SLC[np.newaxis, :])
            * (np.sqrt(1 - (self.wl * self.fa[:, np.newaxis] / 2 / self.ve_g[np.newaxis, :]) ** 2) - 1)
        ).astype("float32")
        self.phAC_dec_iono = np.cos(self.phAC_dec_iono) + 1j * np.sin(self.phAC_dec_iono)
        # Comment this for the momment to save time and memory, but it's a nice to have line
        # self.phAC_dec_ground = (4 * np.pi / self.wl * (self.r0SLC[np.newaxis, :]) * (np.sqrt(1 - (self.wl * self.fa[:, np.newaxis] / 2 / self.ve_g[np.newaxis, :])**2)-1)).astype('float32')

        bps_logger.info(
            "Autofocus - Initialization of filter for hiono compression lasted {:.02f} [s]".format(time() - hl)
        )

    def __apply_correction_2D_zp_iono__(self, ph_int, pad=True):
        hl = time()
        ph_ext = np.pad(ph_int, [(self.NaSLCStart, self.NaRaw - self.NaSLCEnd), (0, 0)], mode="edge")
        ph_ext = np.cos(ph_ext) - 1j * np.sin(ph_ext)

        # Apply phase screen correction
        dataOut = self.data_SLC_hiono * ph_ext[:, :, np.newaxis]

        for jj in range(self.nPol):
            dataOut[..., jj] = np.fft.fft(dataOut[..., jj], axis=0, norm="ortho")
            dataOut[..., jj] *= np.conj(self.phAC_dec_iono)
            dataOut[..., jj] = np.fft.ifft(dataOut[..., jj], axis=0, norm="ortho")

        self.entr = np.append(self.entr, calc_entropy(dataOut[..., 0]))
        bps_logger.info("Autofocus - Entropy {}: {:.05f}".format(self.iteration + 1, self.entr[-1]))

        # Crop SLC and pad it
        self.data_SLC_zp = np.zeros((self.Nax, self.Nrx, self.nPol), self.data_SLC_hiono.dtype)
        self.data_SLC_zp[
            self.delta_aa // 2 : self.NaSLC + self.delta_aa // 2,
            self.delta_rr // 2 : self.NrSLC + self.delta_rr // 2,
            :,
        ] = dataOut[self.NaSLCStart : self.NaSLCEnd, :, :]

        bps_logger.info("Autofocus - Applying correction  lasted: {:.02f} [s]".format(time() - hl))

    def __compress_at_hiono__(self):
        bps_logger.info("Autofocus - Entered compression at hiono...")
        hl = time()
        self.data_SLC_hiono = self.data_imp.copy()

        for jj in range(self.nPol):
            self.data_SLC_hiono[..., jj] = np.fft.fft(self.data_SLC_hiono[..., jj], axis=0, norm="ortho")
            self.data_SLC_hiono[..., jj] *= self.phAC_dec_iono
            self.data_SLC_hiono[..., jj] = np.fft.ifft(self.data_SLC_hiono[..., jj], axis=0, norm="ortho")

        bps_logger.info("Autofocus - Compression at hiono lasted {:.02f} [s]".format(time() - hl))

    def __import_images_geometry__(self):
        bps_logger.info("Autofocus - Entered module to import images and define geometry.")
        pf_in = io.open_product_folder(self.folder2operate)
        channel = io.read_metadata(pf_in.get_channel_metadata(1))
        sc = channel.get_sampling_constants()
        ri = channel.get_raster_info()
        bi = channel.get_burst_info()
        dsi = channel.get_dataset_info()
        svd = channel.get_state_vectors()

        # Iterate over all available channels and note the channel indexes for each polarizatio
        # pol_ch_idx = np.empty(pf_in.get_number_channels(), 'str')
        self.pol_ch_idx = []

        self.nPol = len(pf_in.get_channels_list())  # number of polarizations available

        for ii in pf_in.get_channels_list():
            self.pol_ch_idx.append(io.read_metadata(pf_in.get_channel_metadata(ii)).get_swath_info().polarization.name)

        # now compare witht hte self.pols_use vector and see what indexes out of the 4 avaiable are requested for the calculation
        self.idx_use = np.empty(self.npols_use, "int8")
        for ii in range(self.npols_use):
            if self.pols_use[ii].casefold() in self.pol_ch_idx:
                self.idx_use[ii] = self.pol_ch_idx.index(self.pols_use[ii].casefold())

        bps_logger.info("Autofocus - Polarizations to be used: {}".format(self.pols_use))

        # Extract Doppler rate vector for one channel
        self.drv = channel.get_doppler_rate(0)

        self.NaRaw = ri.lines
        self.NrSLC = ri.samples

        self.wl = c0 / dsi.fc_hz
        self.PRF = 1 / ri.lines_step
        self.RSF = 1 / ri.samples_step
        self.dopplerCentroid = 0.0  # change this: use channel.get_doppler_centroid()
        self.azBw = sc.baz_hz
        self.aosf = self.PRF / self.azBw
        self.trRawStart = bi._bursts[0].range_start_time
        self.taRawStart = bi._bursts[0].azimuth_start_time
        self.SA = (
            synth_ap_samples(self.drv, self.NrSLC // 1 / self.RSF, self.PRF, self.azBw) - 1000
        )  # Make it a bit larger just to be sure I cover the entire SLC area

        self.NaSLCStart = int(self.SA // 2)

        self.NaSLCEnd = int(ri.lines - self.NaSLCStart)
        self.NaSLC = self.NaSLCEnd - self.NaSLCStart

        _, _, self.xyzRef, _ = get_geometry_parameters(self.folder2operate)

        self.sha = self.delta_aa // self.olfa
        self.shr = self.delta_rr // self.olfr

        self.__cell_centers__()
        self.__calculate_dimensions_new__()

        self.t0SLC = np.arange(self.NaSLC) / self.PRF  # time vector for SLC starting from zero

        self.r0SLC = (np.arange(self.NrSLC) / self.RSF + self.trRawStart) * c0 / 2  # range vector to data_SLC

        self.pSat = svd.position_vector
        self.vSat = svd.velocity_vector
        self.ta = np.arange(self.pSat.shape[0]) * svd._dt_sv_s

        self.__init_comp_vars_corr__()

        # IMPORT IMAGE FROM GPP
        bps_logger.info("Autofocus - Importing images...")
        hl = time()
        self.data_imp = np.empty(
            [self.NaRaw, self.NrSLC, self.nPol], "complex64"
        )  # here use self.nPol since I'm loading all images to be corrected

        for jj in range(self.nPol):
            channel = pf_in.get_channel_data(jj + 1)
            meta = pf_in.get_channel_metadata(jj + 1)
            ri = io.read_metadata(meta).get_raster_info()
            self.data_imp[..., jj] = io.read_raster_with_raster_info(channel, ri)[: self.NaRaw, : self.NrSLC]
        data_SLC = self.data_imp[self.NaSLCStart : self.NaSLCEnd, :, :]

        bps_logger.info("Autofocus - Importing images lasted: {:.02f} [s]".format(time() - hl))
        self.entr = calc_entropy(data_SLC[..., 0])
        bps_logger.info("Autofocus - Entropy 0: {:.05f}".format(self.entr))

        # Zero -pad half a block in each direction, resize to acocmodate to appropriate MDA dimentions and calculate range and time vectors
        bps_logger.info("Autofocus - Resizing matrices...")

        hl = time()
        self.data_SLC_zp = np.zeros((self.Nax, self.Nrx, self.nPol), "complex64")
        self.data_SLC_zp[
            self.delta_aa // 2 : self.NaSLC + self.delta_aa // 2,
            self.delta_rr // 2 : self.NrSLC + self.delta_rr // 2,
            :,
        ] = data_SLC

        bps_logger.info("Autofocus - Resizing matrices lasted: {:.02f} [s]".format(time() - hl))

        # Calculate Doppler rate at hiono height at centers of blocks - save them into the data structures
        bps_logger.info("Autofocus - Calculating Doppler rate at centers of blocks at hiono...")
        hl = time()

        La = 11  # hardcoded

        Ta_iono = (self.r0SLC - self.rvec_iono) / (La * self.ve_g) * self.wl
        Ta = (self.r0SLC) / (La * self.ve_g) * self.wl

        Kt_use = -self.Kt_g * (Ta / Ta_iono)

        self.Kt_use = (
            np.ones((self.N_az_blocks, self.N_rg_blocks), np.float32)
            * np.pad(Kt_use, (self.delta_rr // 2, self.Nrx - (self.delta_rr // 2 + self.NrSLC)), mode="edge")[
                self.iterables[1]
            ][np.newaxis, :]
        )

        bps_logger.info(
            "Autofocus - Calculating Doppler rate at centers of blocks at hiono lasted: {:.02f} [s]".format(time() - hl)
        )

    def __routine_phase_extract__(self, shifts_prev=None):
        # Create space to save shifts from mapdrift
        shifts_a = np.empty((self.N_az_blocks, self.N_rg_blocks, self.npols_use), np.float64)
        # shifts_r = np.empty(shifts_a.shape, np.float32)

        # Calculate shifts
        for j, jj in enumerate(self.idx_use):
            bps_logger.info("Autofocus - Estimating shifts for polarizarion:{}".format(self.pols_use[j]))
            shifts_a[:, :, j], _ = self.shift_calculator.operate(img=self.data_SLC_zp[:, :, jj])

        shifts_a -= np.mean(shifts_a, axis=0)[np.newaxis, :, :]

        if self.iteration == 0:
            self.vmin_aux = np.percentile(shifts_a, 3)
            self.vmax_aux = np.percentile(shifts_a, 97)

        if shifts_prev is not None:
            shifts_a += shifts_prev

        # Intregrate phase screenph_int
        bps_logger.info("Autofocus - Doing LMS integration...")
        ph = self.__shifts2phase__(shifts_a)

        if self.iteration == 0:
            self.std_ph_flag = np.std(ph)
            bps_logger.info(
                "Autofocus - std of screen in iteration {}: {:.05f}".format(self.iteration + 1, self.std_ph_flag)
            )
        else:
            self.std_ph_flag = np.append(self.std_ph_flag, np.std(ph))
            if self.std_ph_flag[-1] > 1.5 * self.std_ph_flag[-2]:
                bps_logger.info("Autofocus - std of screen flag activated:")
                bps_logger.info(
                    "Autofocus - std of screen in iteration {}: {:.05f}".format(
                        self.iteration + 1, self.std_ph_flag[-1]
                    )
                )
                bps_logger.info(
                    "Autofocus - std of screen in iteration {}: {:.05f}".format(self.iteration, self.std_ph_flag[-2])
                )

                self.__abort_af__()

                self.abort = True

        if not self.abort:
            # Interpolate integated phase screen and thene xtrapolate it
            aux1 = np.linspace(0, ph.shape[1] - 1, num=len(self.rg_coord), endpoint=True)
            aux2 = np.linspace(0, ph.shape[0] - 1, num=len(self.t0SLC_), endpoint=True)

            ph_int = interpolate_to_grid(ph, np.arange(ph.shape[1]), np.arange(ph.shape[0]), aux1, aux2)

            if self.iteration == 0:
                self.ph_int = (ph_int[: self.NaSLC, : self.NrSLC])[..., np.newaxis]
            else:
                self.ph_int = np.append(self.ph_int, (ph_int[: self.NaSLC, : self.NrSLC])[..., np.newaxis], axis=2)

            try:
                self.ph_int[..., self.iteration] += self.phase_fr.plane_phase
            except Exception:
                pass

            # If self.acc exist then interpolate it to full resolution too
            try:
                acc = interpolate_to_grid(
                    self.acc[..., -1], np.arange(self.N_rg_blocks), np.arange(self.N_az_blocks), aux1, aux2
                )

                if self.iteration == 0:
                    self.acc_int = (acc[: self.NaSLC, : self.NrSLC])[..., np.newaxis]
                else:
                    self.acc_int = np.append(self.acc_int, (acc[: self.NaSLC, : self.NrSLC])[..., np.newaxis], axis=2)
            except AttributeError:
                pass

        return shifts_a

    def __shifts2phase__(self, shifts_a):
        """
        Transform from shift measure by MDA to phase
        :param shifts: [px]
        :return: phi(t)  [rad]
        """
        # Theres only a proportionality relation between the shifts and dba
        dba1 = 2 * self.Kt_use[:, :, np.newaxis] ** 2 / (self.azBw**2 * self.aosf) * shifts_a  # * 2

        ddph1 = 2 * np.pi * dba1.astype("float32")

        if self.phase_fr is None:
            if self.weighted:
                sigmas2 = np.empty(ddph1.shape, "float32")

                for ii in range(self.npols_use):
                    # Calculate sigmas matrix
                    aux = ddph1[..., ii]
                    sigmas2[..., ii] = (
                        smooth(aux**2, window_len=3) - smooth(aux, window_len=3) ** 2
                    )  # smooth = boxcar if side window_len
                sigmas2[np.where(sigmas2 < 0.0)] = np.max(sigmas2)

                # In the first iteration, calcualte the max sigma2 for plot
                if self.iteration == 0:
                    self.aux_sigmas2_vmax = np.percentile(sigmas2, 95)

                if self.constrained:
                    ph, acc = self.integrator.LMSW_cons(ddph1, 1 / sigmas2)
                else:
                    ph, acc = self.integrator.LMSW(ddph1, 1 / sigmas2)

                if self.iteration == 0:
                    self.acc = acc[..., np.newaxis]
                else:
                    self.acc = np.append(self.acc, acc[..., np.newaxis], axis=2)
            else:
                ph = self.integrator.integration_global2D(ddy=ddph1)

        else:
            if self.weighted:
                # Then create matrix of weights: up to nPol just with the ones sigmas2 of quadpol, last converted sigma in FR to sigma in phase from coherence
                sigmas2 = np.empty((ddph1.shape[0], ddph1.shape[1], self.npols_use + 1), "float32")

                for ii in range(self.npols_use):
                    sigmas2[..., ii] = (
                        smooth(ddph1[..., ii] ** 2, window_len=3) - smooth(ddph1[..., ii], window_len=3) ** 2
                    )  # smooth = boxcar if side window_len
                sigmas2[np.where(sigmas2 < 0.0)] = np.max(sigmas2)

                sigmas2[..., -1] = self.phase_fr.sigma2_fr

                ph, acc = self.integrator.LMSWF(ddph1, sigmas2)
                if self.iteration == 0:
                    self.acc = acc[..., np.newaxis]
                else:
                    self.acc = np.append(self.acc, acc[..., np.newaxis], axis=2)
            else:
                ph = self.integrator.integration_global2D_faraday(ddy=ddph1)

        return ph

    def __cell_centers__(self):
        """
        Return list of possible combinations for cell centers
        :return:
        """
        bps_logger.info("Autofocus - Calculating block centers....")
        # Account for zero padding of half a block i beforea nd after in each dimension
        self.Nax = self.NaSLC + self.delta_aa
        self.Nrx = self.NrSLC + self.delta_rr

        Mr = int((self.Nrx - self.delta_rr) / self.shr + 1)
        Ma = int((self.Nax - self.delta_aa) / self.sha + 1)

        # indeces of lis of iterables in each dimension
        ir = np.arange(Mr)
        ia = np.arange(Ma)

        # Allocate centers starting from the middle of the first block and then counting on shift between blocks
        cr = self.delta_rr // 2 + ir * self.shr
        ca = self.delta_aa // 2 + ia * self.sha

        # If there were some more columns not covered by the blocks created so far, create an extra line of centers
        if self.delta_rr + (Mr - 1) * self.shr < self.Nrx:
            cr = np.append(cr, int(cr[-1] + self.shr))
            self.Nrx = cr[-1] + self.delta_rr // 2

        # If there were some rows not covered by the blocks created so far, create an extra line of centers
        if self.delta_aa + (Ma - 1) * self.sha < self.Nax:
            ca = np.append(ca, int(ca[-1] + self.sha))
            self.Nax = ca[-1] + self.delta_aa // 2

        # Create a list of two elemets, with the iterables
        self.iterables = [ca, cr]

        # Iterate to find all possible combinations and create a grid of center coordiantes
        self.centers = sorted(list(itertools.product(*self.iterables)))

    def __calculate_dimensions_new__(self):
        bps_logger.info("Autofocus - Calculating dimensions of image for AF...")
        # Dimensions of new data_SLC, after having adjusted dimensions to that the centers are all in a regular grid
        self.NaSLC_ = self.Nax - self.delta_aa
        self.NrSLC_ = self.Nrx - self.delta_rr

        self.t0SLC_ = np.arange(self.NaSLC_ + 1) / self.PRF  # Azimuth time vector
        self.r0SLC_ = (np.arange(self.NrSLC_ + 1) / self.RSF + self.trRawStart) * c0 / 2  # Range to data_SLC_

        self.rg_coord = (
            np.arange(self.NrSLC_) + self.delta_rr // 2
        )  # range coordinates that enclose zp image in pixels not in meters

        self.NaSLCEnd_ = self.NaSLCStart + self.NaSLC_
        self.N_az_blocks = len(self.iterables[0])
        self.N_rg_blocks = len(self.iterables[1])

        self.dt = self.sha / self.PRF  # time inbetween centers of blocks in azimtuh

        # Create vector of ranges at centers of blocks
        self.r0vec = self.r0SLC_[self.N_rg_blocks - self.delta_rr // 2]

    def __abort_af__(self):
        bps_logger.info("Autofocus - Entered AF abort. AF will not be used.")

        # Log message to the user

        # Return zeros matrices for ph_int and acc

        # Overwite output image with input
        save_SLC_aresys(self.folder2operate, self.folderOutputSLC, self.data_imp)


class Integrator2D:
    def __init__(self, dt, Na_blocks, Nr_blocks, nPol, weighted=False, constrained=True, phase_fr=None):
        hl = time()

        bps_logger.info("Autofocus - Initializing integrator...")

        self.dt = dt
        self.Na_blocks = Na_blocks
        self.Nr_blocks = Nr_blocks
        self.NN = self.Na_blocks * self.Nr_blocks
        self.nPol = nPol

        self.phase_fr = phase_fr
        self.weighted = weighted
        self.constrained = constrained

        if self.weighted is False:
            if self.phase_fr is None:
                bps_logger.info("Autofocus - No phase input from FR, intialize simple LMS integation...")
                self.LMSG = self.__LMS_init__()
            else:
                self.LMSF = self.__LMSF_init()  # Initialize lms matrix with Faraday

        else:
            if self.phase_fr is None:
                bps_logger.info("Autofocus - LMSW without Faraday will be done")
                if self.constrained:
                    bps_logger.info("Autofocus - Initializing linear transformation matrix...")
                    L = (
                        (
                            -np.eye(self.NN)
                            + np.eye(self.NN, k=+self.Nr_blocks)
                            + np.eye(self.NN, k=-(self.NN - self.Nr_blocks))
                        )
                        / self.dt
                    ).astype("float32")
                    # L = ((-np.eye(self.NN) + np.eye(self.NN, k=+self.Nr_blocks)) / self.dt).astype('float32')
                    self.LL = np.append(
                        np.tile(L @ L, (self.nPol, 1)), (np.eye(self.NN) - np.eye(self.NN, k=+1)), axis=0
                    )
                else:
                    bps_logger.info("Autofocus - Initializing linear transformation matrix...")
                    L = (
                        (
                            -np.eye(self.NN)
                            + np.eye(self.NN, k=+self.Nr_blocks)
                            + np.eye(self.NN, k=-(self.NN - self.Nr_blocks))
                        )
                        / self.dt
                    ).astype("float32")
                    # L = ((-np.eye(self.NN) + np.eye(self.NN, k=+self.Nr_blocks)) / self.dt).astype('float32')
                    self.LL = np.tile(L @ L, (self.nPol, 1))
            else:
                bps_logger.info("Autofocus - LMSW with Faraday will be done...")

        bps_logger.info("Autofocus - Initilizing integrator lasted {:.02f} [s]".format(time() - hl))

    def LMSW(self, ddy, sigmas2):
        """
        Weighted.
        """

        hl = time()
        yy2, acc = LMSW_LU(ddy, sigmas2, self.LL, detrend_order=2)
        bps_logger.info("Autofocus - LMS LU integration with weights lasted: {:.02f} [s]".format(time() - hl))

        return yy2, np.diagonal(acc).reshape(self.Na_blocks, self.Nr_blocks)

    def LMSW_cons(self, ddy, sigmas2):
        """
        Weighted and constrained.
        """
        sigmas2 = np.append(sigmas2, np.mean(sigmas2, axis=2)[:, :, np.newaxis], axis=2)
        ddy = np.append(ddy, np.zeros((self.Na_blocks, self.Nr_blocks, 1), "float32"), axis=2)

        hl = time()
        yy2, acc = LMSW_LU(ddy, sigmas2, self.LL, detrend_order=2)

        bps_logger.info("Autofocus - LMS LU integration with weights lasted: {:.02f} [s]".format(time() - hl))

        return yy2, np.diagonal(acc).reshape(self.Na_blocks, self.Nr_blocks)

    def __LMS_init__(self):
        hl = time()
        L = (
            (-np.eye(self.NN) + np.eye(self.NN, k=+self.Nr_blocks) + np.eye(self.NN, k=-(self.NN - self.Nr_blocks)))
            / self.dt
        ).astype("float32")
        # L = ((-np.eye(self.NN) + np.eye(self.NN, k=+self.Nr_blocks)) / self.dt).astype('float32')

        # Do more efficient with diagoal matrix properties
        if self.constrained:
            bps_logger.info(
                "Autofocus - Range constrain applied... (this helps get nicer solutions that do not diverge)"
            )
            aux = np.eye(self.NN) - np.eye(self.NN, k=+1)  # append: range constrain
            L = np.append(np.tile(L @ L, (self.nPol, 1)), aux, axis=0)
        else:
            bps_logger.info("Autofocus - No range constrain applied...")
            L = np.tile(L @ L, (self.nPol, 1))

        lmsg = np.linalg.inv(L.T @ L) @ L.T

        bps_logger.info(
            "Autofocus - Initializing matrix for LMS integation in time domain lasted: {:.02f} [s]".format(time() - hl)
        )

        return lmsg

    def integration_global2D(self, ddy, detrend_order=1):
        """
        Global 2D integration
        Args:
            ddy (float32): _description_
            detrend_order (int, optional): order to be detrended from solution. Defaults to 1. Can also be 0 for no detrend and 1 to remove linear component.

        Returns:
            _type_: _description_
        """
        hl = time()

        if self.constrained:
            aux = np.zeros((self.Na_blocks, self.Nr_blocks, 1), "float32")
            datat = np.append(ddy, aux, axis=2).transpose(2, 0, 1)
        else:
            datat = ddy.transpose(2, 0, 1)

        yy = (self.LMSG @ datat.flatten()).reshape(self.Na_blocks, self.Nr_blocks)

        if detrend != 0:
            bps_logger.info("Autofocus - Removing integrated solution trends...")
            _, yy2 = detrend_2d(np.arange(self.Na_blocks) * self.dt, yy, detrend_order)

        bps_logger.info("Autofocus - LMS integration lasted: {:.02f} [s]".format(time() - hl))

        return yy2

    def __LMSF_init(self):
        hl = time()
        bps_logger.info("Autofocus - Entered LMS with Faraday")
        L = (
            (-np.eye(self.NN) + np.eye(self.NN, k=+self.Nr_blocks) + np.eye(self.NN, k=-(self.NN - self.Nr_blocks)))
            / self.dt
        ).astype("float32")

        # Do more efficient with diagoal matrix properties
        if self.constrained:
            bps_logger.info("Autofocus - Range constrain applied...")
            aux = np.empty((2 * self.NN, self.NN), "float32")  # append: ones of FR, range constrain
            aux[: self.NN, :] = np.eye(self.NN)
            aux[self.NN :, :] = np.eye(self.NN) - np.eye(self.NN, k=+1)
            L = np.append(np.tile(L @ L, (self.nPol, 1)), aux, axis=0)

        else:
            bps_logger.info("Autofocus - No range constrain applied...")
            L = np.append(np.tile(L @ L, (self.nPol, 1)), np.eye(self.NN) * 1, axis=0)  # append: ones of FR

        lmsf = np.linalg.inv(L.T @ L) @ L.T

        bps_logger.info(
            "Autofocus - Initializing matrix for LMS integation with Faraday in time domain lasted: {:.02f} [s]".format(
                time() - hl
            )
        )
        return lmsf

    def integration_global2D_faraday(self, ddy):
        hl = time()

        # Check that all this also works when I have one polarization only
        if self.constrained:
            aux = np.empty((self.Na_blocks, self.Nr_blocks, 2), "float32")
            aux[..., 0] = self.phase_fr.data
            aux[..., 1] = 0.0
            datat = np.append(ddy, aux, axis=2).transpose(2, 0, 1)

        else:
            datat = np.append(ddy, self.phase_fr.data[:, :, np.newaxis], axis=2).transpose(2, 0, 1)

        yy = (self.LMSF @ datat.flatten()).reshape(self.Na_blocks, self.Nr_blocks)

        bps_logger.info("Autofocus - LMS integration lasted: {:.02f} [s]".format(time() - hl))
        return yy

    def LMSWF(self, ddy, sigmas2):
        bps_logger.info("Autofocus - Entered Weighted least squares with Faraday...")
        # LMS WEIGHTED and with FARADAY
        hl = time()
        L = (
            (-np.eye(self.NN) + np.eye(self.NN, k=+self.Nr_blocks) + np.eye(self.NN, k=-(self.NN - self.Nr_blocks)))
            / self.dt
        ).astype("float32")

        # Do more efficient with diagonal matrix properties
        L = np.append(np.tile(L @ L, (self.nPol, 1)), np.eye(self.NN), axis=0)  # append: ones of FR

        # # # Append ones for range constrain at the begining and at the end
        # # cons = np.zeros((2*self.Nr_blocks, self.NN), 'float32')
        # # cons[:self.Nr_blocks, :self.Nr_blocks] = np.eye(self.Nr_blocks) - np.eye(self.Nr_blocks, k=+1)
        # # cons[self.Nr_blocks:, -self.Nr_blocks:] = np.eye(self.Nr_blocks) - np.eye(self.Nr_blocks, k=+1)
        # # L = np.append(L, cons, axis=0)

        W = np.diagflat(1 / sigmas2.transpose(2, 0, 1)).astype("float32")

        datat = np.append(ddy, self.phase_fr.data[:, :, np.newaxis], axis=2).transpose(2, 0, 1).flatten()
        # # datat = np.append(datat, np.zeros(auxW2.shape).flatten())

        aux = L.T * np.diagonal(W)[np.newaxis, :]
        acc = np.linalg.inv(aux @ L)
        lmsg = acc @ aux
        yy = (lmsg @ datat).reshape(self.Na_blocks, self.Nr_blocks)

        bps_logger.info("Autofocus - LMS integration with weights lasted: {:.02f} [s]".format(time() - hl))
        return yy, np.diagonal(acc).reshape(self.Na_blocks, self.Nr_blocks)


class ShiftCalculator_C:
    """
    Class containing routine for solving shift in image per blocks
    Upsampling method
    """

    def __init__(self, centers, block_dims, usf=1024, Ca=32, Cr=32, coh_flag=1, mode=0, axis=0, sldist=1):
        """
        :param centers: list of centers of blocks
        :param block_dims: [delta_a, delta_r] dimensions of block
        :param usf: up-sampling factor
        :param Ca: azimuth dimension foc cct crop
        :param Cr: range dimension of cct crop
        :param coh_flag: wanna calculate coherence between sub-looks?
        :param mode: (0) for square, (1) for abs, (2) for sqrt of sub-look
        :param axis: (0) for azimuth, (1) for range
        """
        self.centers = centers
        self.block_dims = block_dims
        self.usf = usf
        self.Ca = Ca
        self.Cr = Cr
        self.coh_flag = coh_flag
        self.mode = mode
        self.axis = axis
        if sldist == 1:
            self.filterList = self.__create_square_filters__(
                self.block_dims[0],
                self.block_dims[0] // 4,
                3 * self.block_dims[0] // 4,
                self.block_dims[0] // 2,
                self.block_dims[0] // 2,
            )
        elif sldist == 2:
            self.filterList = self.__create_square_filters__(
                self.block_dims[0],
                self.block_dims[0] // 8,
                7 * self.block_dims[0] // 8,
                self.block_dims[0] // 4,
                self.block_dims[0] // 4,
            )
        else:
            bps_logger.info("Autofocus - sldist invalid")

    def operate(self, img):
        """
        :param img: image to be operated
        :return: shifts_a, cohs
        """
        hl = time()

        if self.coh_flag:
            shifts_a, shifts_r, cohs = ImagePiecesSolver_ups(
                centers=self.centers,
                delta=self.block_dims,
                filterList=self.filterList,
                usf=self.usf,
                Ca=self.Ca,
                Cr=self.Cr,
                mode=self.mode,
                axis=self.axis,
            ).operate(img)

            bps_logger.info("Autofocus - Time elapsed calculating shifts: {:.02f} [s]".format(time() - hl))
            return shifts_a, shifts_r, cohs

        else:
            shifts_a, shifts_r, _ = ImagePiecesSolver_ups(
                centers=self.centers,
                delta=self.block_dims,
                filterList=self.filterList,
                usf=self.usf,
                Ca=self.Ca,
                Cr=self.Cr,
                mode=self.mode,
                axis=self.axis,
            ).operate(img)

            bps_logger.info("Autofocus - Time elapsed calculating shifts: {:.02f} [s]".format(time() - hl))
            return shifts_a, shifts_r

    def __create_square_filters__(self, N, fcent1, fcent2, B1, B2):
        """
        Create two binary non-overlapping filters. A high pass and a low pass filter, to separate image into two sub-looks.
        :param N: max frequency
        :param fcent1: central frequency filter 1
        :param fcent2: central frequency filter 2
        :param B1: bandwidth around fcent1
        :param B2: bandwidth around fcent2
        """
        # Create an array of zeros
        filter1 = np.zeros(N)
        filter2 = np.zeros(N)

        # Put ones where desired
        filter1[fcent1 - B1 // 2 : fcent1 + B1 // 2] = 1.0
        filter2[fcent2 - B2 // 2 : fcent2 + B2 // 2] = 1.0

        filterList = [] * 2
        filterList.append(filter1)
        filterList.append(filter2)

        return filterList


def extract_polynomial_coefficients(drv, index):
    t_ref_az = drv._poly_list[index].t_ref_az

    ss_ref_az = t_ref_az._seconds + t_ref_az._picoseconds * 1e-9

    t_ref_rg = drv._poly_list[index].t_ref_rg
    coefficients = drv._poly_list[index].coefficients

    coeff = np.append(coefficients[:2], coefficients[4:])[::-1]

    return t_ref_az, ss_ref_az, t_ref_rg, coeff


def drv2ve(drv, aidx, x2, wl, r0vec):
    _, ss_ref_az, _, coeff = extract_polynomial_coefficients(drv, aidx)

    p = np.poly1d(coeff)
    Kt = p(x2)

    ve = np.sqrt(-Kt * wl * r0vec / 2)

    return Kt, ve


def synth_ap_samples(drv, r0, PRF, azBw):
    Ndr = len(drv._poly_list)

    _, _, _, coeff = extract_polynomial_coefficients(drv, Ndr // 2)

    p = np.poly1d(coeff)
    Kt = abs(p(r0))

    return PRF**2 / abs(Kt)


def az_comp(img, rvec, wl, c0, ve, fa):
    phAC = 4 * np.pi / wl * rvec[np.newaxis, :] * np.sqrt(1 - (wl * fa[:, np.newaxis] / 2 / ve[np.newaxis, :]) ** 2)
    img *= np.exp(1j * phAC.astype("float32"))
    return img


def get_Doppler_axis(PRF, na, fdc):
    # Doppler centroid
    dfc = np.round(fdc * na / PRF) * PRF / na
    pfc = np.round((dfc % PRF) * na / PRF)
    # azimuth frequency vector
    fa = np.arange(na, dtype=np.float64) * PRF / na + dfc - PRF * 0.5
    fa = np.roll(fa, int(na * 0.5 + pfc))
    return fa


def get_effective_velocity_iono(pSat, vSat, ta, rScan, h_iono, xyzRef, azBw):
    lenOrbit = len(ta)
    xyz = getTargetInRange(
        pSat[int(lenOrbit / 2), :].flatten(),
        vSat[int(lenOrbit / 2), :].flatten(),
        xyzRef,
        0.0,  # zero-Doppler
        h_iono,
        rScan,
    )
    ve = ComputeEffectiveVelocityAnalytic(
        pSat,
        vSat,
        ta,
        xyz,
        1.0,  # not used in the function!
        0.0,  # zero-Doppler
        azBw,
    )
    return ve.flatten()


def get_range_vector_iono(pSat, rvec, h_iono):
    # getting ranges to iono height (spherical Earth approximation)
    h_sat = np.sqrt(np.sum(pSat[0, :] ** 2)) - R_EQ_EARTH
    look = sr2look(rvec, h_sat, R_EQ_EARTH)
    return look2sr(look, h_sat - h_iono, R_EQ_EARTH + h_iono)


def get_Doppler_rate_iono(RSF, rDelay, c0, pSat, vSat, ta, h_iono, xyzRef, azBw, wl, nr):
    # solving iono geometry (range vector and effective velocities)
    rvec_ground = (np.arange(int(nr)) / RSF + rDelay) * c0 / 2
    rvec_iono = get_range_vector_iono(pSat, rvec_ground, h_iono)

    npoints = 10  # number of points points only
    r_sub = rvec_iono[np.arange(npoints) * int(nr / npoints)]
    ve = get_effective_velocity_iono(pSat, vSat, ta, r_sub, h_iono, xyzRef, azBw)

    interpfunc = interp1d(r_sub, ve, kind="cubic", bounds_error=False, fill_value="extrapolate")
    ve = interpfunc(rvec_iono)

    return -2 * ve**2 / wl / rvec_iono


# ===================================================================================================
# Geometry-related functions
def Cart2Ellip(xyz, r_eq=R_EQ_EARTH, r_pl=R_PL_EARTH, indeg=False):
    # struct llh = [lon, lat, h], where lon and lat are in rad
    e_sq = (r_eq**2 - r_pl**2) / (r_eq**2)
    e_prima_sq = (r_eq**2 - r_pl**2) / (r_pl**2)

    N = int(np.round(np.size(xyz) / 3))
    sh = xyz.shape
    xyz = xyz.reshape([N, 3])
    s = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)

    theta = np.arctan2(xyz[:, 2] * r_eq, s * r_pl)
    llh = np.empty([N, 3], np.float64)
    llh[:, 0] = np.arctan2(xyz[:, 1], xyz[:, 0])
    llh[:, 1] = np.arctan2(xyz[:, 2] + e_prima_sq * r_pl * (np.sin(theta) ** 3), s - e_sq * r_eq * (np.cos(theta) ** 3))
    if indeg:
        llh[:, 0] *= 180.0 / np.pi
        llh[:, 1] *= 180.0 / np.pi

    v = r_eq / np.sqrt(1 - e_sq * (np.sin(llh[:, 1]) ** 2))
    llh[:, 2] = (s / np.cos(llh[:, 1])) - v
    xyz = xyz.reshape(sh)
    return llh


def geocodeTarget(pSat, vSat, p0, r0, wlfdc, h, leftLook=True, r_eq=R_EQ_EARTH, r_pl=R_PL_EARTH):
    if leftLook:
        factor = -1
    else:
        factor = 1

    def f(p):
        return [
            np.sqrt(np.sum((pSat - p) ** 2)) - r0,
            np.sum(factor * vSat * (p - pSat)) / (0.5 * r0) - 0.5 * wlfdc,
            Cart2Ellip(p, r_eq=r_eq, r_pl=r_pl)[0][2] - h,
        ]

    p = scipy.optimize.fsolve(f, p0, maxfev=10000, xtol=1.234e-9)
    e = np.empty(3, np.float64)
    e[2] = Cart2Ellip(p, r_eq=r_eq, r_pl=r_pl)[0][2] - h
    e[0] = 2.0 * np.sqrt(np.sum((pSat - p) ** 2)) - r0
    e[1] = np.sum(vSat * (p - pSat)) / (0.5 * r0) - 0.5 * wlfdc

    return p, e


def getTargetInRange(pSat, vSat, pref, wlfdc, h, rScan, verbose=True, r_eq=R_EQ_EARTH, r_pl=R_PL_EARTH):
    """Computes the coordinates of a set of targets at the same beam-center position and spread along range"""
    M = len(rScan)
    xyz = np.empty([M, 3], np.float64)
    # loop through ranges
    for ii in range(M):
        xyz[ii, :], _ = geocodeTarget(pSat, vSat, pref, rScan[ii], wlfdc, h, r_eq=R_EQ_EARTH, r_pl=R_PL_EARTH)

    return xyz


def solveQuadratic(a, b, c):
    aux = np.sqrt(b**2 - 4 * a * c)
    return (-b + aux) / 2.0 / a, (-b - aux) / 2.0 / a


def look2sr(lookAng, hSat, rPlanet):
    _, r0 = solveQuadratic(1.0, -2 * (rPlanet + hSat) * np.cos(lookAng), (2.0 * rPlanet + hSat) * hSat)
    return r0


def sr2look(r0, hSat, rPlanet):
    return np.arccos(((2.0 * rPlanet + hSat) * hSat + r0**2) / 2.0 / r0 / (rPlanet + hSat))


def ComputeEffectiveVelocityAnalytic(ptx, vtx, ta, xyz, wl, fdc, abw, refine=True):
    # Analytical approximation to be used with
    # monostatic acquisitions.

    # manipulates shapes
    M = len(xyz.shape)
    if M == 2:
        Mr, aux = xyz.shape
        xyzl = xyz.reshape([1, 1, Mr, 3])
    elif M == 3:
        Ma, Mr, aux = xyz.shape
        xyzl = xyz.reshape([1, Ma, Mr, 3])
    else:
        xyzl = xyz.reshape([1, 1, 1, 3])
    _, Ma, Mr, _ = xyzl.shape

    # get range histories
    Na, _ = ptx.shape
    rh = np.sqrt(np.sum((ptx.reshape([Na, 1, 1, 3]) - xyzl) ** 2, axis=3)) * 2
    # compute acceleration
    ac = np.gradient(vtx, (ta[1] - ta[0]), axis=0)

    # get effective velocity
    ve = np.empty([Ma, Mr], np.float64)
    for kk in range(Ma):
        rhl2 = rh[:, kk, :]

        for jj in range(Mr):
            rhl = rhl2[:, jj].flatten()

            # find position of minimum approach
            az = np.argmin(rhl)

            # only calculates Veff if minimum is not at the border
            if az > 10 and az < (len(ta) - 10):
                if not refine:
                    rdist = ptx[az, :] - xyzl[0, kk, jj, :]
                    vs_2 = np.sum(vtx[az, :] ** 2)
                    # get effective velocity    /.
                    ve[kk, jj] = np.sqrt(vs_2 + np.sum(rdist * ac[az, :]))
                else:
                    posAzp = 0.5 * (rhl[az - 1] - rhl[az + 1]) / (rhl[az - 1] + rhl[az + 1] - 2 * rhl[az])
                    az += posAzp
                    azint = int(az)
                    # interpolated distance vector
                    posa = ptx[azint, :]
                    posb = ptx[azint + 1, :]
                    ptxi = posa + (az - azint) * (posb - posa)
                    rdisti = ptxi - xyzl[0, kk, jj, :].flatten()
                    # interpolated velocity square
                    posa = vtx[azint, :]
                    posb = vtx[azint + 1, :]
                    vs_2i = np.sum((posa + (az - azint) * (posb - posa)) ** 2)
                    # interpolated aceleration
                    posa = ac[azint, :]
                    posb = ac[azint + 1, :]
                    aci = posa + (az - azint) * (posb - posa)
                    # get effective velocity
                    ve[kk, jj] = np.sqrt(vs_2i + np.sum(rdisti * aci))
            else:
                ve[kk, jj] = np.nan

    return ve


def get_los_vector(p, v, antennaTiltAngle, left_look):
    # velocity vector
    v_norm = v / np.sqrt(np.sum(v**2.0))
    # nadir vector
    pNadir = -p
    # projecting nadir vector onto Doppler plane to make sure it is
    # orthogonal to v
    pNadir = np.cross(v_norm, np.cross(pNadir, v_norm))
    pNadir /= np.sqrt(np.sum(pNadir**2.0))
    # rotating nadir vector around the Doppler plane using the velocity vector
    # (which is the normal vector of the Doppler plane) using Rodrigues'
    # rotation formula
    # (https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula)

    if left_look != 1:
        factor = -1
    else:
        factor = 1
    foo = factor * antennaTiltAngle
    losVector = (
        pNadir * np.cos(foo)
        + np.cross(v_norm, pNadir) * np.sin(foo)
        + v_norm * np.sum(v_norm * pNadir) * (1 - np.cos(foo))
    )
    losVector /= np.sqrt(np.sum(losVector**2.0))
    return losVector


def get_center_point(p, v, r, antennaTiltAngle, left_look):
    los_vector = get_los_vector(p, v, antennaTiltAngle, left_look)
    ref_pt = p + los_vector * r
    return ref_pt


def get_geometry_parameters(slc_input_dir):
    """This function retrieves some parameters from the SLC product needed to perform the computations"""
    pf_in = io.open_product_folder(slc_input_dir)
    channel = io.read_metadata(pf_in.get_channel_metadata(1))
    f0 = channel.get_dataset_info().fc_hz
    ri = channel.get_raster_info()
    nrg = ri.samples
    # reading orbit and obtaining position in the middle of the acquisition (position and velocity)
    antdir = channel.get_dataset_info().side_looking.value
    state_vectors = channel.get_state_vectors()
    position_vector = state_vectors.position_vector
    velocity_vector = state_vectors.velocity_vector
    # time_steps = state_vectors.time_step
    # reference_time_utc = state_vectors.reference_time
    nsamples, _ = position_vector.shape
    pSat = np.squeeze(position_vector[nsamples // 2, :])  # taking orbit in the middle of the vector
    vSat = np.squeeze(velocity_vector[nsamples // 2, :])  # taking orbit in the middle of the vector

    # reading roll angle (approx. antenna angle)
    antenna_angle = np.radians(abs(np.mean(channel.get_attitude_info().roll_vector)))

    # finding approximate scene center
    r = range_vector_aresys(ri)
    if antdir.upper() == "LEFT":
        left_look = 1
    elif antdir.upper() == "RIGHT":
        left_look = 0
    xyzRef = get_center_point(pSat, vSat, r[nrg // 2], antenna_angle, left_look)

    return pSat, vSat, xyzRef, f0


def range_vector_aresys(ri):
    """Generation of the range vector based on Aresys' raster info"""
    dr = ri.samples_step * c0 / 2
    return np.arange(ri.samples) * dr + ri.samples_start * c0 / 2


def calc_synt_ap_len(wl, La, ri, PRF, vsat):
    dr = ri.samples_step * c0 / 2

    r_far = ri.samples * dr + ri.samples_start * c0 / 2

    return (wl * r_far * PRF) / (La * vsat)


def getPad4fft(na, nr, Nsub=256):
    nazp = int(np.ceil(na / Nsub) * Nsub)
    nrzp = int(np.ceil(nr / Nsub) * Nsub)
    azOffset = int((nazp - na) / 2)
    rgOffset = int((nrzp - nr) / 2)

    return np.array([nazp, nrzp, azOffset, rgOffset])


def pad4fft(data, Nsub=256, centred=False, axis=1, zp_values=None):
    na, nr = data.shape
    if zp_values is None:
        nazp, nrzp, azOffset, rgOffset = getPad4fft(na, nr, Nsub=Nsub)
    else:
        nazp = zp_values[0]
        nrzp = zp_values[1]
        azOffset = zp_values[2]
        rgOffset = zp_values[3]

    if axis == 0:
        # padding in azimuth dimension
        if nazp == na:
            datazp = data
        else:
            datazp = np.zeros([nazp, nr], dtype=np.complex64)
            if centred:
                datazp[azOffset : azOffset + na, :] = data.copy()
            else:
                datazp[0:na, :] = data.copy()
    elif axis == 1:
        # padding in range dimension
        if nrzp == nr:
            datazp = data
        else:
            datazp = np.zeros([na, nrzp], dtype=np.complex64)
            if centred:
                datazp[:, rgOffset : rgOffset + nr] = data.copy()
            else:
                datazp[:, 0:nr] = data.copy()
    else:
        # padding in both dimensions
        if nazp == na and nrzp == nr:
            datazp = data
        else:
            datazp = np.zeros([nazp, nrzp], dtype=np.complex64)
            if centred:
                datazp[azOffset : azOffset + na, rgOffset : rgOffset + nr] = data.copy()
            else:
                datazp[0:na, 0:nr] = data.copy()

    return datazp, azOffset, rgOffset


def wls_plane_coeffs(num_blocks, trg, taz, phase, coher, n_coeff):
    nrg, naz = phase.shape
    num_meas = nrg * naz
    A = np.zeros((num_meas, n_coeff), dtype=float)
    B = np.zeros(num_meas, dtype=float)
    W = np.zeros((num_meas, num_meas), dtype=float)
    count = 0

    # building up system of equations
    for m in range(num_blocks[0]):
        for n in range(num_blocks[1]):
            if n_coeff == 6:
                A[count, :] = [1, trg[n], taz[m], trg[n] * trg[n], trg[n] * taz[m], taz[m] * taz[m]]
            else:
                A[count, :] = [1, trg[n], taz[m], trg[n] * taz[m]]
            B[count] = phase[m, n]
            W[count, count] = coher[m, n] ** 2
            count += 1

    # weighted least-squares
    Aaux = np.matmul(A.T, W)
    coeff = np.matmul(np.matmul(np.linalg.inv(np.matmul(Aaux, A)), Aaux), B)

    return coeff


def remove_plane(yy, n_coeff=6):
    Na_blocks, Nr_blocks = yy.shape

    rg = np.arange(Nr_blocks) / Nr_blocks
    az = np.arange(Na_blocks) / Na_blocks

    # n_coeff = 4
    offset = 0

    coeff = wls_plane_coeffs([Na_blocks, Nr_blocks], rg, az, yy, np.ones(yy.shape), n_coeff)

    if n_coeff == 6:
        a00 = coeff[0] + offset
        a01 = coeff[1]
        a10 = coeff[2]
        a02 = coeff[3]
        a11 = coeff[4]
        a20 = coeff[5]
    else:
        a00 = coeff[0] + offset
        a01 = coeff[1]
        a10 = coeff[2]
        a02 = 0
        a11 = coeff[3]
        a20 = 0

    az = az[:, np.newaxis]
    rg = rg[np.newaxis, :]
    plane = a20 * az * az + a11 * az * rg + a02 * rg * rg + a10 * az + a01 * rg + a00

    return plane, yy - plane


# ===================================================================================================
# Shift Calculation functions
class PieceShiftCalculator_ups:
    """
    Class collecting functions for calculating shift in a given piece, or whole image. Use up-sampling approach.
    """

    def __init__(self, filterList, mode, axis, usf=1024, Ca=32, Cr=32):
        """
        Initialize class
        :param filterList: [filter1, filter2]
        :param mode: (0) for square, (1) for abs, (2) for sqrt of sub-look
        :param axis: (0) for az, (1) for rg.
        :param usf: upsampling factor
        :param Ca:  azimuth dimension of cct crop
        :param Cr:  range dimension of cct crop
        """
        self.filterList = filterList
        self.mode = mode
        self.axis = axis
        self.usf = usf
        self.Ca = Ca
        self.Cr = Cr

        if (self.Ca % 2 != 0) or (self.Cr % 2 != 0):
            raise ValueError("Ca and Cr must be even numbers")

    def routine(self, img):
        """
        Carry out routine
        :param img: Piece / image.
        :return: sub-pixel locations of cct peak in az and in range + coherence between sub-looks.
        """

        ccf, cct, coh = self.cct_calc_rft_norm(img)

        shift = self.__dft_shift_log2__(cct)

        return shift[0], shift[1], coh

    def cct_calc_rft_norm(self, img):
        """
        Extract the result of some intermediate operations only (used for testing)
        :param img: imagpe of piece
        :return: normalzied ccf, cct, and coherence.
        """

        # Calculate spectrum of the image
        img = np.fft.fft(img, axis=self.axis)

        # Separate sub-looks
        SB = self.__separate_spectrum_rft__(img)

        ccf_norm, coh = self.__calculate_ccf_norm__(SB=SB)

        cct = np.fft.irfft2(ccf_norm, axes=(0, 1)) if self.axis == 0 else np.fft.irfft2(ccf_norm, axes=(1, 0))

        return ccf_norm, cct, coh

    def __separate_spectrum_rft__(self, img):
        """
        Separate spectrum into sub-looks
        :param azimuth fft of image
        :param filterList: list with two elements. Each element is one of the filters to be used.
        :return: a list of two elements, each of them is one sub-look.
        """
        Na, Nr = img.shape
        # Create list for return
        SB = [None] * len(self.filterList)

        # iterate for each filter
        for ii, filtro in enumerate(self.filterList):
            # Apply filter in frequency domain. Bring to time domain.

            if self.axis == 0:
                SB[ii] = np.fft.ifft(filtro.reshape([Na, 1]) * img, axis=0)
            else:
                SB[ii] = np.fft.ifft(filtro.reshape([1, Nr]) * img, axis=1)

            if self.mode == 0:
                # Expand spectrum
                SB[ii] = abs(SB[ii]) ** 2

            elif self.mode == 1:
                SB[ii] = abs(SB[ii])

            elif self.mode == 2:
                SB[ii] = np.sqrt(abs(SB[ii]))

            else:
                bps_logger.info("Autofocus - wrong mode")

            # Remove continuous component
            SB[ii] -= np.mean(SB[ii])

            # Pad with zeros in each dimension before and after the data with
            # Ndim/2 zeros so that the shape is now (Na*2 , Nr*2).
            SB[ii] = np.pad(SB[ii], ((int(Na / 2), int(Na / 2)), (int(Nr / 2), int(Nr / 2))), mode="constant")

            # Bring back to frequency domain
            if self.axis == 0:
                SB[ii] = np.fft.rfft2(SB[ii], axes=(0, 1))
            else:
                SB[ii] = np.fft.rfft2(SB[ii], axes=(1, 0))

        return SB

    def __calculate_ccf_norm__(self, SB):
        """
        Calculate normalized ccf and coherence.
        :param SB: sub-looks list
        :return: normalized ccf and coherence between sub-looks
        """

        aux = 1 / np.sqrt(np.mean(abs(SB[0]) ** 2) * np.mean(abs(SB[1]) ** 2))

        ccf_norm = (SB[0] * np.conj(SB[1])) * aux

        coh = abs(np.mean(SB[0] * np.conj(SB[1]))) * aux

        return ccf_norm, coh

    def __dft_shift_log2__(self, cct):
        """
        Routine of calculating location of cct peak
        :param cct: cct
        :return: locations in az and rg directions.
        """

        Na, Nr = cct.shape

        # get cross-co peak position and resulting shifts
        # Pixel accuracy
        maxima = np.unravel_index(np.argmax(np.abs(cct)), cct.shape)
        shifts_c = np.array(maxima)

        # account for cyclic nature of fft cross-co
        if shifts_c[0] > (Na // 2):
            shifts_c[0] -= Na
        if shifts_c[1] > (Nr // 2):
            shifts_c[1] -= Nr

        # inital sub-pixel shift
        shifts = np.array([0, 0])

        flag = True

        if (self.Ca > Na - 2 * abs(shifts_c[0]) - 1) or (self.Cr > Nr - 2 * abs(shifts_c[1]) - 1):
            shifts[0] = -shifts_c[0]
            shifts[1] = -shifts_c[1]

            flag = False

        if flag:
            # crop around cross-co peak and get spectrum
            crop = np.fft.ifftshift(cct)[
                Na // 2 + shifts_c[0] - self.Ca // 2 : Na // 2 + shifts_c[0] + self.Ca // 2,
                Nr // 2 + shifts_c[1] - self.Cr // 2 : Nr // 2 + shifts_c[1] + self.Cr // 2,
            ]

            ccf = np.fft.fft2(np.fft.ifftshift(crop))

            # perform log2 upsampling strategy
            # number of log2 upsampling interations
            N = int(np.ceil(np.log2(self.usf)))

            # inital sub-pixel shift
            shifts_aux = np.array([0, 0])

            # container for 1-D IDFT in y-dimension
            aux1d = np.empty((3, self.Ca), dtype=complex)

            # container for final 2-D IDFT
            cctUp = np.empty((3, 3), dtype=complex)

            for nn in range(N):
                offset_y = 1 - shifts_aux[1] * 2 ** (nn + 1)
                offset_x = 1 - shifts_aux[0] * 2 ** (nn + 1)

                aux_freq_r = np.fft.fftfreq(self.Cr, 2 ** (nn + 1))
                aux_freq_a = np.fft.fftfreq(self.Ca, 2 ** (nn + 1))

                # do matrix-multiply IDFT
                # y-dimension
                kernely1 = np.exp(-offset_y * aux_freq_r * 1j * 2 * np.pi)
                kernely2 = np.exp((1 - offset_y) * aux_freq_r * 1j * 2 * np.pi)
                kernely3 = np.exp((2 - offset_y) * aux_freq_r * 1j * 2 * np.pi)

                aux1d[0, :] = np.sum(ccf * kernely1, axis=-1)
                aux1d[1, :] = np.sum(ccf * kernely2, axis=-1)
                aux1d[2, :] = np.sum(ccf * kernely3, axis=-1)

                # x-dimension
                kernelx1 = np.exp(-offset_x * aux_freq_a * 1j * 2 * np.pi)
                kernelx2 = np.exp((1 - offset_x) * aux_freq_a * 1j * 2 * np.pi)
                kernelx3 = np.exp((2 - offset_x) * aux_freq_a * 1j * 2 * np.pi)

                cctUp[0, :] = np.sum(aux1d * kernelx1, axis=-1)
                cctUp[1, :] = np.sum(aux1d * kernelx2, axis=-1)
                cctUp[2, :] = np.sum(aux1d * kernelx3, axis=-1)

                # get upsampled peak position
                maximaUp = np.unravel_index(np.argmax(np.abs(cctUp)), cctUp.shape)

                shiftUp = np.array(maximaUp) - 1
                shiftSub = shiftUp / 2 ** (nn + 1)
                shifts_aux = shifts_aux + shiftSub

            shifts = -(shifts_c + shifts_aux)
        return shifts


class ImagePiecesSolver_ups:
    """
    Solve image block-wise using up-sampling approach.
    """

    def __init__(self, centers, delta, filterList, usf=1024, Ca=32, Cr=32, mode=0, axis=0):
        """
        Initilize class
        :param centers: list of block centers
        :param delta: [delta_a, delta_r]  dimensions of block
        :param filterList: [filter1, filter2]
        :param usf: up-sampling factor
        :param Ca: az dimension of crop in cct
        :param Cr: rg dimension of crop in cct
        :param mode: (0) for square, (1) for abs, (2) for sqrt of sub-look
        :param axis: (0) for azimuth, (1) for range
        """
        self.centers = centers

        iaz, irg = zip(*centers)
        self.iterables = (np.unique(iaz), np.unique(irg))

        self.da, self.dr = delta
        self.filterList = filterList
        self.mode = mode
        self.axis = axis

        self.usf = usf
        self.Ca = Ca
        self.Cr = Cr

    def operate(self, img):
        """
        Carry out operations
        :param img: image
        :return: sub-pixel location of cct peak in az and range + coherence between sub-looks
        """
        # Create array of zeros to save results
        Npieces = len(self.centers)
        shifts_a = np.zeros(Npieces)
        shifts_r = np.zeros(Npieces)
        cohs = np.zeros(Npieces)

        # Initialize object
        myRoutine = PieceShiftCalculator_ups(self.filterList, self.mode, self.axis, self.usf, self.Ca, self.Cr)

        # iterate over all coordinate pairs
        for ii, center in enumerate(self.centers):
            # Extract the piece given its center
            piece = self.__extract_piece__(center, img)

            # Run routine in each piece
            shifts_a[ii], shifts_r[ii], cohs[ii] = myRoutine.routine(piece)

        shifts_a = shifts_a.reshape(len(self.iterables[0]), len(self.iterables[1]))
        shifts_r = shifts_r.reshape(len(self.iterables[0]), len(self.iterables[1]))
        cohs = cohs.reshape(len(self.iterables[0]), len(self.iterables[1]))

        return shifts_a, shifts_r, cohs

    def __extract_piece__(self, coordinates, img):
        """
        Extract piece given the coordinates of it center and deepcopy it into another variable.
        :param coordinates: [px], [px]
        :return: piece
        """
        az = (np.arange(self.da) + (coordinates[0] - np.floor(self.da / 2))).astype(int)
        rg = (np.arange(self.dr) + (coordinates[1] - np.floor(self.dr / 2))).astype(int)

        return deepcopy(img[az, :][:, rg])
