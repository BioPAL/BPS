# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
Chirp replica utilities functions
---------------------------------
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from arepytools.io import (
    iter_channels,
    metadata,
    open_product_folder,
    read_raster_with_raster_info,
)

FFT_LENGTH = 512
INTERPOLATION_FACTOR = 16
IRF_3DB_FACTOR = 0.885


@dataclass
class ReplicaParameters:
    """Chirp replica parameters"""

    polarization: str = "H/H"
    bandwidth: float = 0.0
    pslr: float = 0.0
    islr: float = 0.0
    location_error: float = 0.0
    validity_flag: bool = True


def load_replica(pf_path: Path) -> dict[str, (np.ndarray, metadata.Pulse)]:
    """Load replica"""
    replica_dict = {}
    if pf_path.exists():
        replica_pf = open_product_folder(pf_path)
        for channel_id, channel_metadata in iter_channels(replica_pf):
            replica = read_raster_with_raster_info(
                replica_pf.get_channel_data(channel_id), channel_metadata.get_raster_info()
            ).squeeze()
            polarization = channel_metadata.get_swath_info().polarization.value
            pulse = channel_metadata.get_pulse()
            replica_dict[polarization] = (replica, pulse)
    return replica_dict


def generate_ideal_chirp(pulse: metadata.Pulse) -> np.ndarray:
    """Generate ideal chirp"""
    chirp_samples = int(np.ceil(pulse.pulse_length * pulse.pulse_sampling_rate))
    chirp_axis = np.arange(chirp_samples) / pulse.pulse_sampling_rate
    chirp_sign = 1 if pulse.pulse_direction.value == "UP" else -1
    chirp = np.exp(
        1j
        * (
            chirp_sign * np.pi * pulse.bandwidth / pulse.pulse_length * chirp_axis**2
            + 2 * np.pi * pulse.pulse_start_frequency * chirp_axis
            + pulse.pulse_start_phase
        )
    ).squeeze()
    return chirp


def compress_replica(data_in: np.ndarray, compression_function: np.ndarray) -> np.ndarray:
    """Compress replica"""
    data_out = np.fft.fft(data_in, FFT_LENGTH) * np.conj(np.fft.fft(compression_function, FFT_LENGTH))
    data_out = np.fft.ifftshift(np.fft.ifft(data_out))
    data_out = data_out / np.max(np.abs(data_out))
    return data_out


def interpolate(data_in: np.ndarray) -> np.ndarray:
    """Interpolate signal"""
    data_out = np.fft.fftshift(np.fft.fft(data_in))
    data_out = np.concatenate((data_out, np.zeros((len(data_out) * (INTERPOLATION_FACTOR - 1),))), axis=0)
    data_out = np.fft.ifft(np.fft.ifftshift(data_out))
    data_out = data_out / np.max(np.abs(data_out))
    return data_out


def compute_resolution(profile: np.ndarray) -> float:
    """Compute resolution"""
    # Compute profile at -3dB
    profile_m3db = 10 * np.log10(np.abs(profile**2)) - 10 * np.log10(0.5)

    # Indexes of main lobe above 3db
    indexes = np.where(profile_m3db > 0)[0]
    if np.logical_or.reduce((indexes.size == 0, indexes[0] == 0, indexes[-1] == profile.size - 1)):
        return None

    # Peak width is where values are > maximum - 3dB
    dsx = 1 + profile_m3db[(indexes[0] - 1)] / (profile_m3db[indexes[0]] - profile_m3db[(indexes[0] - 1)])
    ddx = -profile_m3db[indexes[-1]] / (profile_m3db[indexes[-1] + 1] - profile_m3db[indexes[-1]])

    # Compute resolution
    resolution = (indexes.size - 1) + ddx + dsx

    return resolution


def convert_resolution_to_bandwidth(res: float, sampling_rate: float) -> float:
    """Convert resolution to bandwidth"""
    band = IRF_3DB_FACTOR / (res / INTERPOLATION_FACTOR / sampling_rate)
    return band


def generate_peak_mask(data: np.ndarray) -> np.ndarray:
    """Compute peak mask"""
    # Find peak position
    max_pos = np.abs(data).argmax()

    # Convert absolute of input data to dB and derive it
    def func(arg):
        return np.diff(20 * np.log10(np.abs(arg)))

    # Select all decreasing part of sinc lobes, for each direction
    id_dx = np.argwhere(func(data[:max_pos]) < 0).squeeze()
    id_sx = np.argwhere(func(data[max_pos:]) > 0).squeeze()
    if np.logical_or.reduce((id_dx.size == 0, id_sx.size == 0)):
        return np.ones_like(data)

    # Initialize mask
    mask = np.zeros_like(data)

    # Filling in mask with 1s where the main lobe lies
    # Find the first zeroes of the main lobe in all directions and filling inside them (towards the peak center)
    mask[id_dx[-1] + 1 : max_pos + id_sx[0] + 1] = 1

    return mask


def generate_rectangular_mask(axis: np.ndarray, size: float) -> np.ndarray:
    """Generate rectangular mask"""
    indexes_in_halfwindow = np.abs(axis) <= size / 2
    mask = np.zeros((axis.size,))
    mask[np.ix_(indexes_in_halfwindow)] = 1
    return mask


def generate_resolution_mask(axis: np.ndarray, res: float, multiplier: float) -> np.ndarray:
    """Generate resolution mask"""
    window = multiplier * res
    return generate_rectangular_mask(axis, window)


def pslr_masking(data: np.ndarray, res: float, peak_pos: int, number_res_cells: int = 10) -> np.ndarray:
    """PSLR masking"""
    # Compute pixel axes of the interpolated data
    data_axis = (np.arange(0, len(data)) - peak_pos) / INTERPOLATION_FACTOR

    # Compute data masks
    main_lobe_mask = generate_peak_mask(data)

    # Generate a bigger mask centered on the main lobe extending up to a given number of resolution cells in both
    # directions, i.e. selecting only a given number of side lobes when it is applied to input data
    oversized_peak_mask = generate_resolution_mask(axis=data_axis, res=res, multiplier=number_res_cells)

    # Subtract masks
    side_lobes_mask = oversized_peak_mask - main_lobe_mask

    return side_lobes_mask


def islr_masking(
    data: np.ndarray, res: float, peak_pos: int, number_res_cells: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """ISLR masking"""
    # Compute pixel axes of the interpolated data
    data_axis = (np.arange(0, len(data)) - peak_pos) / INTERPOLATION_FACTOR

    # Compute data masks
    main_lobe_mask = generate_peak_mask(data)

    # Generate strip mask covering side lobes from the first to the nth
    # nth is 20-dependent (20 resolution cells in that direction)
    side_lobes_mask = generate_resolution_mask(axis=data_axis, res=res, multiplier=number_res_cells)
    islr_mask = side_lobes_mask - main_lobe_mask
    islr_mask[(islr_mask < 0) | (islr_mask > 1)] = 1

    return main_lobe_mask, islr_mask


def compute_pslr(data: np.ndarray, res: float) -> float:
    """Compute PSLR"""
    # Compute data power
    data = np.abs(data) ** 2

    # Find data peak
    main_lobe_value = np.max(data)

    # Mask original data and evaluate the max side lobes value
    side_lobes_mask = pslr_masking(data=data, res=res, peak_pos=data.argmax())
    max_side_lobe_value = np.max(data * side_lobes_mask)

    # Compute PSLR
    pslr = 10 * np.log10(max_side_lobe_value / main_lobe_value)

    return pslr


def compute_islr(data: np.ndarray, res: float) -> float:
    """Compute ISLR"""
    # Compute data power
    data = np.abs(data) ** 2

    # Mask original data and evaluate integrals over main lobe and side lobes
    main_lobe_mask, islr_mask = islr_masking(data=data, res=res, peak_pos=data.argmax())
    main_lobe_energy = np.sum(np.abs(data * main_lobe_mask))
    side_lobes_energy = np.sum(np.abs(data * islr_mask))

    # Compute ISLR
    islr = 10 * np.log10(side_lobes_energy / main_lobe_energy)

    return islr


def parabolic_interpolation(array: np.ndarray) -> tuple[float, float]:
    """Parabolic peak interpolation"""
    alpha = array[0]  # before max
    beta = array[1]  # max
    gamma = array[2]  # after max
    peak_relative_position = (np.abs(alpha) - np.abs(gamma)) / (np.abs(alpha) - 2 * np.abs(beta) + np.abs(gamma)) / 2
    peak_value = beta - (alpha - gamma) * peak_relative_position / 4
    return peak_value, peak_relative_position


def compute_location_error(data: np.ndarray) -> float:
    """Compute location error"""
    # Compute data power
    data = np.abs(data) ** 2

    # Coarse peak estimation
    max_pos_coarse = np.min([np.max([2, data.argmax()]), len(data) - 1])

    # Interpolation around maximum coordinates with parabolic fitting around 3 points near maximum
    # to better estimate the the peak position (subpixel precision)
    _, delta_position = parabolic_interpolation(np.abs(data[max_pos_coarse - 1 : max_pos_coarse + 2]))

    # Final peak position
    location_error = (delta_position + max_pos_coarse - len(data) // 2) / INTERPOLATION_FACTOR

    return location_error


def analyse_chirp_replica(replica_path: Path) -> list[ReplicaParameters]:
    """Analyse chirp replica"""
    # If exists, load internal calibration replica
    replica_dict = load_replica(replica_path)

    # Loop on polarizations
    replica_parameters = []
    for key, value in replica_dict.items():
        replica = value[0]

        # Generate ideal chirp
        chirp = generate_ideal_chirp(value[1])

        # Cross-correlate replica and ideal chirp
        compressed_replica = compress_replica(replica, chirp)

        # Interpolate compressed replica
        compressed_replica = interpolate(compressed_replica)

        # Compute crossCorrelationBandwidth
        resolution = compute_resolution(compressed_replica)
        if resolution is None:
            replica_parameters.append(ReplicaParameters(key, 0, 0, 0, 0, False))
            continue
        bandwidth = convert_resolution_to_bandwidth(resolution, value[1].pulse_sampling_rate)

        # Compute crossCorrelationPslr
        pslr = compute_pslr(compressed_replica, resolution)

        # Compute crossCorrelationIslr
        islr = compute_islr(compressed_replica, resolution)

        # Compute crossCorrelationPeakLocation
        location_error = compute_location_error(compressed_replica)

        # Compute reconstructedReplicaValidFlag
        validity_flag = True

        replica_parameters.append(ReplicaParameters(key, bandwidth, pslr, islr, location_error, validity_flag))
    return replica_parameters
