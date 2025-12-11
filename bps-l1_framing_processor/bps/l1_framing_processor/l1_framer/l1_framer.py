# Project: BIOMASS Processing Suite (BPS)
#
# Copyright (c) 2025, ARESYS S.r.l.
# Developed under contract with the European Space Agency (ESA)
#
# SPDX-License-Identifier: MIT

"""
L1 Framer module
"""

import enum

import numpy as np
from bps.l1_framing_processor.orbit.eocfiorbit import EOCFIOrbit
from bps.l1_framing_processor.utils.constants import (
    FRAME_ANGLE_GRID_SPACING,
    FRAME_OVERLAP,
    L0_SENSING_TIME_MARGIN,
    L0_THEORETICAL_TIME_MARGIN,
    L1_PROCESSING_MARGIN,
    MINIMUM_FRAME_LENGTH,
    NOMINAL_FRAME_LENGTH,
    NOMINAL_FRAME_LENGTH_MARGIN,
    NUMBER_OF_FRAMES_PER_ORBIT,
    SECONDS_PER_DAY,
    SLICE_OVERLAP,
)
from bps.l1_framing_processor.utils.time_conversions import to_datetime64, to_mjd2000
from scipy.interpolate import interp1d


class DataLossPosition(enum.Enum):
    """DataLossPosition class

    Parameters
    ----------
    enum : enum.Enum
        Generic enumeration class
    """

    RIGHT = "RIGHT"
    LEFT = "LEFT"
    NONE = "NONE"
    NOT_SET = "NOT_SET"


class DataTakePosition(enum.Enum):
    """DataTakePosition class

    Parameters
    ----------
    enum : enum.Enum
        Generic enumeration class
    """

    FIRST = "FIRST"
    LAST = "LAST"
    CENTRAL = "CENTRAL"
    UNIQUE = "UNIQUE"
    NOT_SET = "NOT_SET"


class FrameStatus(enum.Enum):
    """FrameStatus class

    Parameters
    ----------
    enum : enum.Enum
        Generic enumeration class
    """

    NOMINAL = "NOMINAL"
    MERGED = "MERGED"
    PARTIAL = "PARTIAL"
    INCOMPLETE = "INCOMPLETE"
    NOT_FRAMED = "NOT_FRAMED"
    NOT_SET = "NOT_SET"


class Frame:
    """Frame class"""

    def __init__(
        self,
        index,
        start_time,
        stop_time,
        start_angle,
        stop_angle,
    ):
        """Initialise Frame object

        Parameters
        ----------
        index : int
            Frame index
        start_time : np.datetime64
            Frame start time
        stop_time : np.datetime64
            Frame stop time
        start_angle : float
            Frame start OPS angle
        stop_angle : float
            Frame stop OPS angle
        """
        self.index = index
        self.start_time = start_time
        self.stop_time = stop_time
        self.start_angle = start_angle
        self.stop_angle = stop_angle

        self.duration = self.__compute_duration()

        self.dt_position = DataTakePosition.NOT_SET
        self.dl_position = DataLossPosition.NOT_SET
        self.status = FrameStatus.NOT_SET

    def __compute_duration(self):
        """Compute frame duration

        Returns
        -------
        float
            Frame duration
        """
        return (self.stop_time - self.start_time) / np.timedelta64(1, "s")

    def overlaps_with_interval(self, start_time, stop_time):
        """Check if frame overlaps with a given time interval

        Parameters
        ----------
        start_time : np.datetime64
            Time interval start time
        stop_time : np.datetime64
            Time interval stop time

        Returns
        -------
        bool
            True if overlaps, False otherwise
        """
        return (self.start_time <= start_time <= self.stop_time) or (start_time <= self.start_time <= stop_time)

    def update_frame(self, start_time=None, stop_time=None, start_angle=None, stop_angle=None):
        """Update frame attributes

        Parameters
        ----------
        start_time : np.datetime64, optional
            New frame start time, by default None
        stop_time : np.datetime64, optional
            New frame stop time, by default None
        start_angle : float, optional
            New frame start OPS angle, by default None
        stop_angle : float, optional
            New frame stop OPS angle, by default None

        Returns
        -------
        bool
            Status (True for success, False for unsuccess)
        """
        if (start_time is not None) and (start_angle is not None):
            self.start_time = start_time
            self.start_angle = start_angle

        if (stop_time is not None) and (stop_angle is not None):
            self.stop_time = stop_time
            self.stop_angle = stop_angle

        self.duration = self.__compute_duration()

        return True

    def set_dt_position(self, position):
        """Set frame position in the data-take

        Parameters
        ----------
        position : DataTakePosition
            New frame position

        Returns
        -------
        bool
            Status (True for success, False for unsuccess)
        """
        self.dt_position = position
        return True

    def set_dl_position(self, position):
        """Set data-loss position (if any) w.r.t. frame

        Parameters
        ----------
        position : DataLossPosition
            New data-loss position

        Returns
        -------
        bool
            Status (True for success, False for unsuccess)
        """
        self.dl_position = position
        return True

    def set_status(self, status):
        """Set frame status

        Parameters
        ----------
        status : FrameStatus
            New frame status

        Returns
        -------
        bool
            Status (True for success, False for unsuccess)
        """
        self.status = status
        return True

    def __str__(self):
        """Get frame string representation

        Returns
        -------
        str
            Frame string representation
        """
        frame_as_string = (
            f"Frame: {self.index}   "
            + f"Interval: {np.datetime_as_string(self.start_time, unit='us')} : {np.datetime_as_string(self.stop_time, unit='us')} "
            + f"({self.start_angle:.3f} : {self.stop_angle:.3f})   "
            + f"Duration: {self.duration:.3f}   "
            + f"Position DT\\DL: {self.dt_position.value}\\{self.dl_position.value}   "
            + f"Status: {self.status.value}"
        )
        return frame_as_string


class L1Framer:
    """L1Framer class"""

    def __init__(
        self,
        orbit_path: str,
        dt_start_time: np.datetime64,
        dt_stop_time: np.datetime64,
    ):
        """Initialise L1Framer object

        Parameters
        ----------
        orbit_path : str
            Path to orbit file
        dt_start_time : np.datetime64
            Data-take start time
        dt_stop_time : np.datetime64
            Data-take stop time
        """
        self.orbit_path = orbit_path
        self.dt_start_time = dt_start_time
        self.dt_stop_time = dt_stop_time

        self._orbit = EOCFIOrbit(self.orbit_path)

        self.frames_list = []

        self.status = self.__compute_time2angle_lut()

    def __compute_time2angle_lut(self):
        """Compute time-to-OPS angle LUT

        Returns
        -------
        bool
            Status (True for success, False for unsuccess)
        """
        time_axis = np.arange(
            to_mjd2000(self.dt_start_time) / SECONDS_PER_DAY,
            (to_mjd2000(self.dt_stop_time) + 1) / SECONDS_PER_DAY,
            1 / SECONDS_PER_DAY,
        )
        time_axis = np.asarray([t for t in time_axis if self._orbit.start_time < t < self._orbit.stop_time])

        pos, vel, acc = self._orbit.get_pos_vel_acc(time_axis)
        angle_axis = self._orbit.get_ops_angle(time_axis, pos, vel, acc)

        self.anx_crossing_flag = True if angle_axis[-1] < angle_axis[0] else False

        angle_axis = np.unwrap(angle_axis, period=360)

        self._time2angle = interp1d(time_axis, angle_axis, bounds_error=False, fill_value="extrapolate")
        self._angle2time = interp1d(angle_axis, time_axis, bounds_error=False, fill_value="extrapolate")

        return True

    def get_frames_in_datatake(self):
        """Compute L1 frames contained in input data-take

        Returns
        -------
        List of Frame objects
            List of frames
        """
        # Compute indexes of frames bracketing the data-take
        # Note that indexes go from 1 to NUMBER_OF_FRAMES_PER_ORBIT
        frame_indexes = (
            np.arange(
                int(self._angle2time.x[0] // FRAME_ANGLE_GRID_SPACING),
                int(self._angle2time.x[-1] // FRAME_ANGLE_GRID_SPACING) + 1,
            )
        ) % NUMBER_OF_FRAMES_PER_ORBIT + 1

        # Compute frames start and stop OPS angles
        frame_start_angle_wrapped = (frame_indexes - 1) * FRAME_ANGLE_GRID_SPACING
        frame_stop_angle_wrapped = frame_start_angle_wrapped + FRAME_ANGLE_GRID_SPACING

        # Compute frames start and stop times
        frame_start_time = np.atleast_1d(
            to_datetime64(self._angle2time(np.unwrap(frame_start_angle_wrapped, period=360)) * SECONDS_PER_DAY)
        )
        frame_stop_time = np.atleast_1d(
            to_datetime64(
                self._angle2time(np.unwrap(frame_stop_angle_wrapped, period=360)) * SECONDS_PER_DAY + FRAME_OVERLAP
            )
        )

        # Clip frames start and stop times to data-take ones
        frame_start_time = np.atleast_1d(
            np.clip(
                frame_start_time,
                np.datetime64(self.dt_start_time),
                np.datetime64(self.dt_stop_time),
            )
        )
        frame_stop_time = np.atleast_1d(
            np.clip(
                frame_stop_time,
                np.datetime64(self.dt_start_time),
                np.datetime64(self.dt_stop_time),
            )
        )

        # Store frames list
        for index, start_time, stop_time, start_angle, stop_angle in zip(
            frame_indexes,
            frame_start_time,
            frame_stop_time,
            frame_start_angle_wrapped,
            frame_stop_angle_wrapped,
        ):
            self.frames_list.append(Frame(index, start_time, stop_time, start_angle, stop_angle))

        return self.frames_list

    def get_frames_in_slice(
        self,
        slice_start_time,
        slice_stop_time,
        slice_sensing_start_time,
        slice_sensing_stop_time,
        short_datatake_flag=False,
        merge_short_frames_flag=False,
        add_l1processing_margins_flag=True,
    ):
        """Compute L1 frames contained in input L0 slice

        Parameters
        ----------
        slice_start_time : np.datetime64
            L0 slice validity (or theoretical) start time
        slice_stop_time : np.datetime64
            L0 slice validity (or theoretical) stop time
        slice_sensing_start_time : np.datetime64
            L0 slice sensing start time
        slice_sensing_stop_time : np.datetime64
            L0 slice sensing stop time
        short_datatake_flag : bool, optional
            Datatake length lower than minimum one,
            by default False
        merge_short_frames_flag : bool, optional
            Merge short L1 frames at the beginning and/or at the end of the L0 slice and of the data-take,
            by default False
        add_l1processing_margins_flag : bool, optional
            Add L1 processing margins (left and right) to computed L1 frames,
            by default True

        Returns
        -------
        List of Frame objects
            List of frames
        """
        if not self.frames_list:
            self.get_frames_in_datatake()

        # Check overlap between L0 slice and orbit
        # - if exceeding threshold, return error
        if (
            to_mjd2000(slice_sensing_start_time) + L0_SENSING_TIME_MARGIN
        ) / SECONDS_PER_DAY < self._orbit.start_time or (
            to_mjd2000(slice_sensing_stop_time) - L0_SENSING_TIME_MARGIN
        ) / SECONDS_PER_DAY > self._orbit.stop_time:
            return None
        # - if not exceeding threshold, limit to orbit validity
        slice_orbit_stop_time_difference = slice_sensing_stop_time - to_datetime64(
            self._orbit.stop_time * SECONDS_PER_DAY
        )
        if slice_orbit_stop_time_difference > 0:
            slice_stop_time -= slice_orbit_stop_time_difference
            slice_sensing_stop_time -= slice_orbit_stop_time_difference

        # Select L1 frames overlapping with L0 slice
        frames_in_slice_list = [
            frame
            for frame in self.frames_list
            if frame.overlaps_with_interval(slice_sensing_start_time, slice_sensing_stop_time)
        ]
        frames_in_slice = len(frames_in_slice_list)

        # L0 slice
        # - Position in the data-take
        # NOTE: keep a margin to manage potential L0M/L0S sensing start/stop times discrepancies
        delta = np.timedelta64(L0_SENSING_TIME_MARGIN, "s")
        if (
            np.abs(slice_sensing_start_time - self.dt_start_time) < delta
            and np.abs(slice_sensing_stop_time - self.dt_stop_time) < delta
        ):
            slice_dt_position = DataTakePosition.UNIQUE
        elif np.abs(slice_sensing_start_time - self.dt_start_time) < delta:
            slice_dt_position = DataTakePosition.FIRST
        elif np.abs(slice_sensing_stop_time - self.dt_stop_time) < delta:
            slice_dt_position = DataTakePosition.LAST
        else:
            slice_dt_position = DataTakePosition.CENTRAL
        # - Data-loss position (if any) w.r.t. slice
        # NOTE: keep a margin to manage potential sensing/validity times discrepancies (for central frames)
        delta = np.timedelta64(L0_THEORETICAL_TIME_MARGIN, "ms")
        if (slice_sensing_start_time - slice_start_time > delta) and (
            slice_dt_position not in [DataTakePosition.FIRST, DataTakePosition.UNIQUE]
        ):
            slice_dl_position = DataLossPosition.LEFT
        elif (slice_sensing_stop_time - slice_stop_time < -delta) and (
            slice_dt_position not in [DataTakePosition.LAST, DataTakePosition.UNIQUE]
        ):
            slice_dl_position = DataLossPosition.RIGHT
        else:
            slice_dl_position = DataLossPosition.NONE

        # Loop on L1 frames
        indexes_to_discard = []
        for index, frame in enumerate(frames_in_slice_list):
            if frame.status == FrameStatus.MERGED:
                continue
            # Set attributes
            # - Start and stop times: cut first and last L1 frames according to L0 slice boundaries
            if index == 0:
                slice_start_angle = self._time2angle(to_mjd2000(slice_sensing_start_time) / SECONDS_PER_DAY).item()
                frame.update_frame(start_time=slice_sensing_start_time, start_angle=slice_start_angle)
            elif index == frames_in_slice - 1:
                slice_stop_angle = self._time2angle(to_mjd2000(slice_sensing_stop_time) / SECONDS_PER_DAY).item()
                frame.update_frame(stop_time=slice_sensing_stop_time, stop_angle=slice_stop_angle)
            # - Position in the data-take
            if (index == 0) and (slice_dt_position in [DataTakePosition.FIRST, DataTakePosition.UNIQUE]):
                frame.set_dt_position(DataTakePosition.FIRST)
            elif (index == frames_in_slice - 1) and (
                slice_dt_position in [DataTakePosition.LAST, DataTakePosition.UNIQUE]
            ):
                frame.set_dt_position(DataTakePosition.LAST)
            else:
                frame.set_dt_position(DataTakePosition.CENTRAL)
            # - Data-loss position (if any) w.r.t. frame
            if (index == frames_in_slice - 1) and (slice_dl_position == DataLossPosition.RIGHT):
                frame.set_dl_position(DataLossPosition.RIGHT)
            elif (index == 0) and (slice_dl_position == DataLossPosition.LEFT):
                frame.set_dl_position(DataLossPosition.LEFT)
            else:
                frame.set_dl_position(DataLossPosition.NONE)
            # - Status
            if index in (0, frames_in_slice - 1):
                if frame.dl_position == DataLossPosition.NONE:
                    frame.set_status(FrameStatus.PARTIAL)
                else:
                    frame.set_status(FrameStatus.INCOMPLETE)
            else:
                frame.set_status(FrameStatus.NOMINAL)

            # If required:
            # - merge short L1 frames at beginning/end of L0 slice
            # - discard short L1 frames in the middle of datatake
            # - merge frames in case of short datatake
            if frame.duration < MINIMUM_FRAME_LENGTH + 2 * L1_PROCESSING_MARGIN or short_datatake_flag is True:
                if merge_short_frames_flag and frames_in_slice > 1:
                    if frame.dt_position == DataTakePosition.FIRST or frame.dl_position == DataLossPosition.LEFT:
                        frames_in_slice_list[index + 1].update_frame(
                            start_time=frame.start_time,
                            start_angle=frame.start_angle,
                        )
                        frames_in_slice_list[index + 1].set_status(FrameStatus.MERGED)
                        if frame.duration > frames_in_slice_list[index + 1].duration:
                            frames_in_slice_list[index + 1].index = frame.index
                    if frame.dt_position == DataTakePosition.LAST or frame.dl_position == DataLossPosition.RIGHT:
                        frames_in_slice_list[index - 1].update_frame(
                            stop_time=frame.stop_time,
                            stop_angle=frame.stop_angle,
                        )
                        frames_in_slice_list[index - 1].set_status(FrameStatus.MERGED)
                        if frame.duration > frames_in_slice_list[index - 1].duration:
                            frames_in_slice_list[index - 1].index = frame.index
                indexes_to_discard.append(index)
            elif frame.duration < SLICE_OVERLAP - L1_PROCESSING_MARGIN:
                if frame.dt_position == DataTakePosition.CENTRAL:
                    indexes_to_discard.append(index)

        for index in sorted(indexes_to_discard, reverse=True):
            del frames_in_slice_list[index]

        # Add L1 processing margins
        if add_l1processing_margins_flag:
            for index, frame in enumerate(frames_in_slice_list):
                start_time = max(
                    frame.start_time - np.timedelta64(L1_PROCESSING_MARGIN, "s"),
                    slice_sensing_start_time,
                )
                start_angle = self._time2angle(to_mjd2000(start_time) / SECONDS_PER_DAY).item()
                frame.update_frame(start_time=start_time, start_angle=start_angle)
                stop_time = min(
                    frame.stop_time + np.timedelta64(L1_PROCESSING_MARGIN, "s"),
                    slice_sensing_stop_time,
                )
                stop_angle = self._time2angle(to_mjd2000(stop_time) / SECONDS_PER_DAY).item()
                frame.update_frame(stop_time=stop_time, stop_angle=stop_angle)
                if frame.status == FrameStatus.NOMINAL and frame.duration < (
                    NOMINAL_FRAME_LENGTH + 2 * L1_PROCESSING_MARGIN - NOMINAL_FRAME_LENGTH_MARGIN
                ):
                    frame.set_status(FrameStatus.PARTIAL)

        return frames_in_slice_list
