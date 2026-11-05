<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:atbd-l1-sar-6)=
# 6. Ancillary Functions and Parameters

(sec:atbd-l1-sar-6-1)=
### 6.1 Block Processing

The different steps of the L1 processing chain operate on data in two main ways:

- on an echo basis (e.g., range compression, radiometric correction, ground projection, …);
- on a block basis (e.g., azimuth compression, ionospheric calibration, multilook, …).

For the second case and due to the specific characteristics of BIOMASS data (swath coverage vs range sampling rate), the blocketization is performed only along azimuth direction. The definition of blocks dimensions and of their overlap is performed considering the best trade-off between usage of available hardware resources (a single block shall be kept and processed in memory) and efficiency (the ratio between block overlap and length shall be as greater as possible than 1). In addition, it shall be considered that a given block of data is processed using a fixed set of processing parameters (Doppler Centroid, Doppler Rate, …), so in case they present strong variations along azimuth direction inaccuracies in the results can be introduced. Finally, the algorithm implemented plays a key role and put some constraints on the definition (e.g., for azimuth compression overlap is of at least a synthetic aperture, in order to avoid border effects, while block size is of several apertures, e.g. 4).

In order to enhance computational performances and to reduce as much as possible I/O effort, data pre-fetching techniques are adopted. In this way dedicated processing threads are used to manage I/O operations, while the other threads working in parallel are dedicated only to processing operations.

(sec:atbd-l1-sar-6-2)=
### 6.2 Interpolation

Several steps in the L1 processing chain include interpolation operations. In particular the ones performed on SAR signal are always mono-dimensional (data resampling along azimuth to manage PRF changes, ground projection along range, …).

In order to preserve SAR signal characteristics (spatial resolution, 2D spectra and phase, …) the following methods are used, depending on the specific cases:

- interpolation through FFT method, if the data is regularly sampled and shall be interpolated of a rationale factor (e.g., oversampling of a factor of 2). In this case the procedure applied is the following one:
    - the original signal ($f_{in}$ sampling frequency, $N$ samples) is transformed in frequency domain through an FFT operator;
    - the signal is then zero-padded at Nyquist frequency to reach the desired length ($M$ samples);
    - the signal is then inverse-transformed in time domain through an inverse-FFT operator ($f_{out}=\frac{M}{N}f_{in}$ sampling frequency, $M$ samples);
- interpolation using a bank of FIR filters (FIR, SINC, GLS), if the data is generically sampled and shall be resampled to a generic sampling frequency and/or on a generic axis, also irregular. In this case the procedure applied is the following one:
    - the bank of filter is generated creating a set of $N_F$ filters of length $L_F$. All the filters are of the same type (FIR, SINC, GLS), sampled at the frequency of the input signal ($f_{in}$) and with a bandwidth that can be the one of the input signal too ($B_{in}$) or lower in case it is required to apply a low-pass filtering in addition to the re-sampling ($B_{out}$). The filters are generated such that they are shifted between them of a fraction of sample (e.g., if $N_F=100$, the shift of two consecutive filters in the bank is equal to $\frac{1}{100}$ of sample);
    - each sample of the re-sampled signal is then computed through a correlation of the original signal with a properly selected filter in the bank. The selection is performed considering where the output sample falls with respect to the input sampling grid (e.g., if it falls exactly on the input sampling grid the first filter in the bank is selected, if it falls exactly in the middle between two samples of the input grid the central filter is selected, and so on ...).

The second option is the one used in the most critical steps of the chain, as the two ones cited at the beginning of this section. The characteristics of the filter banks (type and number of filters, filters length, …) depend on the characteristics of the SAR signal to be interpolated (bandwidth vs sampling frequency ratio, …). For example, keeping fixed the filters length, GLS filters have a higher ripples suppression ratio with respect to SINC ones. A proper design of such filter banks is mandatory in order to ensure the best trade-off between data quality and computational performances. As per BPS requirements, the interpolation operations performed by L1 Processor shall satisfy the preservation of the following characteristics on L1a products:

- resolution degradation \<0.5% of the nominal one;
- spectral amplitude degradation \<0.1dB w.r.t. the nominal one;
- interferometric phase error satisfying the CEOS Offset Test requirements (mean ≤ 0.1deg, standard deviation ≤ 5.5deg).

In each processing step requiring this solution, the filter bank is computed at runtime only once at the beginning of the processing and then applied to the entire data, in order to reduce as much as possible the computational time.

(sec:atbd-l1-sar-6-3)=
### 6.3 Orbit Interpolation

In the SAR Focusing Kernel orbit interpolation operations are performed using internal libraries instead of EO-CFI ones. The demonstration that the same performances are ensured has been object of a dedicated activity in the framework of BIOMASS L1 GPP project.

In order to retrieve the sensor position and velocity at a generic azimuth time $t$ the orbit must be interpolated starting from the available discrete set of positions and velocities and the related times. These parameters can be represented in the following way:

- $p_s(t\left[ k \right])$ is the position of the sensor at time $t\left[ k \right]$, with $k=0:N-1$
- $v_s(t\left[ k \right])$ is the velocity of the sensor at time $t\left[ k \right]$
- the $N$ time points define $N-1$ time intervals $\left[ t\left[ k \right];t\left[ k+1 \right] \right]$, with $k=0:N-2$

The interpolation technique is based on piecewise cubic (or higher order) spline interpolation. One of the key properties of spline interpolation is that it ensures continuity of the resulting curve and its first and second derivatives in the interval $\left[ t\left[ 0 \right];t\left[ N-1 \right] \right]$. In details:

- a spline polynomial is defined for each time interval of the sampled orbit. To define the $N-1$ spline polynomials on the $N-1$ intervals, $N+2$ constraints are needed:
    - $N$ constraints are defined by imposing the interpolation curve to pass through $p_s(t\left[ k \right])$ at time $t\left[ k \right],\ 0\leq{}k<N$;
    - the two remaining conditions are defined by imposing the first derivative (velocity) values at the first and last time points, i.e., $v_s(t\left[ 0 \right])$ and $v_s(t\left[ N-1 \right])$ (clamped spline);
- computation of sensor position at a time $t\left[ 0 \right]$ in the interval $\left[ t\left[ k \right];t\left[ k+1 \right] \right]$ is performed by evaluating the k-th spline polynomial at the specified time;
- velocity and acceleration are computed in a similar manner, but evaluating the first and second derivative polynomials respectively;
- note that $v_s(t\left[ 1 \right])$..$v_s(t\left[ N-2 \right])$ are not used to perform interpolation and evaluation.

Note that the degree of the spline interpolator can be easily increased in case of need, if deemed necessary to ensure accuracy of L1 Processor or of following processing steps (Stack Processor). For L1 Processor, by default 5th order is used.

(sec:atbd-l1-sar-6-4)=
### 6.4 Attitude Interpolation

In the SAR Focusing Kernel attitude interpolation operations are performed using internal libraries instead of EO-CFI ones.

As for the orbit, in order to retrieve the sensor attitude (Euler angles) at a generic azimuth time $t$ the attitude must be interpolated starting from the available discrete set of yaw, pitch and roll angles and the related times.

The procedure is similar to the one described in the previous section, with the difference that the interpolation is based on piecewise 4th-order polynomials, instead than on splines, then 5 constraints are needed to fit them.

(sec:atbd-l1-sar-6-5)=
### 6.5 Weighting Functions

Weighting is applied after SAR processing in order to reduce side-lobes and the effective extension in both azimuth and range of the impulse response of the system. The price to pay is, of course, in terms of maximum achievable resolution. Typical values for side-lobe attenuation are over 15 dB with resolution losses of about 15%. Weighting functions are typically applied in the frequency domain after range and azimuth compression, respectively. The windows of interest for the L1 Processor are the following:

- Raised cosine, with a spectral shape of the form

$$
W_{RC}\left[ n \right]=\alpha+(1-\alpha)\cdot\cos (\frac{2\pi\cdot{}n}{N-1})\ ,
\tag{6.1}
$$ (eq:atbd-l1-sar-6-1)

where $n$ represents the discrete frequency variable, $N$ the total, number of samples, and the parameter $\alpha$ controls the level of raising of the cosine. Hamming and Hann windows are specific forms of the raised cosine with values of $\alpha$ 0.54 and 0.5, respectively. The raised cosine is currently the selected window for the L1 Processor with different parameters in range and azimuth for the $\alpha$ parameter;

- Kaiser whose spectral shape of the form

$$
W_K\left[ n \right]=\frac{I_0(\alpha\cdot\sqrt{1-(2\cdot\frac{n}{N-1}-1)^2})}{I_0(\alpha)},
\tag{6.2}
$$ (eq:atbd-l1-sar-6-2)

where $I_0(\cdot)$ is the zero-th order modified Bessel function, and the parameter $\alpha$ controls the extension of the main lobe of the impulse response;

- Chebyshev, with a spectral shape of the form

$$
W_C\left[ n \right]=\frac{\cos\left\{ N\cdot\arccos \left[ \cosh (\frac{\cosh^{-1}{10^{\alpha}}}{N})\cdot\cos (\frac{\pi\cdot{}n}{N}) \right] \right\}}{\cosh^{-1}{10^{\alpha}}}\ ,
\tag{6.3}
$$ (eq:atbd-l1-sar-6-3)

where $\alpha$ is the parameter the attenuation of the side-lobes, kept at a constant level.
