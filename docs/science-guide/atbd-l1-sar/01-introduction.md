<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:atbd-l1-sar-1)=
# 1. Introduction

(sec:atbd-l1-sar-1-1)=
### 1.1 Context

BIOMASS is ESA’s 7th Earth Explorer mission. The purpose of the BIOMASS is to reduce the uncertainty in the worldwide spatial distribution and dynamics of forest biomass in order to improve current assessments and future projections of the global carbon cycle. This objective will be achieved by the implementation of a P-band SAR mission, providing global maps of forest biomass stocks, forest disturbance and growth.

The BIOMASS mission will also provide 3D views of forests through the “tomographic” phase and will also support additional science needs arising from the opportunity to explore the Earth for the first time with a P-band SAR system from space offering the possibility to “see” below the vegetated areas and in beneath bare soil or icy regions.

To achieve its challenging objectives, the BIOMASS mission requires a specific data processing strategy that combines different technologies at different processing level in order to generate the final Above Ground Biomass global map.

(sec:atbd-l1-sar-1-2)=
### 1.2 Scope of the document

The purpose of this document is to provide the “L1 Processor Algorithms Theoretical Baseline Document (ATBD)” for the BIOMASS Processing Suite (BPS) project.

This document is organized as follows:

- {sec}`Section|2`: Brief introduction about BPS and L1 Processor;
- {sec}`Section|3`: Description of L1 Pre-Processing algorithms;
- {sec}`Section|4`: Description of L1 Core Processing algorithms;
- {sec}`Section|5`: Description of L1 Post-Processing algorithms;
- {sec}`Section|6`: Description of ancillary functions and parameters;
- {sec}`Section|7`: Description of configurable input parameters;
- {sec}`Section|8`: List of open points.

(sec:atbd-l1-sar-1-3)=
### 1.3 Acronyms

:::{dropdown} Acronyms
| Acronym | Description |
|---|---|
| AD | Applicable Document |
| ADC | Analog-to-Digital Converter |
| APID | Application Process IDentifier |
| ANC | ANCillary |
| ARESYS | Advanced Remote Sensing and Systems |
| ASM | Average Signal Magnitude |
| ATBD | Algorithm Theoretical Baseline Document |
| AUX | Auxiliary |
| BAQ | Bit Adaptive Quantization |
| BPS | BIOMASS Processing Suite |
| CAL | Calibration |
| CCDB | Calibration and Characterisation Data Base |
| CDASP | Characterised Doublet Antenna Secondary Pattern |
| CFI | Customer Furnished Items |
| CRC | Cyclical Redundancy Checking |
| DEM | Digital Elevation Model |
| DFDL | Data Format Description Language |
| DGM | Detected Ground-range Multilooked |
| ECSS | European Cooperation for Space Standardization |
| EO-CFI | Earth Observation Custom Furnished Items |
| ESA | European Space Agency |
| GLS | Generalised Least Square |
| GPP | Ground Processor Prototype |
| GPS | Global Positioning System |
| HK | House Keeping |
| HKTM | House Keeping TeleMetry |
| ICAL | Internal Calibration |
| IERS | International Earth Rotation Service |
| INT | Interferometric Phase |
| IRF | Impulse Response Function |
| ISLR | Integrated Sidelobe Ratio |
| ISP | Instrument Source Packet |
| L0 | Level-0 product |
| L1a | Level-1a product |
| L1b | Level-1b product |
| L1M | Level-1 Monitoring product |
| L1S | Level-1 Standard product |
| L1PP | Level 1 Processor Prototype |
| LUT | Look-Up Table |
| PARC | Polarimetric Active Radar Calibrator |
| PAS | Power Amplifier Subsystem |
| PCAT | Packet CATegory |
| PF | Processing Facility |
| PFD | Product Format Specification Document |
| PGP | Power Gain Product |
| PID | Process IDentifier |
| PSLR | Peak-to-Sidelobe Ratio |
| PTAR | Point Target Ambiguity Ratio |
| QC | Quality Check |
| RD | Reference Document |
| RFI | Radio-Frequency Interferences |
| RXO | RX-Only product |
| SAR | Synthetic Aperture Radar |
| SCR | Signal-to-Clutter Ratio |
| SCS | Single-look Complex Slant-range |
| SNR | Signal-to-Noise Ratio |
| SSPA | Solid State Power Amplifier |
| SWL | Sample Window Length |
| SWP | Sampling Window Position |
| TBD | To Be Defined |
| TBC | To Be Confirmed |
| TOM | Tomographic Phase |
| UTC | Coordinated Universal Time |
| XML | `eXtended` Markup Language |
:::

(sec:atbd-l1-sar-1-4)=
### 1.4 Symbols and Variables

:::{dropdown} Acronyms
| Acronym | Description |
|---|---|
| $f_r$ | Range frequency |
| $f_a$ | Azimuth frequency |
| $\omega_r$ | Range wavenumber |
| $k_r$ | Range wavenumber after Stolt interpolation |
| $k_a$ | Azimuth (Doppler) wavenumber |
| $\tau$ | Fast time (slant range) |
| $t$ | Slow time (azimuth) |
| $\tau_r$ | Chirp duration in seconds |
| $N_r$ | Chirp duration in samples |
| $B_r$ | Chirp bandwidth |
| $B_a$ | Azimuth bandwidth |
| $f_0$ | Carrier frequency |
| $\omega_0$ | Wavenumber at carrier frequency |
| $\lambda$ | The wavelength |
| $f_s$ | Range sampling frequency |
| $PRF$ | Pulse repetition frequency |
| $K_r$ | Chirp rate |
| $K_a$ | Doppler rate |
| $f_{DC}$ | Doppler centroid |
| $v_s,\ v_s^i$ | Satellite velocity in ECEF and inertial reference frame, respectively |
| $v_e,v_e^i$ | Effective velocity in ECEF and inertial reference frame, respectively |
| $v_{e0C}$ | Effective velocity at reference range |
| $v_g$ | Ground velocity |
| $r_0$ | Zero-Doppler closest approach distance |
| $r$ | Slant-range vector |
| $\beta$ | Squint angle |
| $c_0$ | Speed of light in vacuum |
| $\Omega$ | One-way Faraday rotation angle |
| $S_{xx}$ | Scattering coefficient at transmit/receive polarization $xx$ |
| $\widehat{k}$ | Unit propagation vector of transmitted wave |
| $\overrightarrow{B}$ | Earth’s magnetic field vector |
| $TEC$ | Total Electron Content |
| $h_{iono}$ | Ionospheric height (thin layer model) |
| $\theta$ | Incidence angle |
| $\theta_l$ | Off-nadir look angle |
| $\varphi_{iono}$ | Ionospheric phase screen |
| $t_0$ | Zero-Doppler position for a given target |
| $\boldsymbol{q}$ | Quaternions of the attitude product |
| $\overrightarrow{p_{LOS}},\ \overrightarrow{p_{LOS,PF}}$ | Antenna pointing vector in the ECEF coordinates and in the pointing reference frame. |
| $R_e,R_p$ | Equatorial and polar radius of the reference ellipsoid, respectively. |
:::

(sec:atbd-l1-sar-1-5)=
### 1.5 Applicable documents

| Code | Document |
|---|---|
| <span id="ad1"></span>`[AD1]` | SAR Data ICD, `BIO-ICD-ADSF-SY-1000563385`, Issue 5 |
| <span id="ad2"></span>`[AD2]` | Biomass Coordinate System Definition Document, BIO.TCN.ASU.SY.00012, Issue 6 |
| <span id="ad3"></span>`[AD3]` | Instrument Internal Calibration Description, `BIO-PL-ADSF-SY-1000144284`, Issue 7.0 |
| <span id="ad4"></span>`[AD4]` | Processing of ICAL Data (Technical Memorandum), `BIO-TN-ADSP-SY-1000468852`, Issue 2.0, Nov. 2018 |
| <span id="ad5"></span>`[AD5]` | Action 77355 – Use of Antenna Pattern and ICAL Data for Ground Processing, `BIO-TN-ADSP-SY-1000362485`, Airbus D&S, Issue 1, May 2018 |
| <span id="ad6"></span>`[AD6]` | Instrument Data Processing Guidelines for the GPP, `BIO-NO-ADSS-SY-1001405024`, Issue 04, Apr 2025 |

(sec:atbd-l1-sar-1-6)=
### 1.6 Reference documents

| Code | Document |
|---|---|
| <span id="rd1"></span>`[RD1]` | BIOMASS Ground Processor Prototype: L1 Algorithm Theoretical Baseline Document, BIO-GPP-ATBD-GP06, Issue 2.2, July 2022 |
| <span id="rd2"></span>`[RD2]` | F. Li, D. Held, J. Curlander, and C. Wu, “Doppler Parameter Estimation for Spaceborne Synthetic Aperture Radars,” IEEE Transactions on Geoscience and Remote Sensing, Jan. 1985. |
| <span id="rd3"></span>`[RD3]` | P. H. Eichel and C. V. Jakowatz, Jr., “Phase-gradient algorithm as an optimal estimator of the phase derivative,” Opt. Lett., vol. 14, no. 20, pp. 1101–1103, Oct. 1989.1 |
| <span id="rd4"></span>`[RD4]` | C. V. Jakowatz, Jr. and D. E. Whal, “Eigenvector method for maximumlikelihood estimation of phase errors in synthetic-aperture radar imagery,” J. Opt. Soc. Amer. A, Opt. Image Sci., vol. 10, no. 12, pp. 2539–2546, Oct. 1993. |
| <span id="rd5"></span>`[RD5]` | D. E.Wahl, P. Eichel, D. C. Ghiglia, and C. V. Jakowatz, Jr., “Phase gradient autofocus—A robust tool for high resolution SAR phase correction,” IEEE Trans. Aerosp. Electron. Syst., vol. 30, no. 3, pp. 827–835, Jul. 1994. |
| <span id="rd6"></span>`[RD6]` | W. Ye, T. S. Yeo, and Z. Bao, “Weighted least-squares estimation of phase errors for SAR/ISAR autofocus,” IEEE Trans. Geosci. Remote Sens., vol. 37, no. 5, pp. 2487–2494, Sep. 1999. |
| <span id="rd7"></span>`[RD7]` | J. R. Moreira, “A new method of aircraft motion error extraction from radar raw data for real time motion compensation,” IEEE Trans. Geosci.Remote Sens., vol. 28, no. 4, pp. 620–626, Jul. 1990. |
| <span id="rd8"></span>`[RD8]` | K. Camara de Macedo, R. Scheiber, A. Moreira, “An Autofocus Approach for Residual Motion Errors With Application to Airborne Repeat-Pass SAR Interferometry,” IEEE Transactions on Geoscience and Remote Sensing, Oct. 2008. |
| <span id="rd9"></span>`[RD9]` | W. G. Carrara, R. S. Goodman, and R. M. Majewski, Spotlight Synthetic Aperture Radar—Signal Processing Algorithms. Boston, MA: Artech House, 1995. |
| <span id="rd10"></span>`[RD10]` | S. H. Bickel and R. T. H. Bates, “Effects of magnetoionic propagation on the polarization scattering matrix,” Proc. IEEE, vol. 53, no. 8, pp. 1089-1901, 1965 |
| <span id="rd11"></span>`[RD11]` | “Ground Processing Strategy for RFI Mitigation - BIOMASS Phase B1”, BM-SS-ARE-RP-0042, Issue 2, ARESYS S.r.l. |
| <span id="rd12"></span>`[RD12]` | J. S. Kim, K. Papathanassiou, R. Scheiber and S. Quegan, “Correcting distortion of polarimetric SAR data induced by ionospheric scintillation,” IEEE Transactions on Geoscinence and Remote Sensing, No. vol. 12, pp. 2015 |
| <span id="rd13"></span>`[RD13]` | ENVISAT Ground Segment – ASAR Processing and Product Quality Requirements – Appendix A.1, PO-RS-ESA-GS-00245, Issue 3.3, July 1997. |
| <span id="rd14"></span>`[RD14]` | S. N. Madsen, "Estimating the Doppler centroid of SAR data," in IEEE Transactions on Aerospace and Electronic Systems, vol. 25, no. 2, pp. 134-140, March 1989. |
| <span id="rd15"></span>`[RD15]` | R Stolt, “Migration by Fourier transform”, Geophysics, 43:23–48, 1978. |
| <span id="rd16"></span>`[RD16]` | F. Rocca, “Synthetic aperture radar: A new application for wave equation techniques”, Stanford Exploration Project Report, SEP-56:167–189, 1987. |
| <span id="rd17"></span>`[RD17]` | R. Bamler, “A comparison of range-Doppler and wave-number domain SAR focusing algorithms”, IEEE Transactions on Geoscience and Remote Sensing, 30(4):706–713, July 1992. |
| <span id="rd18"></span>`[RD18]` | D. D’Aria, A. Monti Guarnieri, “High-resolution space borne SAR focusing by SVD-Stolt”, Geoscience and Remote Sensing Letters, IEEE, 4(4):639–643, Oct. 2007. |
| <span id="rd19"></span>`[RD19]` | E. Meler, U. Frei, D. Nüesch, “Precise Terrain Corrected Geocoded Images. In: SAR Geocoding: Data and Systems,” (Scheier G. ed.), Herbert Wichmann Verlag GmbH, Karlstuhe, pp.173-186, 1993. |
| <span id="rd20"></span>`[RD20]` | J. J. Mohr, S.N. Madsen, “Geometric Calibration of ERS Satellite SAR Images,” IEEE Transactions on Geoscience and Remote Sensing, 39 (4), pp. 842-850, 2001. |
| <span id="rd21"></span>`[RD21]` | E. Sansosti, P. Berardino, M. Manunta, F. Serafino, G. Fornaro, “Geometrical SAR image registration,” IEEE Transactions on Geoscience and Remote Sensing, 44(10), pp.2861-2870, 2006. |
| <span id="rd22"></span>`[RD22]` | On Range Compression Approaches for the BIOMASS GPP, BIO-GPP-DLR-TN01, October 2018, DLR |
| <span id="rd23"></span>`[RD23]` | BIOMASS GPP - Analysis of Transponder Delay Impacts on Impulse Response Function, ARE-012361, September 2018, ARESYS |
| <span id="rd24"></span>`[RD24]` | Earth Observation Mission CFI Software: General Software User Manual, Deimos Space S.L.U., Issue 4.15, April 2018 |
| <span id="rd25"></span>`[RD25]` | [`BPS_AUX_FMT`] BPS Auxiliary Product Format, I/R 2/1/1 |
| <span id="rd26"></span>`[RD26]` | [`BPS_L1_PFD`] BIOMASS L1a/b/c Products Format Specification, I/R 1/2/2 |
| <span id="rd27"></span>`[RD27]` | Antenna Pattern Computation, `BIO-TN-ADSP-SY-1000550571`, Issue 02 |
| <span id="rd28"></span>`[RD28]` | W. H. Press, S. A. Teukolsky, W. T. Vetterling, B. P. Flannery, “Numerical Recipes in C”, Cambridge University Press, Second Edition |
| <span id="rd29"></span>`[RD29]` | `BPS_TN_IONO`, “BIOMASS Ionospheric Corrections Technical Note”, I/R 1/0 |
| <span id="rd30"></span>`[RD30]` | Karam, L.J., and J. H. McClellan. "Complex Chebyshev Approximation for FIR Filter Design." IEEE Transactions on Circuits and Systems II: Analog and Digital Signal Processing 42, no. 3 (March 1995): 207–216 |
| <span id="rd31"></span>`[RD31]` | Gomba, G., Parizzi, A., De Zan, F., Eineder, M., & Bamler, R. (2015). Toward operational compensation of ionospheric effects in SAR interferograms: The split-spectrum method. IEEE Transactions on Geoscience and Remote Sensing, 54(3), 1446-1461. |
| <span id="rd32"></span>`[RD32]` | Altadill, D., S. Magdaleno, J.M. Torta, and E. Blanch C, “Global Empirical Models of the Density Peak Height and of the Equivalent Scale Height for Quiet Conditions”, Adv. Space Res. 52, 1756-1769, 2013. |
| <span id="rd33"></span>`[RD33]` | https://github.com/timduly4/pyglow |
| <span id="rd34"></span>`[RD34]` | Rino, C. L., & Fremouw, E. J. (1977). The angle dependence of singly scattered wavefields. Journal of Atmospheric and Terrestrial Physics, 39(8), 859-868. |
| <span id="rd35"></span>`[RD35]` | Fremouw, E. J., & Secan, J. A. (1984). Modeling and scientific application of scintillation results. Radio Science, 19(03), 687-694. |
| <span id="rd36"></span>`[RD36]` | Welch, P. D. (1967), "The use of Fast Fourier Transform for the estimation of power spectra: A method based on time averaging over short, modified periodograms" (PDF), IEEE Transactions on Audio and Electroacoustics, AU-15 (2): 70–73 |
| <span id="rd37"></span>`[RD37]` | `BIO-UDB-ADSF-SY-1000695798`, Instrument RDB, Version 17 |
| <span id="rd38"></span>`[RD38]` | Alken, P., Thébault, E., Beggan, C. D., Amit, H., Aubert, J., Baerenzung, J., ... & Zhou, B. (2021). International geomagnetic reference field: the thirteenth generation. Earth, Planets and Space, 73(1), 1-25 |
| <span id="rd39"></span>`[RD39]` | `BPS_SDD`, “BPS Software Design Document”, I/R 3/2/4 |
| <span id="rd40"></span>`[RD40]` | ETAD-DLR-DD-0008, S-1 ETAD ATBD, I/R 2/3 |
