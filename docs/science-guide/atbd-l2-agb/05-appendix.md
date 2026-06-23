<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

```{include} ../_includes/atbd-logo-banner.md
```


# 5. Appendix: effect of GN

The power associated with each pixel of $I_{GN}$ can be expressed as {rd}`4`:

(equation-eq-appendix-gn-1)=
\begin{equation}
|I_{GN}(r,a)|^2 = 2 \int \sigma(r,a,z)^2 [1 - \cos(k_z(r,a)\, z)]\, dz
\tag{5.1}
\end{equation}

where $\sigma(r,a,z)^2$ is uncorrelated reflectivity density along $z$, and
$k_z(r,a) = k_z(r,a,1) - k_z(r,a,2)$ is the interferometric pair vertical wavenumber.

