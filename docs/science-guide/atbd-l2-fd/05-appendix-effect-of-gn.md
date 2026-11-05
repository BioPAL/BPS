<!-- SPDX-FileCopyrightText: 2026 European Space Agency (ESA) -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

:::{atbd-logo-banner}
:::


(sec:atbd-fd-5)=
# 5. Appendix: effect of GN

The power associated with each pixel of $I_{GN}$ can be expressed as {rd}`4`

$$
\left\vert I_{GN}(r,\ a) \right\vert^2=\ 2\int_{}^{}{{\sigma({r,\ a,\ z})}^2\left[ 1-\cos({k_z(r,a)z}) \right]}\ dz
\tag{5.1}
$$ (eq:atbd-fd-5-1)

where ${\sigma({r,\ a,\ z})}^2$ is the uncorrelated reflectivity density of the distributed target along z, and $k_z(r,\ a)\ =\ k_z(r,\ a,\ 1)\ -\ k_z(r,\ a,\ 2)$ is the interferometric pair vertical wavenumber.

Note that quantity in {eq}`eq|5.1` is never computed within `L2A_P`
