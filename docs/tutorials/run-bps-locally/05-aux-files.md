<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# 5. Configure AUX files

Download the AUX package matching your BPS version from the
[Biomass DISC release notes](https://biomass-disc.info/release_note):

```bash
cd ~/bps-work/BPS/docs/tutorials/run-bps-locally/CONFIGURATION_FILE
mkdir AUX_443
unzip BIO_AUX_*.ZIP -d AUX_443/
rm *.ZIP
```

Resulting layout:

```
CONFIGURATION_FILE/
└── AUX_443/
    ├── BIO_AUX_INS____...
    ├── BIO_AUX_PP1____...
    ├── BIO_AUX_PP2_2A_...
    └── BIO_AUX_PPS____...
```

`config.ini` already points to `AUX_443` by default (after the `sed`
substitution from
[step 1](01-prerequisites.md#replace-the-bps_root-placeholders)).
For a different BPS version, edit `AUX_DEFAULT_DIR`.
