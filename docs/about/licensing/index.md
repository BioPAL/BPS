<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Licensing

BIOMASS BPS is an open-source project committed to transparency, collaboration,
and legal compliance. This section outlines licensing requirements, legal
obligations, and best practices for contributors and users of the codebase.

::::{grid} 1 2 2 2
:gutter: 3
:class-container: intro-grid

:::{grid-item-card}
:link: project-license
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-solid fa-scale-balanced fa-fw" aria-hidden="true"></i> **Project license**
^^^
Apache 2.0 terms, patent grant, trademark usage, and obligations for downstream users.
:::

:::{grid-item-card}
:link: contributions
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-solid fa-pen-nib fa-fw" aria-hidden="true"></i> **Contributions**
^^^
Contributor license terms, SPDX headers in source files, and the contributor checklist.
:::

:::{grid-item-card}
:link: dependencies
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-solid fa-puzzle-piece fa-fw" aria-hidden="true"></i> **Dependencies**
^^^
License compatibility for libraries, third-party code, NOTICE file, and PR requirements.
:::

:::{grid-item-card}
:link: reuse-compliance
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-solid fa-shield-halved fa-fw" aria-hidden="true"></i> **REUSE compliance**
^^^
`LICENSES/` directory, REUSE / SPDX standard, and the blocking CI gate.
:::

::::

## Summary

- **BIOMASS BPS uses Apache License 2.0**
- **All contributions must be Apache 2.0 compatible**
- **External dependencies must be license-compatible**
- **Proper attribution is required for third-party code**
- **Source files must include SPDX headers** (REUSE compliance: blocking CI gate)
- **`LICENSES/` directory must contain all license texts**
- **When in doubt, ask maintainers before submitting**

## Questions

If you have questions about licensing or legal requirements:

1. Review the pages linked above
2. Check existing issues on GitHub for similar questions
3. Open a new issue with the `licensing` label
4. Contact maintainers for urgent legal questions

If you believe you have found a license violation, do **not** create a public issue
immediately. Contact the project maintainers privately, provide specific details,
and allow time for investigation and resolution.

```{toctree}
:caption: Licensing
:maxdepth: 2
:hidden:

Project license <project-license>
Contributions <contributions>
Dependencies <dependencies>
REUSE compliance <reuse-compliance>
```
