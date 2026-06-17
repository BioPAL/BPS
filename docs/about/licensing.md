# Licensing and Legal Requirements

## Overview

BIOMASS BPS is an open-source project committed to transparency, collaboration, and legal compliance. This document outlines the licensing requirements, legal obligations, and best practices for contributors and users of the BIOMASS BPS codebase.

---

## License: Apache License 2.0

BIOMASS BPS is licensed under the **Apache License 2.0** (Apache-2.0). This permissive open-source license allows you to:

- **Use** the software for any purpose, including commercial use
- **Modify** the code to suit your needs
- **Distribute** the original or modified versions
- **Sublicense** and distribute under different terms (with proper attribution)

### Key Requirements of Apache 2.0

When using or distributing BIOMASS BPS, you must:

1. **Include the original license and copyright notices**
2. **State any significant changes made to the code**
3. **Include a copy of the Apache 2.0 license**
4. **Include the NOTICE file** (if applicable)

For the full text of the Apache License 2.0, see: [https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

---

## Contributor License Requirements

### All Contributions Must Be Apache 2.0 Compatible

**By contributing to BIOMASS BPS, you agree that your contributions will be licensed under the Apache License 2.0.**

This means:

- You retain copyright ownership of your contributions
- You grant the BIOMASS BPS project (and all users) a perpetual, worldwide, non-exclusive, royalty-free license to use, modify, and distribute your contributions under Apache 2.0
- You have the right to submit the contribution
- Your contribution does not violate any third-party rights

### Copyright Assignment

You do **not** need to assign copyright to the BIOMASS BPS project. You retain ownership of your contributions, but you grant the necessary licenses for the project to use them under Apache 2.0.

---

## External Libraries and Dependencies

### License Compatibility

When adding external libraries or dependencies to BIOMASS BPS, you **must** ensure they are compatible with Apache 2.0. This is critical for maintaining legal clarity and avoiding license conflicts.

#### Compatible Licenses

The following licenses are generally compatible with Apache 2.0:

- **MIT License** - Fully compatible
- **BSD 2-Clause / 3-Clause** - Fully compatible
- **Apache 2.0** - Fully compatible
- **ISC License** - Fully compatible
- **Public Domain** - Compatible

#### Licenses Requiring Review

The following licenses may be compatible but require careful review:

- **LGPL 2.1 / 3.0** - Compatible if used as a dynamically linked library (not statically linked)
- **MPL 2.0** - Generally compatible, but review required
- **EPL 1.0 / 2.0** - Generally compatible, but review required

#### Incompatible Licenses

The following licenses are **NOT compatible** with Apache 2.0 and **cannot** be used:

- **GPL 2.0 / 3.0** - Incompatible (copyleft requirements conflict)
- **AGPL** - Incompatible
- **Proprietary licenses** - Incompatible
- **Any license that requires derivative works to use the same license** - Incompatible

### Requirements for Pull Requests

When submitting a Pull Request (PR) that includes external libraries or dependencies, you **must**:

1. **Document the license** of each new dependency in the PR description
2. **Verify compatibility** with Apache 2.0 before submission
3. **Include license information** in dependency management files (e.g., `requirements.txt`, `package.json`, `Cargo.toml`)
4. **Update the NOTICE file** (if applicable) with attribution requirements
5. **Provide justification** for why the dependency is necessary

#### Example PR Description Template

```markdown
## Dependencies Added

- **library-name** (v1.2.3)
  - License: MIT
  - Purpose: [Brief description]
  - Compatibility: Compatible with Apache 2.0
  - License URL: [Link to license]
```

### License Verification Process

Before merging any PR with new dependencies:

1. **Automated checks** will verify license compatibility
2. **Maintainers** will review license information
3. **Legal review** may be required for complex cases
4. **Documentation** must be updated with license information

---

## Third-Party Code and Code Snippets

### Using Code from Other Sources

If you want to include code from other sources (Stack Overflow, blogs, other projects, etc.), you must:

1. **Verify the license** of the source code
2. **Ensure compatibility** with Apache 2.0
3. **Provide proper attribution** in code comments
4. **Document the source** in your PR description

#### Attribution Format

```python
# This function is based on code from:
# Source: [URL or project name]
# License: [License name]
# Author: [Author name, if known]
```

### Code from Public Domain or Permissive Licenses

Code from sources with permissive licenses (MIT, BSD, Apache 2.0) can generally be included with proper attribution. Always verify the specific license terms.

### Code from GPL or Other Incompatible Licenses

**Do not include code from GPL-licensed projects or other incompatible licenses.** Even small snippets can create legal issues. If you need similar functionality, implement it from scratch or find an Apache 2.0 compatible alternative.

---

## Patent Rights

The Apache License 2.0 includes a patent grant clause. This means:

- Contributors grant patent licenses for their contributions
- Users receive patent licenses for using the software
- Patent rights are terminated if you file a patent lawsuit against the project

This provides additional protection for both contributors and users.

---

## Trademark Usage

### BIOMASS BPS Trademark

The "BIOMASS BPS" name and logo are trademarks. When using BIOMASS BPS:

- **You may use** the name to refer to the software
- **You may not use** the name or logo to imply endorsement without permission
- **You may not use** the name in a way that could cause confusion

### ESA and BIOMASS Mission

BIOMASS BPS is affiliated with the European Space Agency (ESA) and the BIOMASS mission. Respect ESA's trademark and branding guidelines when referencing these entities.

---

## License Headers in Source Files

### Required Header Format

All source code files should include a license header. The preferred format is the **SPDX short-form** (required for REUSE compliance: see section below):

```python
# SPDX-FileCopyrightText: 2026 BIOMASS BPS Contributors
#
# SPDX-License-Identifier: Apache-2.0
```

The long-form Apache header is also accepted for backwards compatibility:

```python
# Copyright [Year] BIOMASS BPS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

### Language-Specific Headers

- **Python**: Use `#` comments
- **JavaScript/TypeScript**: Use `//` or `/* */` comments
- **C/C++**: Use `//` or `/* */` comments
- **Rust**: Use `//` comments
- **Other languages**: Follow language-specific comment conventions

### Files That Don't Need Headers

The following files typically don't need license headers:

- Configuration files (YAML, JSON, TOML)
- Data files
- Generated files
- Very short utility scripts (use your judgment)

---

## LICENSES/ Directory

The `LICENSES/` directory at the root of the repository must contain the full text of every license used in the project. This is required by the REUSE standard (see section below).

Currently required files:

- `LICENSES/Apache-2.0.txt`: primary license of the project
- `LICENSES/MIT.txt`: license of certain dependencies or contributions

If you add a dependency with a new compatible license, add its full text to this directory and use the correct SPDX identifier in the file headers.

---

## Standard REUSE / SPDX

BIOMASS BPS follows the **FSFE REUSE** standard for license compliance. This standard is automatically verified by `fsfe/reuse-action` in the CI pipeline (`job: baseline-reuse`): it is a **blocking gate**: a PR with non-compliant files will be rejected.

### What the REUSE standard requires

1. **SPDX headers in every source file** (see "Required Header Format" above)
2. **`LICENSES/` directory** at the root of the project containing the full license texts
3. **Valid SPDX identifiers** in headers (`Apache-2.0`, `MIT`, etc.)

### Local verification

```bash
# Via pre-commit (reuse hook)
pre-commit run reuse

# Or directly
pip install reuse
reuse lint
```

### Resources

- REUSE standard: https://reuse.software/
- SPDX license list: https://spdx.org/licenses/
- GitHub Action: https://github.com/fsfe/reuse-action

---

## NOTICE File

The NOTICE file contains important attribution and legal information. If you add dependencies that require attribution, you may need to update the NOTICE file.

### When to Update NOTICE

Update the NOTICE file when:

- Adding dependencies with attribution requirements
- Including code from other projects that requires attribution
- Adding third-party components with specific notice requirements

---

## License Compliance Checklist for Contributors

Before submitting a Pull Request, verify:

- [ ] All new code is your original work or properly attributed
- [ ] All external dependencies are Apache 2.0 compatible
- [ ] License information is documented in the PR description
- [ ] Source files include SPDX headers (`SPDX-FileCopyrightText` + `SPDX-License-Identifier: Apache-2.0`)
- [ ] `reuse lint` passes locally (or pre-commit reuse hook passes)
- [ ] NOTICE file is updated (if required)
- [ ] No GPL or other incompatible code is included
- [ ] Third-party code snippets are properly attributed

---

## License Compliance Checklist for Users

When using BIOMASS BPS:

- [ ] Include the Apache 2.0 license file
- [ ] Include copyright notices
- [ ] Include the NOTICE file (if present)
- [ ] State any modifications made
- [ ] Comply with any additional requirements from dependencies

---

## Questions and Legal Issues

### Getting Help

If you have questions about licensing or legal requirements:

1. **Review this document** thoroughly
2. **Check existing issues** on GitHub for similar questions
3. **Open a new issue** with the `licensing` label
4. **Contact maintainers** for urgent legal questions

### Reporting License Violations

If you believe you've found a license violation:

1. **Do not** create a public issue immediately
2. **Contact** the project maintainers privately
3. **Provide** specific details about the violation
4. **Allow time** for investigation and resolution

---

## Additional Resources

- [Apache License 2.0 Full Text](https://www.apache.org/licenses/LICENSE-2.0)
- [Apache License 2.0 FAQ](https://www.apache.org/foundation/license-faq.html)
- [Open Source License Compatibility](https://opensource.org/licenses)
- [SPDX License List](https://spdx.org/licenses/)

---

## Summary

- **BIOMASS BPS uses Apache License 2.0**
- **All contributions must be Apache 2.0 compatible**
- **External dependencies must be license-compatible**
- **Proper attribution is required for third-party code**
- **Source files must include SPDX headers** (REUSE compliance: blocking CI gate)
- **`LICENSES/` directory must contain all license texts**
- **When in doubt, ask maintainers before submitting**

By contributing to BIOMASS BPS, you help maintain a legally compliant, open-source project that benefits the entire scientific community.

---

**Last Updated:** 2026

