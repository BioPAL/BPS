# Contributions

## Contributor license requirements

### All contributions must be Apache 2.0 compatible

**By contributing to BIOMASS BPS, you agree that your contributions will be licensed under the Apache License 2.0.**

This means:

- You retain copyright ownership of your contributions
- You grant the BIOMASS BPS project (and all users) a perpetual, worldwide, non-exclusive, royalty-free license to use, modify, and distribute your contributions under Apache 2.0
- You have the right to submit the contribution
- Your contribution does not violate any third-party rights

### Copyright assignment

You do **not** need to assign copyright to the BIOMASS BPS project. You retain ownership of your contributions, but you grant the necessary licenses for the project to use them under Apache 2.0.

## License headers in source files

### Required header format

All source code files should include a license header. The preferred format is the **SPDX short-form** (required for [REUSE compliance](reuse-compliance.md)):

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

### Language-specific headers

- **Python**: Use `#` comments
- **JavaScript/TypeScript**: Use `//` or `/* */` comments
- **C/C++**: Use `//` or `/* */` comments
- **Rust**: Use `//` comments
- **Other languages**: Follow language-specific comment conventions

### Files that do not need headers

The following files typically don't need license headers:

- Configuration files (YAML, JSON, TOML)
- Data files
- Generated files
- Very short utility scripts (use your judgment)

## License compliance checklist for contributors

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

**Previous:** [Project license](project-license.md) | **Next:** [Dependencies](dependencies.md)
