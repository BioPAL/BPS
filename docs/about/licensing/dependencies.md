# Dependencies

## License compatibility

When adding external libraries or dependencies to BIOMASS BPS, you **must** ensure they are compatible with Apache 2.0. This is critical for maintaining legal clarity and avoiding license conflicts.

### Compatible licenses

The following licenses are generally compatible with Apache 2.0:

- **MIT License** - Fully compatible
- **BSD 2-Clause / 3-Clause** - Fully compatible
- **Apache 2.0** - Fully compatible
- **ISC License** - Fully compatible
- **Public Domain** - Compatible

### Licenses requiring review

The following licenses may be compatible but require careful review:

- **LGPL 2.1 / 3.0** - Compatible if used as a dynamically linked library (not statically linked)
- **MPL 2.0** - Generally compatible, but review required
- **EPL 1.0 / 2.0** - Generally compatible, but review required

### Incompatible licenses

The following licenses are **NOT compatible** with Apache 2.0 and **cannot** be used:

- **GPL 2.0 / 3.0** - Incompatible (copyleft requirements conflict)
- **AGPL** - Incompatible
- **Proprietary licenses** - Incompatible
- **Any license that requires derivative works to use the same license** - Incompatible

## Requirements for pull requests

When submitting a Pull Request (PR) that includes external libraries or dependencies, you **must**:

1. **Document the license** of each new dependency in the PR description
2. **Verify compatibility** with Apache 2.0 before submission
3. **Include license information** in dependency management files (e.g., `requirements.txt`, `package.json`, `Cargo.toml`)
4. **Update the NOTICE file** (if applicable) with attribution requirements
5. **Provide justification** for why the dependency is necessary

### Example PR description template

```markdown
## Dependencies Added

- **library-name** (v1.2.3)
  - License: MIT
  - Purpose: [Brief description]
  - Compatibility: Compatible with Apache 2.0
  - License URL: [Link to license]
```

## License verification process

Before merging any PR with new dependencies:

1. **Automated checks** will verify license compatibility
2. **Maintainers** will review license information
3. **Legal review** may be required for complex cases
4. **Documentation** must be updated with license information

## Third-party code and code snippets

### Using code from other sources

If you want to include code from other sources (Stack Overflow, blogs, other projects, etc.), you must:

1. **Verify the license** of the source code
2. **Ensure compatibility** with Apache 2.0
3. **Provide proper attribution** in code comments
4. **Document the source** in your PR description

### Attribution format

```python
# This function is based on code from:
# Source: [URL or project name]
# License: [License name]
# Author: [Author name, if known]
```

### Code from public domain or permissive licenses

Code from sources with permissive licenses (MIT, BSD, Apache 2.0) can generally be included with proper attribution. Always verify the specific license terms.

### Code from GPL or other incompatible licenses

**Do not include code from GPL-licensed projects or other incompatible licenses.** Even small snippets can create legal issues. If you need similar functionality, implement it from scratch or find an Apache 2.0 compatible alternative.

## NOTICE file

The NOTICE file contains important attribution and legal information. If you add dependencies that require attribution, you may need to update the NOTICE file.

### When to update NOTICE

Update the NOTICE file when:

- Adding dependencies with attribution requirements
- Including code from other projects that requires attribution
- Adding third-party components with specific notice requirements

---

**Previous:** [Contributions](contributions.md) | **Next:** [REUSE compliance](reuse-compliance.md)
