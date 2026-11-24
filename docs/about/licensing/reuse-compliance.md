# REUSE compliance

BIOMASS BPS follows the **FSFE REUSE** standard for license compliance. This standard is automatically verified by `fsfe/reuse-action` in the CI pipeline (`job: baseline-reuse`): it is a **blocking gate** — a PR with non-compliant files will be rejected.

For SPDX header format, see [Contributions](contributions.md#required-header-format).

## LICENSES/ directory

The `LICENSES/` directory at the root of the repository must contain the full text of every license used in the project. This is required by the REUSE standard.

Currently required files:

- `LICENSES/Apache-2.0.txt`: primary license of the project
- `LICENSES/MIT.txt`: license of certain dependencies or contributions

If you add a dependency with a new compatible license, add its full text to this directory and use the correct SPDX identifier in the file headers.

## What the REUSE standard requires

1. **SPDX headers in every source file** (see [Contributions](contributions.md#required-header-format))
2. **`LICENSES/` directory** at the root of the project containing the full license texts
3. **Valid SPDX identifiers** in headers (`Apache-2.0`, `MIT`, etc.)

## Local verification

```bash
# Via pre-commit (reuse hook)
pre-commit run reuse

# Or directly
pip install reuse
reuse lint
```

## CI context

REUSE verification runs as part of the baseline CI tier. See
[CI automation and contribution tiers](../../contributing/ci-automation-and-contribution-tiers.md)
for how baseline checks gate pull requests.

## Resources

- REUSE standard: https://reuse.software/
- SPDX license list: https://spdx.org/licenses/
- GitHub Action: https://github.com/fsfe/reuse-action

---

**Previous:** [Dependencies](dependencies.md) | **Next:** [Licensing](index.md)
