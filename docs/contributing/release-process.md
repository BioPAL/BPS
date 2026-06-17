# Release Process

## Overview

```
develop → release → main → tag bps-<module>-X.Y.Z → release.yml → GitHub Release
```

`release` is a long-lived release candidate branch. The promotion path is:

1. PR from a feature branch into `develop` (daily integration).
2. PR from `develop` into `release` (release candidate assembly, Heavy CI mandatory).
3. PR from `release` into `main` (ESA approval, production promotion).
4. Tag on `main` triggers `release.yml` and produces the GitHub Release.

Releases are per-module (each `bps-*` package has its own semver tag) but share a global suite version tracked in `VERSION`.

---

## Steps

### 1. Open a PR from `develop` to `release`

When the integration branch is ready to become a release candidate, open a PR with `release` as the target branch.

> **Heavy CI is mandatory on `release` PRs.** The `CI gate` will be red on every `pull_request` event until a maintainer triggers the workflow via `Actions → Run workflow → run_heavy=true`. This is the explicit consent step that records Heavy authorisation before promotion to `main`.

The PR requires:

- **2 approvals** (Maintainer + SME, per the `release` branch ruleset)
- **GPG/SSH signed commits**
- **`CI gate`** passing (which itself requires `run_heavy=true` to have been dispatched)

### 2. Update VERSION and CHANGELOG

On the same PR, bump the version and update the changelog:

```bash
nox -s version_update
```

Then update `CHANGELOG.md` with the changes for this release.

> **ESA gate:** both `VERSION` and `CHANGELOG.md` are listed in `CODEOWNERS`. Any PR modifying these files requires explicit approval from the ESA representatives (Klaus Scipal, final authority, and Clement Albinet). See [Repository stewards](../governance/repository-stewards.md). No release can be merged into `main` without this approval.

Commit with sign-off:

```bash
git commit -s -m "chore: bump version to x.y.z and update CHANGELOG"
git push
```

### 3. Merge to `release`

Squash merge into `release` once the 2 approvals are in and `CI gate` is green.

### 4. Open a PR from `release` to `main`

Once `release` is stable and validated, open a promotion PR to `main`.

The PR requires:

- **3 approvals** (Maintainer + SME + ESA, per the `main` branch ruleset)
- **GPG/SSH signed commits**
- **`CI gate`** passing
- **ESA reviewer approval** on `VERSION` and `CHANGELOG.md`

### 5. Merge to main

Squash merge onto `main`. The final commit message becomes the release title.

### 6. Create the tag

Tag format: `bps-<module>-X.Y.Z`

```bash
git checkout main
git pull origin main
git tag bps-l2b_fh_processor-5.0.0
git push origin bps-l2b_fh_processor-5.0.0
```

The tag push automatically triggers `release.yml`.

---

## Release Pipeline (automatic)

Once the tag is pushed, the following jobs run automatically:

| Job | Description |
|-----|-------------|
| **Build** | `python -m build`: produces `sdist` + `wheel` in `dist/` |
| **SBOM** | Generates a Software Bill of Materials in CycloneDX JSON format (`sbom.json`) |
| **REUSE check** | Final `reuse lint` compliance check |
| **GitHub Release** | Creates a GitHub Release with `dist/*` + `sbom.json` as artefacts and `CHANGELOG.md` as release notes |

---

## PyPI Publication (future)

Publication to PyPI via OIDC trusted publishing is prepared in `release.yml` but commented out. It will be activated once the PyPI project is configured.

---

## Version Format

| Scope | Format | Example |
|-------|--------|---------|
| Global BPS suite | `MM.PP` (Major.Patch) | `5.0` |
| Per-module tag | `bps-<module>-X.Y.Z` (semver) | `bps-l2b_fh_processor-5.0.0` |

Version updates across all files are automated via `nox -s version_update`.

---

## SBOM (Software Bill of Materials)

The CycloneDX SBOM lists all project dependencies and their licenses. It is generated automatically at each release and attached to the GitHub Release artefacts. This supports ESA traceability and open science requirements.

---

## Pre-Release Checklist

- [ ] PR opened from `develop` to `release`
- [ ] `VERSION` updated via `nox -s version_update`
- [ ] `CHANGELOG.md` updated with all changes for this release
- [ ] Heavy CI dispatched on the `develop → release` PR with `run_heavy=true`
- [ ] `CI gate` green on the `develop → release` PR (baseline + extended + heavy)
- [ ] 2 approvals obtained on the `develop → release` PR (Maintainer + SME)
- [ ] PR opened from `release` to `main`
- [ ] ESA reviewer approval obtained on `VERSION` + `CHANGELOG.md`
- [ ] 3 approvals obtained on the `release → main` PR (Maintainer + SME + ESA)
- [ ] Commits signed with GPG/SSH on `release` and `main`
- [ ] Tag created in the correct format (`bps-<module>-X.Y.Z`) after merge to `main`
- [ ] `release.yml` pipeline completed successfully
- [ ] GitHub Release visible with correct artefacts

---

**Last Updated:** 2026

---

**Previous:** [Becoming a maintainer](becoming-a-maintainer.md) | **Next:** [BioPAL Code of Conduct](code-of-conduct.md)
