# CI automation and contribution tiers

Reference for how pull requests are classified, which CI stages run,
and which branch rules apply. For what to do once your PR is open, see
[Review and integration](review-and-integration.md).

## Verify every change, automatically

Every contribution is risk-scored before any human reviews it. The
classifier reads its policy from the base branch so the PR cannot
rewrite its own judge.

::::{grid} 2 2 4 4
:gutter: 2

:::{grid-item-card} 10
:class-card: sd-text-center sd-fs-2

Parallel baseline checks per PR.

`REUSE · DCO · Bandit · pre-commit · build · tests · docs · ...`
:::

:::{grid-item-card} 3
:class-card: sd-text-center sd-fs-2

Automated tier levels.

`Tier 0 · Tier 1 · Tier 2`, computed from the diff.
:::

:::{grid-item-card} 1
:class-card: sd-text-center sd-fs-2

Single source of truth.

`tier-policy.yml`, read from the base branch, never from the PR.
:::

:::{grid-item-card} 100 %
:class-card: sd-text-center sd-fs-2

Auto-classified changes.

No labels to set. The CI scores every PR before any human reviews it.
:::

::::

---

## Contribution tiers in one paragraph

Every pull request is automatically classified into one of three tiers
based on the diff it introduces. The tier determines which CI stages
run and how many approvals are required to merge.

- **Tiers 0, 1 and 2** are computed automatically by the CI from the
  policy declared in `.github/tier-policy.yml`. The contributor does
  not assign the tier; the CI does.
- **Tier 3** is a separate qualitative governance category for release
  decisions and policy changes that go through an ESA process outside
  the automated pipeline.

---

## The three automated tiers

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} 🟢 Tier 0 &middot; Baseline
No risk-elevating change detected.

**Triggers**
- No `locked_paths` modified.
- No SME-owned paths touched.
- Baseline marker unchanged.

**Approvals**: 1 maintainer.
**CI**: Baseline gate only.

*Examples*: bug fixes, doc updates, refactors without scientific impact,
typo corrections.
:::

:::{grid-item-card} 🟡 Tier 1 &middot; Extended
Locked or SME-owned paths modified.

**Triggers**
- Touches `locked_paths`.
- Touches SME-owned paths.
- Baseline marker differs.
- Dependabot major version bump.

**Approvals**: 2 (incl. SME).
**CI**: Baseline + Extended.

*Examples*: small parameter updates, minor algorithm changes,
CI configuration changes.
:::

:::{grid-item-card} 🔴 Tier 2 &middot; Heavy
Release branch or explicit upclass.

**Triggers**
- PR targets the `release` branch.
- `run_heavy=true` requested on dispatch.
- Manual upclass from Tier 1.

**Approvals**: 3 (incl. SME and ESA).
**CI**: Baseline + Extended + Heavy.

*Examples*: new retrieval approaches, significant changes to error
models, anything promoted to `release`.
:::

::::

### Tier 3 &middot; governance

Tier 3 is a separate path for **release decisions and policy changes**.
It is not computed by the CI. These changes are approved through an
explicit ESA decision in consultation with the governance group, then
reflected in the repository through the standard branching mechanism
once the decision has been recorded.

*Examples*: new processor version releases, governance changes, major
policy updates.

### Judge from base, never from PR

```{important}
🔒 **The tier policy is read from the base branch SHA, never from the PR
head.** The policy file, the baseline references, and the test harness
are all locked at the base commit. A pull request cannot rewrite its
own judge.
```

This is the core security guarantee of the CI: every PR is evaluated
against the rules in effect on the branch it targets, not against the
rules it tries to introduce.

### At a glance

| | Tier 0 &middot; Baseline | Tier 1 &middot; Extended | Tier 2 &middot; Heavy |
|---|---|---|---|
| **CI stages** | Baseline | Baseline + Extended | Baseline + Extended + Heavy |
| **Approvals** | 1 maintainer | 2 (incl. SME) | 3 (incl. SME + ESA) |
| **Expected volume** | 60–70% of PRs | 20–25% of PRs | 5–10% of PRs |
| **Typical timeline** | under 3 days | 5 to 7 days | 10 to 14 days |
| **Review cadence** | Weekly | Monthly | Quarterly |

---

## Pipeline overview

When a pull request is opened, GitHub triggers `ci.yml`. The workflow runs
baseline checks in parallel, classifies the tier from the base-branch policy,
then runs Extended and Heavy stages when required.

```{raw} html
<img alt="CI/CD Workflow" src="../images/CI-CD-Workflow.drawio.svg" class="only-light">
<img alt="CI/CD Workflow" src="../images/CI-CD-Workflow_Shadow.drawio.svg" class="only-dark">
```

### Diagram appendix

```{raw} html
<img alt="CI/CD Annexe" src="../images/CI-CD-Annexe_Light.drawio.svg" class="only-light">
<img alt="CI/CD Annexe" src="../images/CI-CD-Annexe_Shadow.drawio.svg" class="only-dark">
```

### Pull request sequence

```{mermaid}
%%{init: {'theme':'base', 'themeVariables': {'primaryColor':'#f5f5f5','primaryTextColor':'#333','primaryBorderColor':'#9e9e9e','lineColor':'#666','secondaryColor':'#f5f5f5','tertiaryColor':'#f5f5f5'}}}%%
sequenceDiagram
    participant Dev as Developer
    participant Fork as Fork (external only)
    participant GH as GitHub PR
    participant CI as CI/CD (ci.yml)
    participant Bot as PR Guidance Bot
    participant Rev as Reviewer(s)
    participant Develop as develop

    alt Internal contributor
        Dev->>GH: Push feature branch · open PR → develop
    else External contributor
        Dev->>Fork: Push feature branch on fork
        Fork->>GH: Open PR from fork → upstream develop
    end

    GH->>CI: pull_request event triggers ci.yml

    CI->>CI: Baseline checks in parallel<br/>(DCO · REUSE · pre-commit · bandit · build · unit tests · docs)

    alt Baseline failure
        CI->>GH: baseline-gate ✗
        GH->>Dev: Fix failing job and re-push
        Dev->>GH: Push fix (or to fork)
        GH->>CI: Re-trigger ci.yml
    end

    CI->>CI: Automatic tier triage<br/>(read tier-policy.yml from base branch)

    alt Tier 1: SME-owned paths modified
        CI->>CI: Extended pipeline (pytest -m extended)
    end

    alt Tier 2: locked paths modified
        CI->>CI: Extended + Heavy pipeline (pytest -m heavy)
    end

    CI->>GH: CI gate ✓
    CI->>Bot: Post/update sticky PR comment
    Bot->>GH: Summary: tier · jobs status · reviewers needed

    Rev->>GH: Review and comments
    Dev->>GH: Address comments · re-push
    GH->>CI: Re-trigger ci.yml

    Rev->>GH: Approve (N approvals per branch ruleset)
    GH->>Develop: Squash merge
```

---

## Branch protection and approvals

Four branches, three protection levels, one promotion path. Every
branch is governed by GitHub repository rulesets, never by ad-hoc
settings.

::::{grid} 1 2 2 4
:gutter: 2

:::{grid-item-card} Fork &middot; `feature/*`
:class-card: sd-border-light

**Personal**

Free editing. No approval required. This is where you build the change
before it touches the project.
:::

:::{grid-item-card} `develop`
:class-card: sd-border-info

**Integration**

1 approval &middot; Maintainer + SME review &middot; squash only &middot;
`CI gate` required.
:::

:::{grid-item-card} `release`
:class-card: sd-border-warning

**Pre-release**

2 approvals &middot; Maintainer + SME &middot; GPG/SSH signed commits
&middot; squash only &middot; Heavy CI mandatory.
:::

:::{grid-item-card} `main`
:class-card: sd-border-danger

**Production**

3 approvals &middot; Maintainer + SME + ESA &middot; no admin bypass
&middot; signed commits &middot; squash only.
:::

::::

### Rules enforced everywhere

- ✓ Require CODEOWNERS review.
- ✓ Require a linear history (squash-only merges).
- ✓ Dismiss stale approvals on every new commit.
- ✓ Block force-pushes.
- ✓ `CI gate` is the required status check on every protected branch.

```{important}
**PRs targeting `release` need explicit Heavy authorisation.** The
`CI gate` stays red on every `pull_request` event until a maintainer
triggers the workflow via `Actions → Run workflow` with
`run_heavy=true`. This is the explicit consent step before promotion
to `main`.

Any modification to `VERSION` or `CHANGELOG.md` requires an explicit
approval from the ESA reviewer defined in `CODEOWNERS`. No release
can be merged without this approval.
```

### Branching strategy

- **`main`**: Operational/production branch. Promoted to from `release`
  after ESA approval.
- **`release`**: Release candidate branch. Pre-release validation runs
  here; Heavy CI is mandatory before promotion to `main`.
- **`develop`**: Main development branch. Latest development state.
- **Feature branches**: `feature/description`, `bugfix/issue-id`,
  `docs/topic`. Opened from `develop` on a fork.

## Dependabot: automated dependency updates

Dependabot opens PRs every Monday to update pip and GitHub Actions dependencies, always targeting `develop`. These PRs go through the normal CI pipeline.

**Important rule**: a **major version bump** on a pip dependency is automatically classified as **Tier 1** by the CI, requiring SME review.

---

## Automated CI/CD checks

All pull requests go through automated checks. The required checks and approvals depend on the contribution tier.

**Baseline Pipeline (all PRs: 10 parallel jobs):**
- `baseline-marker-signal`: `pytest -m baseline` on `test/baseline/` (feeds the tier decision)
- `baseline-dco`: Signed-off-by trailer on every non-merge commit, identity must match author or committer
- `baseline-reuse`: REUSE / SPDX header compliance
- `baseline-pre-commit`: black, ruff, mypy, detect-secrets hooks
- `baseline-security`: Bandit static analysis
- `baseline-build`: sdist + wheel build (non-blocking during migration)
- `baseline-sensitive-files`: rejects `*.bak` files and files larger than 10 MB
- `baseline-unit-tests`: `pytest -m unit` (non-blocking during migration)
- `baseline-docs`: Sphinx build if `docs/api/conf.py` exists (non-blocking)
- `baseline-dependabot`: verifies `.github/dependabot.yml` exists and signals major bumps

All 10 jobs feed `baseline-gate`, the aggregate blocking check.

**Extended Pipeline (Tier 1+):**
- Extended test suite (`pytest -m extended`)
- ⚠ Requires TDS dataset: see [Code standards](code-standards.md) for access procedure

**Heavy Pipeline (Tier 2):**
- Full scientific regression (`pytest -m heavy`)
- ⚠ Requires TDS dataset: see [Code standards](code-standards.md) for access procedure

### Automatic tier detection

The CI/CD pipeline **automatically determines the tier** by analysing the files modified in your PR against the policy defined in `.github/tier-policy.yml`. You do not need to add a label manually.

**How the tier is decided:**

- Files listed in `locked_paths` (e.g. `VERSION`, `pyproject.toml`, `.github/workflows/**`, `CODEOWNERS`, test directories) → **Tier 2** automatically
- Files in `sme_owned_paths` (processor directories `bps-*`) → triggers SME review
- A Dependabot PR with a **major version bump** → **Tier 1** automatically
- Everything else → **Tier 0**

The policy file is always read from the **base branch** (not your PR head), which prevents a PR from modifying its own judge.

**What runs based on the detected tier:**

- **Tier 0**: Baseline pipeline only (DCO, REUSE, pre-commit, security, build, unit tests, docs)
- **Tier 1**: Baseline + Extended pipeline (`pytest -m extended`)
- **Tier 2**: Baseline + Extended + Heavy pipeline (`pytest -m heavy`)
- **Tier 3**: ESA governance decision: no automated pipeline

### Review and approval requirements

- **Tier 0**: At least **1 core maintainer** approval
- **Tier 1**: **Scientific module expert** approval + **Core maintainer** approval
- **Tier 2**: **Scientific module expert(s)** approval + **Open Science Lead** approval + **ESA representative(s)** approval + **Core maintainer** approval
- **Tier 3**: Explicit ESA decision in consultation with governance group

**The PR cannot be merged until all required checks pass and all required approvals are obtained.**

Full policy rules: [`.github/tier-policy.yml`](https://github.com/BioPAL/BPS/blob/main/.github/tier-policy.yml).

---

**Previous:** [Review and integration](review-and-integration.md) | **Next:** [Quality and validation](quality-and-validation.md)
