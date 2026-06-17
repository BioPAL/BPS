# Implementation

**Stage 2 of the contribution workflow.** Once your issue carries an approval
label, fork the repository, implement the change, run local checks, sign your
commits, and open a pull request.

For step-by-step commands, see [Practical workflow](practical-workflow.md).

---

```{raw} html
<img alt="Contribution Process" src="../images/CI-CD-Contribution_Light.drawio.svg" class="only-light">
<img alt="Contribution Process" src="../images/CI-CD-Contribution_Shadow.drawio.svg" class="only-dark">
```

## Five steps from approved issue to live PR

You start with an **approved issue in hand** and end with a **pull
request live on GitHub** that the CI has begun classifying. Each step
is small; together they keep the diff focused and the CI stable.

::::{grid} 1 1 2 5
:gutter: 2

:::{grid-item-card} 01 &middot; Fork and branch
:class-card: sd-border-info

Fork the repository on GitHub and create a feature branch from an
up-to-date `develop`. Branch prefix: `feature/`, `bugfix/`, or `docs/`.
:::

:::{grid-item-card} 02 &middot; Implement and test
:class-card: sd-border-info

Code the change inside the approved scope. Add or update tests so the
behaviour is exercised at the right tier (`unit`, `baseline`,
`extended`).
:::

:::{grid-item-card} 03 &middot; Local checks
:class-card: sd-border-info

Run `ruff`, `mypy`, and `pytest -m unit` locally. Add `baseline` if you
expect the marker output to move. Save the CI a round-trip.
:::

:::{grid-item-card} 04 &middot; Commit, signed off
:class-card: sd-border-info

Every commit needs a `Signed-off-by:` trailer for the DCO. Group
related changes into atomic commits with clear messages.
:::

:::{grid-item-card} 05 &middot; Open the pull request
:class-card: sd-border-success

Push the branch and open a PR against `develop` that links the issue
with `Closes #N`. The CI starts and the tier is computed from the diff.
:::

::::

```{tip}
**Local mirror of the CI gate.** Running
`ruff check . && mypy . && pytest -m unit` before pushing catches more
than 90 % of CI failures.
```

## Workflow conventions

- **Start from develop**: always create feature branches from an
  up-to-date `develop`.
- **One feature per branch**: keep changes focused and atomic.
- **Stay in sync**: rebase or merge `develop` regularly to avoid
  surprise conflicts.
- **Commit small and often**: small, logical commits make review and
  bisect much easier.
- **Mirror the CI locally**: run `pre-commit`, `ruff`, `mypy`, and the
  unit suite before pushing.

## DCO: Developer Certificate of Origin

Every commit (excluding merge commits) must carry a `Signed-off-by:` trailer. This is a legal statement that you wrote the code and have the right to contribute it under the project's license.

```bash
# Sign a commit
git commit -s -m "feat: my contribution"

# Enable automatic signing for all commits
git config format.signoff true
```

Remediation if you forgot:
```bash
git commit --amend --signoff
git push --force-with-lease
```

The `baseline-dco` CI job blocks any PR containing an unsigned commit.

---

**Previous:** [Proposal and approval](proposal-and-approval.md) | **Next:** [Review and integration](review-and-integration.md)
