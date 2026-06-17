# Review and integration

**Stage 3 of the contribution workflow.** Once your pull request is open, CI runs
automatically and routes reviewers. Iterate until CI is green and all required
approvals are obtained, then the change is squash-merged into `develop`.

If CI blocks your PR or you need to understand tier classification, see
[CI automation and contribution tiers](ci-automation-and-contribution-tiers.md).

---

## CI green or CI red

The moment the pull request is opened the CI starts running. From there
the journey forks: a **green path** that ends with a released tag and a
DOI, or a **red path** that loops back to the contributor until the CI
turns green.

```{mermaid}
flowchart TD
    Open([PR opened]) --> CI{CI status}

    %% Path A · green
    CI -->|green| GreenPath
    subgraph GreenPath[Path A · CI green to release]
        direction TB
        Review[01 · Review<br/>SME · Maintainer<br/>ESA on Tier 2+]
        Approve[02 · Approve<br/>stale approvals dismissed on push]
        Merge[03 · Squash merge<br/>develop → release → main]
        Released[04 · Released<br/>tagged + GitHub Release + DOI]
        Review --> Approve --> Merge --> Released
    end

    %% Path B · red
    CI -->|red| RedPath
    subgraph RedPath[Path B · CI red, iterate, loop back]
        direction TB
        ReadLog[01 · Read the failing job log]
        Iterate[02 · Iterate<br/>address comments · push fixes<br/>sign each commit]
        Rerun[03 · Re-run<br/>stale approvals are dismissed]
        ReadLog --> Iterate --> Rerun
    end
    Rerun -.->|loop back| CI

    classDef green fill:#E8F5E9,stroke:#43A047,color:#000
    classDef red fill:#FFEBEE,stroke:#E53935,color:#000
    class GreenPath green
    class RedPath red
```

When CI fails, read the failing job log, fix the issue, and push again.
Each push re-triggers the pipeline and **dismisses any stale approval**.
The sticky comment from the guidance bot summarises the tier, the job
status, and the reviewers still required.

## Open and update your pull request

When you're ready to ask for a code review, file a pull request. Before you do, complete the [pre-submission checklist](templates-and-checklists.md#before-submitting-a-pr)).

1. **Navigate to your repository on GitHub** at https://github.com/your-username/BPS (or the main repository if working directly)
2. **Click on "Branches"**
3. **Click on the "Compare" button** for your feature branch
4. **Select the "base" and "compare" branches**, if necessary. This will be `develop` and `feature/your-feature-name`, respectively.

To submit a pull request:

1. **Navigate to your repository on GitHub**
2. **Click on the "Pull Request" button**
3. **Click on "Commits" and "Files Changed"** to verify the diff one last time
4. **Write a description** following the [PR template](https://github.com/BioPAL/BPS/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
5. **Click "Create Pull Request"**

### Using draft pull requests

If you don't think your request is ready to be merged, use GitHub's **Draft PR** feature:

- Indicate that the code is still work-in-progress
- Allow reviewers to provide early feedback
- Don't block other work or confuse reviewers about readiness
- Can be converted to a regular pull request when ready

Mention anything you'd like particular attention for, such as a complicated change or some code you are not happy with.

If you need to make more changes after submitting your PR, push to your branch. The pull request updates automatically and CI re-runs. See [Practical workflow](practical-workflow.md#updating-your-pull-request)) for commands.

## Accepting an intentional baseline change

If your PR intentionally changes processor output (e.g. a scientific fix or new algorithm), the baseline check will report a difference and CI may elevate the PR to a higher tier. See [CI automation and contribution tiers](ci-automation-and-contribution-tiers.md#automatic-tier-detection)) for how that works.

To get the change merged:

1. Confirm with the relevant SME (auto-assigned via `CODEOWNERS`) that the change is correct.
2. Update `test/baseline/` reference outputs in the same PR so the new behaviour becomes the new baseline.
3. The SME reviews the diff (code + baseline update) and approves via a standard GitHub review.

There is no `baseline:accepted` label any more; SME approval flows through the native CODEOWNERS-required review on the PR.

## Responding to reviews

- Address all reviewer comments promptly (< 3 days)
- Update the PR with requested changes
- CI re-runs automatically on each push
- Iterate until all approvals are obtained

Release-related files such as `VERSION` and `CHANGELOG.md` require ESA approval via `CODEOWNERS`. See [branch protection rules](ci-automation-and-contribution-tiers.md#branch-protection-rules)) for the full policy.

## After merge

Your pull request is squash-merged into `develop`, the linked issue closes automatically, and the change follows the normal release path when promoted.

You may delete your feature branch. See
[Practical workflow](practical-workflow.md#delete-your-merged-branch-optional)) for commands.

---

**Previous:** [Implementation](implementation.md) | **Next:** [CI automation and contribution tiers](ci-automation-and-contribution-tiers.md)
