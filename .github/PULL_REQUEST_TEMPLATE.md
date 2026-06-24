<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0

Thanks for contributing to BIOMASS BPS. Before opening this PR, make sure a
tracking issue exists, opened with one of the five issue templates. The issue
captures the context (component, motivation, scientific justification). This
PR describes the change (what is in the diff).
-->

## Linked issue

<!--
The linked issue must already exist in the backlog and have been triaged.
A PR opened against a non-triaged or rejected issue will not be reviewed.
-->

Closes #

- [ ] The linked issue is labelled `status:approved`, `good-first-issue` or `help-wanted`. 
Being open in the backlog is not enough on its own: only these three labels mean the scope has been triaged and approved for implementation.

## What this PR changes

<!--
Three to five bullets describing the diff in plain language.
Do not re-describe the problem, that lives in the linked issue.
-->

*
*
*

## Notes for reviewers

<!--
Optional. Anything reviewers should look at carefully: a non-obvious design
choice, a regression risk you mitigated, an area you would like a second
opinion on, a known limitation deferred to a follow-up issue.
Delete this section if there is nothing to flag.
-->

## User-facing change

- [ ] This PR introduces a user-facing change (API, CLI, output format, performance, documentation).

If yes, write one short release-note sentence here (it will be picked up in `CHANGELOG.md`):


## Documentation

- [ ] This PR modifies the documentation (Sphinx site, README, wiki, ATBD, Science Guide).
- [ ] This PR does not modify the documentation, and no documentation update is needed.
- [ ] This PR does not modify the documentation, but a documentation update is needed and tracked in issue #_____.

## Testing

- [ ] Unit tests cover new or changed logic
- [ ] Integration or workflow tests added where relevant
- [ ] Scientific regression handled (`baseline` / `extended` / `heavy` updated if applicable)
- [ ] Coverage ≥ 60% on touched code; all relevant tests pass locally
- [ ] Tests are independent; test data is included or documented

## Tier rationale

The CI computes the tier automatically from the diff against the base branch. You do not assign it, but stating your expectation helps reviewers spot a mismatch quickly.

| Tier | Triggers | Checks that run |
|---|---|---|
| **0** | Routine changes, no sensitive path touched | Baseline only |
| **1** | Locked paths, SME-owned paths, marker fail, Dependabot major | Baseline + Extended |
| **2** | `VERSION` promoted to `main`, designated heavy paths, manual `run_heavy` | Baseline + Extended + Heavy |

Full rules: [`.github/tier-policy.yml`](.github/tier-policy.yml). Background: [Contribution tiers in the contributor guide](../../wiki/CONTRIBUTING_PART1#contribution-tiers).

* Expected tier: <!-- 0 / 1 / 2 -->
* Why:

## AI assistance disclosure

<!--
Following the practice established by NumPy, xarray and others, we ask
contributors to disclose AI assistance. This is informational, not gating.
-->

- [ ] No AI tools were used to prepare this PR.
- [ ] AI tools were used. Tool(s):  &nbsp;<!-- e.g. Copilot, Claude, ChatGPT, Codex -->
      <br>What was generated (code, tests, documentation, commit messages):
      <br>I have reviewed the generated content and take responsibility for it.

## Checklist

- [ ] This PR closes exactly one tracking issue, linked above.
- [ ] The scope of the diff matches the approved scope in the linked issue. No drift, no extras.
- [ ] A breaking change is explicitly flagged in the release note sentence above (if applicable).
- [ ] The reviewer assigned has the relevant domain knowledge for this change.
