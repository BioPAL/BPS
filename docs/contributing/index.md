<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Contributing

Thank you for your interest in contributing to the **BIOMASS Processing
Suite (BPS)**. We welcome contributions from the scientific community,
industrial partners, and external developers.

## How a contribution flows

Every contribution follows the same five steps. **No code is written
before the issue has been approved.** This is a deliberate guardrail
that protects contributors from wasted effort and protects the project
from scope creep.

::::{grid} 1 1 1 1
:gutter: 2

:::{grid-item-card} 1 &middot; Open an issue
:class-card: sd-border-success

Pick one of the five templates: bug report, feature or enhancement
request, algorithm proposal, documentation issue, or security report.
:::

:::{grid-item-card} 2 &middot; Triage
:class-card: sd-border-success

A maintainer reviews the issue and applies routing labels:
`needs-triage`, `needs-sme`, or `needs-discussion`.
:::

:::{grid-item-card} 3 &middot; Approval (the gate)
:class-card: sd-border-warning

The issue is labelled `status:approved`, `good-first-issue`,
or `help-wanted`. **Until one of these labels is set, do not code.**
:::

:::{grid-item-card} 4 &middot; Code and open a PR
:class-card: sd-border-success

Fork, branch, commit with `Signed-off-by` (DCO), open a pull request
that links the issue.
:::

:::{grid-item-card} 5 &middot; Review and merge
:class-card: sd-border-success

CODEOWNERS review, CI green, squash merge. The linked issue closes
automatically.
:::

::::

For questions or open-ended discussions that are not yet a concrete
issue, use [GitHub Discussions](https://github.com/BioPAL/BPS/discussions).
Issues are reserved for actionable items.

## The three stages

```{mermaid}
flowchart TD
    subgraph Before["1. Proposal and approval"]
        direction TB
        Backlog["Backlog"]
        PathA["Path A<br/>Pick an approved issue"]
        PathB["Path B<br/>Propose a new issue"]
        Gate{"Wait for the label<br/><code>status:approved</code><br/><code>good-first-issue</code><br/><code>help-wanted</code>"}
        Backlog --> PathA
        Backlog --> PathB
        PathA --> Gate
        PathB --> Gate
    end

    subgraph Building["2. Implementation"]
        direction TB
        Fork["Fork &amp; branch"]
        Implement["Implement + tests"]
        Local["Local checks<br/>ruff · mypy · pytest"]
        Commit["Commit signed off"]
        OpenPR["Open the PR"]
        Fork --> Implement --> Local --> Commit --> OpenPR
    end

    subgraph After["3. Review and integration"]
        direction TB
        CI{"CI status"}
        Review["Review"]
        Merge["Squash merge"]
        Released["Released<br/>tag + DOI"]
        Iterate["Iterate<br/>push fixes"]
        CI -->|green| Review
        Review --> Merge
        Merge --> Released
        CI -->|red| Iterate
        Iterate --> CI
    end

    Gate -->|approved| Fork
    OpenPR --> CI

    classDef gate fill:#FFE082,stroke:#FFB300,color:#000
    class Gate gate
```

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} 1. Proposal and approval
:link: proposal-and-approval
:link-type: doc

Backlog, issue templates, triage, and the approval gate. Do not code until the issue is approved.
:::

:::{grid-item-card} 2. Implementation
:link: implementation
:link-type: doc

Fork, implement, local checks, DCO-signed commits, and open the pull request.
:::

:::{grid-item-card} 3. Review and integration
:link: review-and-integration
:link-type: doc

CI green/red loop, review, baseline changes, merge, and release path.
:::

::::

## Foundations

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} 📜 Licensing
:link: ../about/licensing/index
:link-type: doc

Apache 2.0 terms, REUSE compliance, SPDX headers, and dependency licence
requirements. Every contribution must comply.
:::

:::{grid-item-card} Code of Conduct
:link: code-of-conduct
:link-type: doc

Community standards that govern interactions inside the project.
Required reading for all contributors.
:::

:::{grid-item-card} Governance
:link: ../governance/index
:link-type: doc

Roles (Maintainer, SME, ESA), decision authority, and the chain of
approvals. Who decides what, and when.
:::

::::

## Quick reference

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} CI automation and contribution tiers
:link: ci-automation-and-contribution-tiers
:link-type: doc

Tier classification, CI job catalog, branch protection, and judge-from-base policy.
:::

:::{grid-item-card} Templates and checklists
:link: templates-and-checklists
:link-type: doc

Issue and PR templates, CODEOWNERS routing, and pre-submission checklists.
:::

:::{grid-item-card} Quality and validation
:link: quality-and-validation
:link-type: doc

Testing, scientific validation, backwards compatibility, and documentation expectations.
:::

:::{grid-item-card} Practical workflow
:link: practical-workflow
:link-type: doc

Environment setup, Git commands, and code examples for tests and standards.
:::

:::{grid-item-card} Becoming a maintainer
:link: becoming-a-maintainer
:link-type: doc

Path from contributor to maintainer: stages, qualities, and responsibilities.
:::

:::{grid-item-card} Community
:link: ../communication/index
:link-type: doc

Meetings, GitHub Discussions categories, workshops, and communication channels.
:::

:::{grid-item-card} Release process
:link: release-process
:link-type: doc

Maintainer runbook: develop to release to main, versioning, and ESA gate.
:::

::::

## Technical reference

::::{grid} 1 3 3 3
:gutter: 3

:::{grid-item-card} Architecture
:link: architecture
:link-type: doc

Monorepo layout, `bps-*` modules, dependency graph, and processor structure.
:::

:::{grid-item-card} Code standards
:link: code-standards
:link-type: doc

Naming, formatting, type hints, tests, error handling, and logging.
:::

:::{grid-item-card} Documentation standards
:link: documentation-standards
:link-type: doc

Docstrings, writing conventions, and documentation update expectations.
:::

:::{grid-item-card} 📄 Interface specifications
:link: ../about/applicable-documents
:link-type: doc

Official ICD, IODD, and auxiliary product format PDFs for integrators.
:::

::::

---

**Need help?** Use [GitHub Discussions](https://github.com/BioPAL/BPS/discussions).
The [Q&A category](https://github.com/BioPAL/BPS/discussions/categories/q-a)
is the recommended place for usage questions, and
[Scientific discussions](https://github.com/BioPAL/BPS/discussions/categories/scientific-discussions)
is the right space for algorithm and methodology questions.

```{toctree}
:caption: Contribution journey
:maxdepth: 1
:hidden:

Proposal and approval <proposal-and-approval>
Implementation <implementation>
Review and integration <review-and-integration>
```

```{toctree}
:caption: Automation and quality
:maxdepth: 1
:hidden:

CI automation and contribution tiers <ci-automation-and-contribution-tiers>
Quality and validation <quality-and-validation>
```

```{toctree}
:caption: Technical reference
:maxdepth: 1
:hidden:

Architecture <architecture>
Code standards <code-standards>
Documentation standards <documentation-standards>
```

```{toctree}
:caption: Workflow and templates
:maxdepth: 1
:hidden:

Practical workflow <practical-workflow>
Templates and checklists <templates-and-checklists>
```

```{toctree}
:caption: Policy and maintainers
:maxdepth: 1
:hidden:

Becoming a maintainer <becoming-a-maintainer>
Release process <release-process>
BioPAL Code of Conduct <code-of-conduct>
```
