<!--
SPDX-FileCopyrightText: 2026 European Space Agency (ESA)
SPDX-License-Identifier: Apache-2.0
-->

# Getting started

Welcome to **BIOMASS BPS**, the open source Processing Suite for the
ESA BIOMASS Earth Explorer mission. This page is the front door:
it points you to the right resources in less than thirty seconds,
whether you want to use BPS, discuss it, or contribute to it.

## Find your path

Choose the path that matches your goal. Each branch links to the
resources described in the sections below.

```{mermaid}
flowchart TD
    start([What brings you here?])

    start --> u_start
    start --> d_start
    start --> c_start

    subgraph use_path [Use BPS]
        direction TB
        u_start[Run processors and products]
        u_guides[User Guide · Science Guide · Tutorials]
        u_docs[Applicable documents]
        u_start --> u_guides --> u_docs
    end

    subgraph discuss_path [Ask or discuss]
        direction TB
        d_start[GitHub Discussions]
        d_cats["Q&A · Ideas · Scientific · Governance"]
        d_start --> d_cats
    end

    subgraph contrib_path [Contribute]
        direction TB
        c_start[Open or select an issue]
        c_gate{Approval label assigned?}
        c_wait[Wait for maintainer triage]
        c_impl[Implement on a signed branch]
        c_pr[Open pull request]
        c_review[CODEOWNERS review and CI]
        c_merge[Squash merge]
        c_start --> c_gate
        c_gate -->|no| c_wait --> c_start
        c_gate -->|yes| c_impl --> c_pr --> c_review --> c_merge
    end
```

---

## I want to use BPS

You want to run a processor, understand a data format, or read the
science behind an algorithm.

::::{grid} 1 2 2 2
:gutter: 3
:class-container: intro-grid

:::{grid-item-card}
:link: ../user-guide/index
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-regular fa-book-open fa-fw" aria-hidden="true"></i> **User Guide**
^^^
Software User Manual (SUM) and authoritative user reference.
:::

:::{grid-item-card}
:link: ../science-guide/index
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-solid fa-satellite fa-fw" aria-hidden="true"></i> **Science Guide**
^^^
ATBDs and Product Format Documents for L1, L2a, and L2b products.
:::

:::{grid-item-card}
:link: ../tutorials/index
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-solid fa-flask fa-fw" aria-hidden="true"></i> **Tutorials**
^^^
Walkthroughs and worked examples.
:::

:::{grid-item-card}
:link: ../about/applicable-documents
:link-type: doc
:class-card: intro-card

<i class="doc-icon fa-regular fa-clipboard fa-fw" aria-hidden="true"></i> **Applicable documents**
^^^
The full list of ATBDs, SUM, ICD, IODD and auxiliary specifications
with PDF download links.
:::

::::

For the published documentation portal, see
[biomass-disc.info](https://www.biomass-disc.info/docs).

---

## I have a question or an idea

Use [**GitHub Discussions**](https://github.com/BioPAL/BPS/discussions).
Open issues are reserved for actionable items; open Discussions for
everything else. Six categories help you find the right thread.

| Category | When to use |
|---|---|
| <i class="doc-icon fa-regular fa-circle-question fa-fw" aria-hidden="true"></i> [Q&A](https://github.com/BioPAL/BPS/discussions/categories/q-a) | Usage, installation, API, processing chain, data formats. Mark the helpful reply as the answer. |
| <i class="doc-icon fa-regular fa-lightbulb fa-fw" aria-hidden="true"></i> [Ideas](https://github.com/BioPAL/BPS/discussions/categories/ideas) | Brainstorm a feature or a change before opening a tracking issue. |
| <i class="doc-icon fa-solid fa-microscope fa-fw" aria-hidden="true"></i> [Scientific discussions](https://github.com/BioPAL/BPS/discussions/categories/scientific-discussions) | Algorithms, validation, methodology, ATBD interpretations, references. |
| <i class="doc-icon fa-solid fa-landmark fa-fw" aria-hidden="true"></i> [Governance](https://github.com/BioPAL/BPS/discussions/categories/governance) | Project governance, maintainer paths, policy. |
| <i class="doc-icon fa-regular fa-hand fa-fw" aria-hidden="true"></i> [Show and tell](https://github.com/BioPAL/BPS/discussions/categories/show-and-tell) | Introductions, usage stories, papers, conference talks. |
| <i class="doc-icon fa-solid fa-bullhorn fa-fw" aria-hidden="true"></i> [Announcements](https://github.com/BioPAL/BPS/discussions/categories/announcements) | Releases and governance decisions. Read only for external contributors. |

---

## I want to contribute

Every contribution to BIOMASS BPS follows the same five steps. **No code
is written before the issue has been approved.** This guardrail protects
contributors from wasted effort and protects the project from scope creep.

### Step 1. Open an issue (or pick an existing one)

Five [issue templates](https://github.com/BioPAL/BPS/issues/new/choose)
cover every actionable case:

| Template | When to use |
|---|---|
| <i class="doc-icon fa-solid fa-bug fa-fw" aria-hidden="true"></i> Bug report | A defect in a processor, the CI, or the documentation. |
| <i class="doc-icon fa-solid fa-wand-magic-sparkles fa-fw" aria-hidden="true"></i> Feature or enhancement request | A non-scientific feature or tooling improvement. |
| <i class="doc-icon fa-solid fa-flask fa-fw" aria-hidden="true"></i> Algorithm proposal | A new scientific algorithm or a methodological change. Justification required. |
| <i class="doc-icon fa-regular fa-file-lines fa-fw" aria-hidden="true"></i> Documentation issue | An error or a gap in the documentation. |
| <i class="doc-icon fa-solid fa-lock fa-fw" aria-hidden="true"></i> Security report | A non-sensitive security concern. Sensitive vulnerabilities go through a [private advisory](https://github.com/BioPAL/BPS/security/advisories/new) instead. |

If an issue that matches what you want to do already exists, pick that
one instead of opening a duplicate.

### Step 2. Wait for the approval label

An open issue is **not** an invitation to start coding. Wait until the
issue is labelled with one of these three:

- `status:approved`: triaged, scoped, ready to be implemented.
- `good-first-issue`: approved and suitable for newcomers.
- `help-wanted`: approved and the project actively welcomes external contributions on it.

Triage usually completes within five working days. Comment on the issue
to express interest; a maintainer will respond.

### Step 3. Code on a fork or feature branch

Once the issue is approved:

- Fork the repository (or branch directly if you are a maintainer).
- Use a conventional branch prefix: `feat/`, `fix/`, `docs/`, `chore/`, `ci/`, `refactor/`, `test/`.
- Sign off every commit (`git commit -s`) to satisfy the Developer Certificate of Origin.
- Touch only files inside the scope approved on the issue.

### Step 4. Open a pull request

Open the PR against `develop`. The PR template guides you through what
to fill in. Make sure `Closes #<issue-number>` appears in the
description so the issue closes automatically on merge.

### Step 5. Review and merge

A reviewer with the relevant domain knowledge is assigned through the
CODEOWNERS configuration. Once CI is green and the reviewer approves,
a maintainer merges with squash, and the linked issue closes.

For the long form, including environment setup, coding standards, the
DCO sign off mechanics, and the tier classification system, see the
[Contributing guide](../contributing/index.md) especially
[Proposal and approval](../contributing/proposal-and-approval.md),
[Implementation](../contributing/implementation.md), and
[Review and integration](../contributing/review-and-integration.md).

---

## Need more help?

- Read the [Contributing guide](../contributing/index.md) for the long form workflow.
- Check the [Communication page](../communication/index.md) for meeting schedules and community channels.
- Ask in [Q&A](https://github.com/BioPAL/BPS/discussions/categories/q-a) on GitHub Discussions.
