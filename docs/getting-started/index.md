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

```{mermaid}
flowchart TD
    Start["👋 Welcome to BIOMASS BPS<br/>What brings you here?"]

    Start --> Use["🚀 I want to <b>use</b> BPS"]
    Start --> Discuss["💬 I have a question<br/>or an idea"]
    Start --> Contribute["🛠️ I want to <b>contribute</b><br/>code or docs"]

    Use --> UseResources["Tutorials<br/>User Guide<br/>Science Guide<br/>Applicable documents (PDFs)"]

    Discuss --> DiscussionsHub["GitHub Discussions"]
    DiscussionsHub --> CatQA["❓ Q&A<br/>usage questions"]
    DiscussionsHub --> CatIdeas["💡 Ideas<br/>brainstorm before an issue"]
    DiscussionsHub --> CatSci["🔬 Scientific discussions<br/>algorithms, methodology"]
    DiscussionsHub --> CatGov["🏛️ Governance"]

    Contribute --> IssueGate["1. Pick an approved issue<br/>or open a new one"]
    IssueGate --> Labels["Wait for the label<br/><code>status:approved</code><br/><code>good-first-issue</code><br/><code>help-wanted</code>"]
    Labels --> Code["2. Fork, branch, code<br/>(commits signed off)"]
    Code --> PR["3. Open a Pull Request<br/>linking the issue"]
    PR --> Review["4. CODEOWNERS review<br/>+ CI green, then squash merge"]

    classDef start fill:#347891,stroke:#347891,color:#fff
    classDef use fill:#9EBE3D,stroke:#9EBE3D,color:#fff
    classDef discuss fill:#5A9CB5,stroke:#5A9CB5,color:#fff
    classDef contribute fill:#FF7E79,stroke:#FF7E79,color:#fff
    classDef gate fill:#FFE082,stroke:#FFB300,color:#000

    class Start start
    class Use,UseResources use
    class Discuss,DiscussionsHub,CatQA,CatIdeas,CatSci,CatGov discuss
    class Contribute,Code,PR,Review contribute
    class IssueGate,Labels gate
```

---

## 🚀 I want to use BPS

You want to run a processor, understand a data format, or read the
science behind an algorithm.

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} 📘 User Guide
:link: ../user-guide/index
:link-type: doc

Software User Manual (SUM) and authoritative user reference.
:::

:::{grid-item-card} 🛰️ Science Guide
:link: ../science-guide/index
:link-type: doc

ATBDs and Product Format Documents for L1, L2a, and L2b products.
:::

:::{grid-item-card} 🧪 Tutorials
:link: ../tutorials/index
:link-type: doc

Walkthroughs and worked examples.
:::

:::{grid-item-card} 📋 Applicable documents
:link: ../about/applicable-documents
:link-type: doc

The full list of ATBDs, SUM, ICD, IODD and auxiliary specifications
with PDF download links.
:::

::::

For the published documentation portal, see
[biomass-disc.info](https://www.biomass-disc.info/docs).

---

## 💬 I have a question or an idea

Use [**GitHub Discussions**](https://github.com/BioPAL/BPS/discussions).
Open issues are reserved for actionable items; open Discussions for
everything else. Six categories help you find the right thread.

| Category | When to use |
|---|---|
| ❓ [Q&A](https://github.com/BioPAL/BPS/discussions/categories/q-a) | Usage, installation, API, processing chain, data formats. Mark the helpful reply as the answer. |
| 💡 [Ideas](https://github.com/BioPAL/BPS/discussions/categories/ideas) | Brainstorm a feature or a change before opening a tracking issue. |
| 🔬 [Scientific discussions](https://github.com/BioPAL/BPS/discussions/categories/scientific-discussions) | Algorithms, validation, methodology, ATBD interpretations, references. |
| 🏛️ [Governance](https://github.com/BioPAL/BPS/discussions/categories/governance) | Project governance, maintainer paths, policy. |
| 👋 [Show and tell](https://github.com/BioPAL/BPS/discussions/categories/show-and-tell) | Introductions, usage stories, papers, conference talks. |
| 📢 [Announcements](https://github.com/BioPAL/BPS/discussions/categories/announcements) | Releases and governance decisions. Read only for external contributors. |

---

## 🛠️ I want to contribute

Every contribution to BIOMASS BPS follows the same five steps. **No code
is written before the issue has been approved.** This guardrail protects
contributors from wasted effort and protects the project from scope creep.

### Step 1. Open an issue (or pick an existing one)

Five [issue templates](https://github.com/BioPAL/BPS/issues/new/choose)
cover every actionable case:

| Template | When to use |
|---|---|
| 🐛 Bug report | A defect in a processor, the CI, or the documentation. |
| ✨ Feature or enhancement request | A non-scientific feature or tooling improvement. |
| 🔬 Algorithm proposal | A new scientific algorithm or a methodological change. Justification required. |
| 📄 Documentation issue | An error or a gap in the documentation. |
| 🔒 Security report | A non-sensitive security concern. Sensitive vulnerabilities go through a [private advisory](https://github.com/BioPAL/BPS/security/advisories/new) instead. |

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
[Contributing guide](../contributing/index.md).

---

## Need more help?

- Read the [Contributing guide](../contributing/index.md) for the long form workflow.
- Check the [Communication page](../governance/communication.md) for meeting schedules and community channels.
- Ask in [Q&A](https://github.com/BioPAL/BPS/discussions/categories/q-a) on GitHub Discussions.

```{toctree}
:hidden:
:caption: Getting started

```
