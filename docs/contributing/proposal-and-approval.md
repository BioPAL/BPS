# Proposal and approval

**Stage 1 of the contribution workflow.** No code is written before the issue
has been approved. This gate protects contributors from wasted effort and
protects the project from scope creep.

For open-ended questions or ideas that are not yet actionable, use
[GitHub Discussions](https://github.com/BioPAL/BPS/discussions). Use issues
only for work that needs triage and tracking.

---

## Path A or Path B?

The contributor arrives with an interest in BIOMASS BPS. The first
decision is whether an issue already exists in the backlog or whether
one needs to be opened.

```{mermaid}
flowchart TD
    Contributor([Contributor with an idea or a problem])
    Backlog[(The backlog<br/>ready-to-pick issues)]

    Contributor -->|existing issue| Backlog
    Contributor -->|new idea| PathB1

    Backlog --> PathA
    subgraph PathA[Path A · Pick an existing issue]
        direction TB
        Cat1[good-first-issue]
        Cat2[help-wanted]
        Cat3[Bug fix]
        Cat4[Scientific feature]
    end
    PathA --> Gate

    subgraph PathBflow[Path B · Propose a new issue]
        direction TB
        PathB1[01 · Open the issue<br/>processor, case, expected impact]
        PathB2[02 · SME assignment<br/>auto-routed to the right expert]
        PathB3[03 · Scientific review<br/>paper, data, prior results]
        PathB4{04 · Go or no-go<br/>maintainer + SME<br/>ESA if required}
        PathB1 --> PathB2 --> PathB3 --> PathB4
    end

    PathB4 -->|iterate, more evidence| PathB3
    PathB4 -->|rejected| Closed[Issue closed]
    PathB4 -->|approved| Gate

    Gate{{"Approval gate<br/>status:approved · good-first-issue · help-wanted"}}
    Gate -->|approved, feeds backlog| Backlog
    Gate -->|picked up| Code[Stage 2 · Implementation]

    classDef gate fill:#FFE082,stroke:#FFB300,color:#000
    class Gate gate
```

### Path A: pick an existing issue

Browse the [Issues tab](https://github.com/BioPAL/BPS/issues) and look
only at issues that already carry one of the three approval labels:

- `status:approved`: triaged, scoped, ready to be picked up.
- `good-first-issue`: approved and explicitly suitable for newcomers.
- `help-wanted`: approved and the project actively welcomes external contributions.

An open issue without one of these labels is still under triage or
discussion. Comment on it to express interest; do not start coding yet.

### Path B: propose a new issue

If no approved issue matches what you want to work on, open a new one
using the appropriate
[issue template](https://github.com/BioPAL/BPS/issues/new/choose).
The proposal then walks through four steps:

::::{grid} 1 2 2 4
:gutter: 2

:::{grid-item-card} 01 &middot; Open the issue
State the processor concerned, the scientific or operational case,
and the expected impact.
:::

:::{grid-item-card} 02 &middot; SME assignment
The issue is automatically routed to the right Scientific Module Expert
based on CODEOWNERS rules.
:::

:::{grid-item-card} 03 &middot; Scientific review
The SME evaluates the evidence: peer-reviewed references, data,
prior results, comparison with alternatives.
:::

:::{grid-item-card} 04 &middot; Go or no-go
The maintainer and SME decide. Significant changes escalate to ESA.
The issue may iterate for more evidence, be rejected, or be approved.
:::

::::

Triage usually completes within five working days. Once scope is
settled, the issue receives `status:approved`, `good-first-issue`, or
`help-wanted` and joins the backlog ready for someone to pick.

Scientific proposals may require SME review and, for significant
changes, escalation to ESA. See
[Governance](../governance/roles-and-authority.md) for roles and
decision authority.

---

## Choose your issue type

```{mermaid}
flowchart TD
    Q([What are you reporting?]) --> Bug[Bug in processor / CI / docs]
    Q --> Feature[Non-scientific feature or tooling]
    Q --> Algo[Scientific algorithm or methodology change]
    Q --> Docs[Documentation error or gap]
    Q --> Sec[Security concern]
    Bug --> T1[01 Bug report template]
    Feature --> T2[02 Feature or enhancement template]
    Algo --> T3[03 Algorithm proposal template]
    Docs --> T4[04 Documentation issue template]
    Sec --> T5[05 Security report template]
```

| Template | When to use | Open |
|----------|-------------|------|
| **Bug report** | A defect in a processor, the CI, or the documentation | [Open](https://github.com/BioPAL/BPS/issues/new?template=01_bug_report.yml) |
| **Feature or enhancement** | A non-scientific feature or tooling improvement | [Open](https://github.com/BioPAL/BPS/issues/new?template=02_feature_request.yml) |
| **Algorithm proposal** | A new scientific algorithm or methodological change (justification required) | [Open](https://github.com/BioPAL/BPS/issues/new?template=03_algorithm_proposal.yml) |
| **Documentation issue** | An error, gap, or improvement in the documentation | [Open](https://github.com/BioPAL/BPS/issues/new?template=04_documentation_issue.yml) |
| **Security report** | A non-sensitive security concern (sensitive issues → [private advisory](https://github.com/BioPAL/BPS/security/advisories/new)) | [Open](https://github.com/BioPAL/BPS/issues/new?template=05_security_report.yml) |

---

## Bug reports and enhancement requests

Bug reports are an important part of making BIOMASS BPS more stable.
A complete bug report allows others to reproduce the bug and provides
insight into fixing it.

Trying out the bug-producing code on the `main` branch is often a
worthwhile exercise to confirm that the bug still exists. It is also
worth searching existing bug reports and pull requests to see if the
issue has already been reported and fixed.

### Submitting a bug report

Open an issue using the [**Bug report** template](https://github.com/BioPAL/BPS/issues/new?template=01_bug_report.yml).
The template guides you through the structured fields the maintainers need.
For feature requests, algorithm proposals, documentation issues, and security
reports, use the matching template instead.

If you are reporting a bug, please use the provided template which includes the following:

1. **Include a short, self-contained code snippet reproducing the problem.** You can format the code nicely by using GitHub Flavored Markdown:

```python
from bps_common import ...

# Your code that reproduces the bug
```

2. **Include the full version string of BPS and its dependencies.** The global version is tracked in the `VERSION` file at the root of the repository.

3. **Explain why the current behavior is wrong/not desired and what you expect instead.**

The issue will then be open to comments/ideas from the team.

See this [Stack Overflow article for tips on writing a good bug report](https://stackoverflow.com/help/mcve).

---

**Previous:** [Contributing overview](index.md) | **Next:** [Implementation](implementation.md)
