# Roles and authority

BIOMASS BPS is ESA mission software developed under contract and released as open
source. Governance balances ESA oversight, maintainer review, and scientific
expertise on processor changes, while keeping the contribution bar low for
documentation and non-scientific fixes.

For who holds repository access and module routing, see
[Repository stewards](repository-stewards.md) and
[Processor ownership](processor-ownership.md).

---

## Decision flow

```{mermaid}
flowchart TD
    contrib[Contributor opens PR] --> ci[CI classifies tier]
    ci --> maint[Core maintainer review]
    ci --> sme[Scientific module expert when required]
    maint --> merge{Squash merge?}
    sme --> merge
    merge -->|Tier 0-1 develop| done[Merged to develop]
    merge -->|Release promotion| esa[ESA approval on VERSION and CHANGELOG]
    esa --> main[Merged to main]
```

Merge rules, branch protection, and tier definitions are in
[CI automation and contribution tiers](../contributing/ci-automation-and-contribution-tiers.md).

---

## Roles and responsibilities

| Role | Responsibility | Decision scope |
| ---- | -------------- | -------------- |
| **ESA** | Mission owner, repository oversight, release authority | Final say on disputes; mandatory approval on `VERSION` and `CHANGELOG.md`; Tier 2–3 governance |
| **Open Science Lead** | FAIR strategy, documentation quality, community coordination | Tier 2 Open Science compliance; documentation standards |
| **Core maintainers** | Technical review, merges, release execution | Tier 0–2 technical approval; branch and release operations |
| **Scientific module experts** | Algorithm and baseline validation in their domain | Tier 1–2 scientific approval on owned processors |
| **Contributors** | Code, docs, tests via pull request | Propose changes; respond to review |

---

## ESA release gate

Any pull request that modifies `VERSION` or `CHANGELOG.md` requires explicit
approval from the ESA representatives defined in
[Repository stewards](repository-stewards.md) and enforced via `CODEOWNERS`.
No promotion to `main` can merge without that approval.

Step-by-step release commands are in the maintainer runbook:
[Release process](../contributing/release-process.md).

---

## Principles

- **Transparency**: decisions are documented and traceable.
- **Scientific rigor**: algorithm changes are validated against agreed criteria.
- **Community engagement**: external contributions are welcome when issues are approved.
- **Operational stability**: production branches stay protected and tested.
- **Institutional neutrality**: authority follows contribution and expertise, not affiliation alone.

---

## Changing governance

The framework is reviewed annually and when ESA or the steward group requests an
update. Proposals go through GitHub Discussions (Governance category) or a
governance issue; ESA has final authority on policy changes.

---

**Previous:** [Governance](index.md) | **Next:** [Repository stewards](repository-stewards.md)
