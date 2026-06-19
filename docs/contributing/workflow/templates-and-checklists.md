# Templates and checklists

Structured forms and checklists so contributors are never lost between
the project and GitHub. Four touchpoints guide every contribution:

1. **Auto tier classification**: every PR is risk-scored on opening from
   the diff; no manual tier labels. See [CI automation and contribution tiers](../automation/ci-automation-and-contribution-tiers.md).
2. **CODEOWNERS routing**: the right SME is auto-assigned from the files
   touched.
3. **Issue and PR templates**: structured forms for every issue type and
   pull request (below).
4. **PR guidance comment**: CI posts a sticky summary with tier, job
   status, and reviewers needed.

This page also lists help resources, learning links, and pre-submission
checklists.

---

## Table of Contents

- Getting Help
- Templates
- Resources
- Checklist for Contributors
- Recognition

## Getting Help

For complete information about communication channels and meetings, see
the [Communication](../../communication/index.md) page.

### Quick reference

| Where                                                            | What for                                                                                                                                      |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| [GitHub Issues](https://github.com/BioPAL/BPS/issues/new/choose) | Actionable items only: bug reports, feature requests, algorithm proposals, documentation issues, security reports. Use the matching template. |
| [GitHub Discussions](https://github.com/BioPAL/BPS/discussions)  | Open ended questions, brainstorming, scientific discussions, governance, community.                                                           |
| Office Hours                                                     | Weekly open session for contributor questions.                                                                                                |
| Community Meetings                                               | Regular community meetings.                                                                                                                   |

The recommended place to ask a question is
[GitHub Discussions](https://github.com/BioPAL/BPS/discussions). The Q&A
category is monitored by maintainers and contributors.

### Before asking for help

1. Check the [documentation](https://biomass-disc.info/docs).
2. Search [open](https://github.com/BioPAL/BPS/issues) and
   [closed](https://github.com/BioPAL/BPS/issues?q=is%3Aissue+is%3Aclosed)
   issues.
3. Search [Discussions](https://github.com/BioPAL/BPS/discussions).
4. Review similar PRs.

### Opening an issue

Every issue must be filed through one of the five templates. The template
chooser at
[`/issues/new/choose`](https://github.com/BioPAL/BPS/issues/new/choose)
walks you through the options.

---

## Templates

### Issue templates

Five issue templates are available at
[`/issues/new/choose`](https://github.com/BioPAL/BPS/issues/new/choose).
Pick the one that matches what you want to report. Each template asks for
exactly the information the maintainers need to triage and route the issue
to the right reviewer.

| Template                              | When to use                                                                                                                                                                                                           |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **01 Bug report**                     | A defect in a processor, the CI/CD pipeline, or the documentation.                                                                                                                                                    |
| **02 Feature or enhancement request** | A non-scientific feature, an enhancement to an existing component, or a tooling improvement.                                                                                                                          |
| **03 Algorithm proposal**             | A new scientific algorithm, a methodological change, or any modification with a scientific impact on the processing chain. Scientific justification is required.                                                      |
| **04 Documentation issue**            | An error, a gap, an outdated section, a broken link, or a request for clarification in the documentation.                                                                                                             |
| **05 Security report**                | A non-sensitive security concern, hardening recommendation, or supply-chain issue. Sensitive vulnerabilities go through a [private security advisory](https://github.com/BioPAL/BPS/security/advisories/new) instead. |

### Discussions categories

Six categories on
[GitHub Discussions](https://github.com/BioPAL/BPS/discussions) host
the conversations that are not yet actionable issues.

| Category | When to use |
|---|---|
| <i class="doc-icon fa-solid fa-bullhorn fa-fw" aria-hidden="true"></i> [Announcements](https://github.com/BioPAL/BPS/discussions/categories/announcements) | Project announcements, releases, governance decisions. Posts restricted to maintainers. |
| <i class="doc-icon fa-regular fa-circle-question fa-fw" aria-hidden="true"></i> [Q&A](https://github.com/BioPAL/BPS/discussions/categories/q-a) | Ask anything about usage, installation, API, processing chain, data formats. Mark the helpful reply as the answer. |
| <i class="doc-icon fa-regular fa-lightbulb fa-fw" aria-hidden="true"></i> [Ideas](https://github.com/BioPAL/BPS/discussions/categories/ideas) | Brainstorm a feature or a change before opening a tracking issue. |
| <i class="doc-icon fa-solid fa-microscope fa-fw" aria-hidden="true"></i> [Scientific discussions](https://github.com/BioPAL/BPS/discussions/categories/scientific-discussions) | Algorithms, validation, methodology, ATBD interpretations, references. |
| <i class="doc-icon fa-solid fa-landmark fa-fw" aria-hidden="true"></i> [Governance](https://github.com/BioPAL/BPS/discussions/categories/governance) | Project governance, maintainer paths, policy questions, open source strategy. |
| <i class="doc-icon fa-regular fa-hand fa-fw" aria-hidden="true"></i> [Show and tell](https://github.com/BioPAL/BPS/discussions/categories/show-and-tell) | Introductions, usage stories, downstream projects, papers, conference talks. |

### Pull Request template

Every PR uses the
[standard template](https://github.com/BioPAL/BPS/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
that ships with the repository. It is auto-filled when you open a PR
from the GitHub UI. The template covers:

- The linked issue and its approval label (`status:approved`, `good-first-issue`, or `help-wanted`).
- A short description of what the PR changes (not the problem, that lives in the issue).
- Notes for reviewers.
- User-facing change flag and release-note sentence.
- Documentation update flag.
- Expected tier (computed automatically by CI, your declaration just helps reviewers spot a mismatch).
- AI assistance disclosure.
- A short checklist of items the CI cannot verify on its own (scope match, single issue, breaking change flagged, reviewer competence).

### Licensing PR Description Template

When adding external libraries or dependencies to your PR, include the following information in your PR description:

```markdown
## Dependencies Added

- **library-name** (v1.2.3)
  - License: MIT
  - Purpose: [Brief description]
  - Compatibility: Compatible with Apache 2.0
  - License URL: [Link to license]
```

For complete licensing requirements, see the [Licensing](../../about/licensing/index.md) section.

---

## Resources

### Documentation

- [Getting Started](https://biomass-disc.info/docs) - Introduction and getting started guide
- [Code of Conduct](../code-of-conduct.md) - Community standards and expectations
- [Contributing overview](../index.md) - Journey-based contribution workflow
- [Governance](../../governance/index.md) - Roles, responsibilities, and decision-making
- [Processor ownership](../../governance/processor-ownership.md) - Scientific and technical contacts per module
- [Architecture](../technical/architecture.md) - Monorepo layout, `bps-*` modules, and processor structure
- [Code standards](../technical/code-standards.md) - Coding conventions and best practices
- [Documentation standards](../technical/documentation-standards.md) - Documentation writing standards and best practices
- [CI automation and contribution tiers](../automation/ci-automation-and-contribution-tiers.md) - Pipeline reference, tier detection, branch protection
- [Release process](../policy/release-process.md) - How to prepare and publish a release
- [Communication](../../communication/index.md) - Communication channels and meeting schedules
- [Licensing](../../about/licensing/index.md) - Apache 2.0 license requirements and legal obligations

### Learning Resources

**Git and Version Control:**

- [GitHub Help Pages](https://help.github.com/)
- [Git Documentation](https://git-scm.com/doc)
- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials)
- [GitHub's Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [Pro Git Book](https://git-scm.com/book/en/v2) (free online)

**Python Development:**

- [Python Documentation](https://docs.python.org/3/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
- [Real Python](https://realpython.com/) - Python tutorials and guides
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

**Testing:**

- [pytest Documentation](https://doc.pytest.org/en/latest/)
- [pytest Best Practices](https://docs.pytest.org/en/latest/explanation/goodpractices.html)
- [Test-Driven Development with Python](https://www.obeythetestinggoat.com/)

**Scientific Python:**

- [NumPy Documentation](https://numpy.org/doc/stable/)
- [SciPy Documentation](https://docs.scipy.org/doc/scipy/)
- [xarray Documentation](https://docs.xarray.dev/) - Similar project structure

**Code Quality:**

- [Black Code Formatter](https://black.readthedocs.io/)
- [Ruff Linter](https://docs.astral.sh/ruff/)
- [mypy Type Checker](https://mypy.readthedocs.io/)

### External Resources

- **FAIR Principles**: [https://www.go-fair.org/fair-principles/](https://www.go-fair.org/fair-principles/)
- **ESA Open Science**: [https://www.esa.int/Science_Exploration/Space_Science/Open_Science](https://www.esa.int/Science_Exploration/Space_Science/Open_Science)
- **Open Source Guides**: [https://opensource.guide/](https://opensource.guide/)

### Workshops and Training

- **Onboarding Workshops**: Monthly sessions for new contributors (2h)
- **Technical Training**: Quarterly sessions on tools and processes
- **Hackathons**: Annual community events
- **Intercomparison Exercises**: Regular scientific validation activities

---

## Checklist for Contributors

### Before Submitting a PR

**Issue and scope:**

- [ ] A tracking issue exists for this change.
- [ ] The issue is labelled `status:approved`, `good-first-issue`, or `help-wanted`.
- [ ] The PR closes exactly one issue, and the scope matches what was approved.

**Code Quality:**

- [ ] Pre-commit hooks installed and passing (`pre-commit run --all-files`).
- [ ] Code formatted with `ruff format src/ test/`.
- [ ] Linting clean: `ruff check --fix src/ test/`.
- [ ] Type hints present where appropriate; `mypy` passes.
- [ ] All code follows the [Code standards](../technical/code-standards.md).
- [ ] No secrets or sensitive data in code.

**DCO and Licensing:**

- [ ] All commits carry a `Signed-off-by:` trailer (`git commit -s`).
- [ ] New files include SPDX headers (`SPDX-FileCopyrightText` + `SPDX-License-Identifier: Apache-2.0`).
- [ ] `reuse lint` passes locally (or the pre-commit reuse hook passes).

**Testing:**

- [ ] Tests added or modified and passing (`pytest -m baseline` on `test/baseline/`).
- [ ] Baseline reference outputs updated if the behaviour intentionally changed.
- [ ] If targeting `release`, you understand that the `CI gate` will be red until a maintainer triggers the workflow with `run_heavy=true`.

**Documentation:**

- [ ] Docstrings added or updated.
- [ ] Documentation site updated if the change is user facing.

**Pull Request:**

- [ ] PR template completed fully.
- [ ] Clear description of what changes (not the problem, that lives in the issue).
- [ ] Commit messages are clear and descriptive.
- [ ] Scientific validation completed or summary included for Tier 1 and Tier 2 changes.
- [ ] Branch up to date with `develop`, no merge conflicts.
- [ ] All CI checks passing (`CI gate` green).

### During Review

- [ ] Respond to comments promptly (< 3 days)
- [ ] Address all points raised by reviewers
- [ ] Update PR with corrections
- [ ] Communicate clearly about changes
- [ ] Re-run tests after changes
- [ ] Update documentation if requested

---

## Recognition

Contributors are recognized in:

- Release notes
- Project documentation
- Community communications
- Annual contributor acknowledgments

Thank you for contributing to BIOMASS BPS! Your efforts help advance open science and Earth observation capabilities.

---

**Questions?** Ask on
[GitHub Discussions](https://github.com/BioPAL/BPS/discussions).
The [Q&A category](https://github.com/BioPAL/BPS/discussions/categories/q-a)
is the recommended place for usage questions, and
[Scientific discussions](https://github.com/BioPAL/BPS/discussions/categories/scientific-discussions)
for algorithm and methodology topics. Open issues are reserved for
actionable items that match one of the five templates.

**Last Updated:** 2026

---

**Previous:** [Practical workflow](practical-workflow.md) | **Next:** [Becoming a maintainer](../policy/becoming-a-maintainer.md)
