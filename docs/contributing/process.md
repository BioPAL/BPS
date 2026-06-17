# Contributing to BIOMASS L2 Processors - Understanding the Project

This section covers all the conceptual information you need to understand before starting to write any code or clone the repository.

**Important:** Please read this entire section before starting to write any code or clone the repository. Understanding our processes, standards, and expectations will help you contribute more effectively.

---

## Table of Contents

- Licensing Requirements
- Code of Conduct
- Governance
- Contribution Tiers
- Development Workflow Overview
- Pull Request Process
- Testing Requirements
- Scientific Validation
- Coding Standards
- Documentation Standards
- Becoming a Maintainer

**Read this entire section before writing any code or setting up your environment.** Understanding our processes, standards, and expectations will make your contribution journey smoother and more successful.

---

## Licensing Requirements

**All contributions to BioPAL must be compatible with the Apache License 2.0.** This is a critical legal requirement that ensures the project can maintain its open-source status and legal clarity.

### Key Requirements

- **All code contributions** must be licensed under Apache 2.0
- **External libraries and dependencies** must be Apache 2.0 compatible
- **Third-party code** must be properly attributed and license-compatible
- **License headers** should be included in source files

### Before Submitting a Pull Request

When adding external libraries or dependencies to your PR, you must:

1. Verify the library's license is compatible with Apache 2.0
2. Document the license in your PR description
3. Ensure no GPL or other incompatible licenses are included
4. Provide proper attribution for any third-party code

For complete information about licensing requirements, license compatibility, external dependencies, and legal obligations, please see the [Licensing documentation](https://biomass-disc.info/docs/licensing).

---

## Code of Conduct


This project adheres to a Code of Conduct that all contributors are expected to follow. For complete details, please see the [Code of Conduct documentation](https://biomass-disc.info/docs/code-of-conduct).

---

## Governance


For complete information about BioPAL's governance structure, including the Steering Council, BDFL, and Institutional Partners, please see the [Governance documentation](https://biomass-disc.info/docs/governance).

---

## Contribution Tiers


All contributions are classified into **tiers** based on their impact and scope. This helps ensure appropriate review and validation processes.

- **Tiers 0, 1, and 2** are computed automatically by the CI on every commit, from the changed files and the policy in `.github/tier-policy.yml`. They determine which CI stages run.
- **Tier 3** is not a CI-computed tier. It is a qualitative governance category covering release decisions and policy changes that go through an ESA decision outside the automated pipeline.

The flowchart below provides a detailed breakdown by tier:

```{mermaid}
flowchart TD
    PR["Pull Request (PR)"]
    Tiers["Contribution Tiers\n(automatic tier detection via tier-policy.yml)"]

    PR --> Tiers

    Tiers --> Tier0["Tier 0\nRoutine / Performance\n1 Reviewer"]
    Tiers --> Tier1["Tier 1\nMinor Scientific\n2 Reviewers"]
    Tiers --> Tier2["Tier 2\nMajor Algorithmic\n3 Reviewers"]
    Tiers --> Tier3["Tier 3\nRelease & Policy\nESA Decision"]

    Tier0 --> Val0["Baseline gate only\n(10 parallel jobs: baseline-marker, DCO, REUSE,\npre-commit, security, build, sensitive-files,\nunit, docs, dependabot)"]
    Tier1 --> Val1["Baseline + Extended CI\n(pytest -m extended on test/extended/)"]
    Tier2 --> Val2["Baseline + Extended + Heavy CI\n(pytest -m heavy on test/heavy/,\nrun_heavy=true required on release PRs)"]
    Tier3 --> Val3["Governance & Strategic Decision\n(no automated pipeline)"]

    Val0 --> Review0["1 Core Maintainer approval"]
    Val1 --> Review1["Scientific Module Expert\n+ Core Maintainer approval"]
    Val2 --> Review2["Scientific Module Expert(s)\n+ ESA representative\n+ Core Maintainer approval"]
    Val3 --> Review3["Explicit ESA decision\nin consultation with governance group"]

    Review0 --> Meeting0["WEEKLY Review Meeting\nOpen and documented"]
    Review1 --> Meeting1["MONTHLY Review Meeting\nOpen and documented"]
    Review2 --> Meeting2["QUARTERLY Review Meeting\nOpen and documented"]
    Review3 --> Merge3["Merge Decision"]

    Meeting0 --> Merge0["Merge Decision"]
    Meeting1 --> Merge1["Merge Decision"]
    Meeting2 --> Merge2["Merge Decision"]

    classDef defaultStyle fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,color:#333
    class PR,Tiers,Tier0,Tier1,Tier2,Tier3,Val0,Val1,Val2,Val3,Review0,Review1,Review2,Review3,Meeting0,Meeting1,Meeting2,Merge0,Merge1,Merge2,Merge3 defaultStyle

    style PR fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Tiers fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Tier0 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style Tier1 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style Tier2 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style Tier3 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style Val0 fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Val1 fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Val2 fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Val3 fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Review0 fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Review1 fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Review2 fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Review3 fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Meeting0 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style Meeting1 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style Meeting2 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style Merge0 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:3px
    style Merge1 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:3px
    style Merge2 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:3px
    style Merge3 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:3px
```

### Tier Comparison Table

| Aspect | Tier 0: Routine Maintenance | Tier 1: Minor Scientific | Tier 2: Major Algorithmic | Tier 3: Release & Policy |
|--------|----------------------------|-------------------------|--------------------------|------------------------|
| **Type** | Routine / Performance | Minor Scientific | Major Algorithmic | Release & Policy |
| **Tier detection** | Automatic: no locked or SME paths modified, baseline marker OK | Automatic: locked paths modified, SME-owned paths modified, baseline marker differs, or Dependabot major bump | Automatic: PR targets `release` branch (Heavy mandatory), or `run_heavy=true` upclass from Tier 1 via `workflow_dispatch` | Not computed by CI: ESA governance category |
| **Reviewers** | 1 Core Maintainer | Scientific Module Expert + Core Maintainer | Scientific Module Expert(s) + ESA representative + Core Maintainer | ESA Decision |
| **Examples** | Bug fixes, documentation, code refactoring without scientific impact, typo corrections | Small parameter updates, improved default values, minor algorithm modifications, changes to CI config or any locked path | New retrieval approaches, significant changes to error models, redefinition of key L2 product variables, anything promoted to the `release` branch | New processor version releases, governance changes, major policy updates |
| **CI Pipeline** | Baseline gate only (10 parallel jobs: marker-signal + DCO + REUSE + pre-commit + security + build + sensitive-files + unit + docs + dependabot) | Baseline + Extended (`pytest -m extended` on `test/extended/`) | Baseline + Extended + Heavy (`pytest -m heavy` on `test/heavy/`). Heavy requires `run_heavy=true` set via `workflow_dispatch` to be merge-eligible. | No automated pipeline: governance process |
| **Validation** | Baseline tests must pass. No scientific validation required. | Extended TDS validation (automatic via CI). Scientific validation summary in PR. | Full heavy TDS validation (automatic via CI). Design document with proposed change and alternatives. | Documented discussion in issues or governance documents |
| **Timeline** | < 3 days | 5–7 days | 10–14 days | By urgency |
| **Expected Volume** | 60–70% of PRs | 20–25% of PRs | 5–10% of PRs | < 5% of PRs |
| **Review Meeting** | Weekly (open and documented) | Monthly (open and documented) | Quarterly (open and documented) | N/A |


---

## Development Workflow Overview

This section provides a conceptual overview of how development works in BioPAL. For detailed step-by-step instructions with commands, see the "Development Workflow - Step by Step" section in Part 2.

```{raw} html
<img alt="Contribution Process" src="../images/CI-CD-Contribution_Light.drawio.svg" class="only-light">
<img alt="Contribution Process" src="../images/CI-CD-Contribution_Shadow.drawio.svg" class="only-dark">
```

### Branching Strategy

- **`main`**: Operational/production branch (protected, stable). Promoted to from `release` after ESA approval.
- **`release`**: Release candidate branch (protected). Pre-release validation runs here; Heavy CI is mandatory before promotion to `main`.
- **`develop`**: Main development branch (protected, latest development state).
- **Feature branches**: `feature/description`, `bugfix/issue-id`, `docs/topic`: opened from `develop`.

### Branch Protection Rules

All three main branches are protected by GitHub rulesets:

| Branch | Approvals required | Signed commits (GPG/SSH) | Merge strategy | Admin bypass |
|--------|--------------------|--------------------------|----------------|-------------|
| `develop` | 1 | No | Squash only | Yes |
| `release` | 2 | Yes | Squash only | Yes |
| `main` | 3 | Yes | Squash only | **No** |

**Squash merges**: all commits from a PR are collapsed into one commit on the target branch. Write a clear and descriptive final commit message.

**Dismiss stale reviews**: any approval is automatically invalidated when a new commit is pushed after the approval. A fresh approval is required.

**`CI gate`** is the required status check on all three branches. A PR cannot be merged until this check passes.

**PRs targeting `release` require explicit Heavy authorisation**: the `CI gate` will be red on every `pull_request` event until a maintainer triggers the workflow via `Actions → Run workflow` with `run_heavy=true`. This is the explicit consent step before promotion to `main`.

### DCO: Developer Certificate of Origin

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

### Dependabot: Automated Dependency Updates

Dependabot opens PRs every Monday to update pip and GitHub Actions dependencies, always targeting `develop`. These PRs go through the normal CI pipeline.

**Important rule**: a **major version bump** on a pip dependency is automatically classified as **Tier 1** by the CI, requiring SME review.

### Workflow Concepts

1. **Start from develop**: Always create feature branches from an up-to-date `develop` branch
2. **One feature per branch**: Keep changes focused and atomic
3. **Regular updates**: Keep your branch synchronized with `develop` to avoid conflicts
4. **Commit frequently**: Make small, logical commits with clear messages
5. **Test before submitting**: Run all checks locally before creating a pull request

### The Editing Process

1. **Make changes** to code or documentation
2. **Review your changes** to ensure they follow our standards
3. **Run tests** to verify everything works
4. **Commit changes** with descriptive messages
5. **Push to your branch** and create a pull request

For detailed commands and step-by-step instructions, see the "Development Workflow - Step by Step" section in Part 2.

---

## Pull Request Process


When you submit a Pull Request, it goes through the CI/CD pipeline on GitHub. The diagrams below show the full workflow: from PR entry point to tier triage and merge decision:

```{raw} html
<img alt="CI/CD Workflow" src="../images/CI-CD-Workflow.drawio.svg" class="only-light">
<img alt="CI/CD Workflow" src="../images/CI-CD-Workflow_Shadow.drawio.svg" class="only-dark">
```


### Diagram Appendix : 
```{raw} html
<img alt="CI/CD Annexe" src="../images/CI-CD-Annexe_Light.drawio.svg" class="only-light">
<img alt="CI/CD Annexe" src="../images/CI-CD-Annexe_Shadow.drawio.svg" class="only-dark">
```


### PR Template

Every PR must use the standard template defined in [`https://github.com/BioPAL/BPS/blob/main/.github/PULL_REQUEST_TEMPLATE.md`](https://github.com/BioPAL/BPS/blob/main/.github/PULL_REQUEST_TEMPLATE.md).

### Automated CI/CD Checks and Review Requirements

All pull requests go through automated checks. The required checks and approvals depend on the contribution tier (see the flowchart and comparison table above for a visual overview).

**CI/CD Pipeline:**

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
- ⚠ Requires TDS dataset: see [Code Standards](https://biomass-disc.info/docs/code-standards) for access procedure

**Heavy Pipeline (Tier 2):**
- Full scientific regression (`pytest -m heavy`)
- ⚠ Requires TDS dataset: see [Code Standards](https://biomass-disc.info/docs/code-standards) for access procedure

**Automatic Tier Detection:**

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

**Review and Approval Requirements:**

- **Tier 0**: At least **1 core maintainer** approval
- **Tier 1**: **Scientific module expert** approval + **Core maintainer** approval
- **Tier 2**: **Scientific module expert(s)** approval + **Open Science Lead** approval + **ESA representative(s)** approval + **Core maintainer** approval
- **Tier 3**: Explicit ESA decision in consultation with governance group

**The PR cannot be merged until all required checks pass and all required approvals are obtained.**

### Review Your Code

When you're ready to ask for a code review, file a pull request. Before you do, once again make sure that you have followed all the guidelines outlined in this document regarding code style, tests, performance tests, and documentation. You should also double check your branch changes against the branch it was based on:

1. **Navigate to your repository on GitHub** -- https://github.com/your-username/BioPAL (or the main repository if working directly)
2. **Click on "Branches"**
3. **Click on the "Compare" button** for your feature branch
4. **Select the "base" and "compare" branches**, if necessary. This will be `develop` and `feature/your-feature-name`, respectively.

### Accepting an Intentional Baseline Change

If your PR intentionally changes processor output (e.g. a scientific fix or new algorithm), the `baseline-marker-signal` job will report a difference. The CI elevates the PR to Tier 1 automatically, and `Extended` runs to produce additional evidence. To get the change merged:

1. Confirm with the relevant SME (auto-assigned via `CODEOWNERS`) that the change is correct.
2. Update `test/baseline/` reference outputs in the same PR so the new behaviour becomes the new baseline.
3. The SME reviews the diff (code + baseline update) and approves via a standard GitHub review.

There is no `baseline:accepted` label any more; SME approval flows through the native CODEOWNERS-required review on the PR.

### ESA Reviewer Gate on Releases

Any modification to `VERSION` or `CHANGELOG.md` requires an explicit approval from the ESA reviewer defined in `CODEOWNERS`. No release can be merged without this approval.

**Pre-submission checklist:**

- [ ] All code follows the [Code Standards](https://biomass-disc.info/docs/code-standards)
- [ ] All tests pass locally (`pytest -m unit`)
- [ ] Code is properly formatted (`black`, `ruff`)
- [ ] Type hints are correct (`mypy` passes)
- [ ] Every commit is signed (`git commit -s`): DCO required
- [ ] All new files have SPDX license headers (REUSE compliance)
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with `develop`
- [ ] No merge conflicts
- [ ] Scientific validation completed (if Tier 1-2)
- [ ] PR template is fully completed

### Finally, Make the Pull Request

If everything looks good, you are ready to make a pull request. A pull request is how code from a local repository becomes available to the GitHub community and can be looked at and eventually merged into the `develop` branch. This pull request and its associated changes will eventually be committed to the `develop` branch, and later merged into `main` for production releases.

The following sequence diagram illustrates the complete pull request process:

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

To submit a pull request:

1. **Navigate to your repository on GitHub**
2. **Click on the "Pull Request" or "Pull Request" button**
3. **You can then click on "Commits" and "Files Changed"** to make sure everything looks okay one last time
4. **Write a description of your changes** in the "Preview Discussion" tab, following the [PR template](https://github.com/BioPAL/BPS/blob/main/.github/PULL_REQUEST_TEMPLATE.md) provided in Help & Resources.
5. **Click "Send Pull Request" or "Create Pull Request"**

This request then goes to the repository maintainers, and they will review the code.

**Using Draft Pull Requests:**

If you don't think your request is ready to be merged, just say so in your pull request message and use the "Draft PR" or "Draft" feature of GitHub. This is a good way of getting some preliminary code review and feedback before your code is ready for final review. Draft pull requests:

- Indicate that the code is still work-in-progress
- Allow reviewers to provide early feedback
- Don't block other work or confuse reviewers about readiness
- Can be converted to a regular pull request when ready

Mention anything you'd like particular attention for - such as a complicated change or some code you are not happy with. This helps reviewers focus on the areas that need the most scrutiny.

If you need to make more changes after submitting your PR, you can update your branch and push the changes. The pull request will automatically be updated with the latest code and restart the Continuous Integration tests.

### Delete Your Merged Branch (Optional)

Once your feature branch is accepted and merged, you may want to clean up by deleting the branch locally and remotely. For detailed instructions on how to delete branches, see the "Development Workflow - Step by Step" section in Part 2.

### Responding to Reviews

- Address all reviewer comments promptly (< 3 days)
- Update the PR with requested changes
- Re-run CI checks automatically
- Iterate until all approvals are obtained

---

## Testing Requirements

For complete information about test coverage requirements, test types, writing tests, and running tests, please refer to the **[Code Standards](https://biomass-disc.info/docs/code-standards)** documentation.

### Key Points for Contributors

- **Minimum coverage**: ≥ 60% on code touched by the PR (`--cov-fail-under=60`)
- **Test types**: Unit, baseline, extended, heavy: see [Code Standards](https://biomass-disc.info/docs/code-standards) for the full marker reference
- **Use pytest**: The project uses pytest for all testing

### Test-Driven Development (TDD)

BioPAL strongly encourages contributors to embrace [test-driven development (TDD)](https://en.wikipedia.org/wiki/Test-driven_development). This development process "relies on the repetition of a very short development cycle: first the developer writes an (initially failing) automated test case that defines a desired improvement or new function, then produces the minimum amount of code to pass that test."

**Why write tests first?**

- **Clarifies requirements**: Writing tests first forces you to think about what the code should do before implementing it
- **Better design**: Tests help you design cleaner interfaces and APIs
- **Prevents regressions**: Having tests in place ensures that future changes don't break existing functionality
- **Documentation**: Tests serve as executable documentation showing how code should be used
- **Confidence**: Tests give you confidence to refactor and improve code

**The TDD cycle:**

1. **Red**: Write a failing test that describes the desired behavior
2. **Green**: Write the minimum code needed to make the test pass
3. **Refactor**: Improve the code while keeping tests passing

Often the test can be taken from the original GitHub issue. However, it is always worth considering additional use cases and writing corresponding tests.

Adding tests is one of the most common requests after code is pushed to BioPAL. Therefore, it is worth getting in the habit of writing tests ahead of time so that this is never an issue.

### Performance Testing

Performance matters and it is worth considering whether your code has introduced performance regressions. BioPAL may include a suite of benchmarking tests to enable easy monitoring of the performance of critical operations.

**When to run performance tests:**

- After making changes to core algorithms
- When optimizing code for speed
- Before submitting a pull request that affects performance-critical code
- When investigating performance issues

**Performance considerations:**

- **Memory usage**: Monitor memory consumption, especially for large datasets
- **CPU usage**: Check if algorithms can be parallelized
- **I/O operations**: Minimize file read/write operations where possible
- **Vectorization**: Use NumPy vectorized operations instead of Python loops
- **Caching**: Consider caching expensive computations when appropriate

**Reporting performance changes:**

If your changes affect performance, include performance metrics in your pull request:

- Execution time before and after your changes
- Memory usage comparisons
- Any optimizations made
- Benchmark results if available

For detailed commands on how to write tests, run tests, and use performance testing tools, see the "Testing Requirements - Code Examples and Commands" section in Part 2.

---

## Scientific Validation


### When Validation is Required

**Tier 1-2 contributions** must include scientific validation.

### Reference Datasets

Validation relies on **TDS (Test Data Sets)**: golden datasets used as a reference to compare processor outputs. Tests are organized in three levels within each module:

- `test/baseline/`: small reference outputs checked on every PR (marker: `baseline`)
- `test/extended/`: broader validation for Tier 1+ (marker: `extended`)
- `test/heavy/`: full scientific regression for Tier 2 (marker: `heavy`)

⚠ The extended and heavy TDS are not distributed with the repository. A procedure to retrieve them will be made available shortly. In the meantime, these pipelines run exclusively on CI servers.

### Validation Workflow

All **Tier 1-2 pull requests** are automatically validated using a standardized validation script that runs in the CI/CD pipeline. The validation process works as follows:

1. **Automatic CI Validation**: When you submit a Tier 1-2 PR, the CI/CD pipeline automatically:
   - Executes the validation script on a predetermined Test Data Set (TDS)
   - Compares your changes against the **nominal version** (baseline) on the same TDS
   - Calculates metrics: RMSE, bias, correlation, etc.
   - Generates a validation report with metrics and visualizations
   - The report is automatically attached to your PR

2. **Custom Dataset Validation** (Optional): If you need validation on a specific dataset that is not part of the standard TDS:
   - Request access to a specific dataset through the PR description
   - The validation will be executed on the requested dataset
   - A separate report will be generated for the custom dataset
   - This is in addition to (not a replacement for) the standard TDS validation

**Important**: The main pull request will always be compared with the nominal version on the same predetermined TDS. Custom dataset validations are supplementary and do not replace the standard validation.


### Validation Report Structure

The validation script automatically generates a comprehensive HTML report that includes:

- **Plots for all variables**: Visualizations showing the results for each variable processed
- **Comparison with nominal version**: Side-by-side comparisons and difference plots between the nominal (baseline) version and your new feature
- **Metrics and statistics**: RMSE, bias, correlation, and other relevant metrics for each variable
- **Summary tables**: Tabulated results for easy review

The HTML report is automatically generated by the validation script and attached to your PR. No manual notebook creation is required - the entire validation process is automated.

### Validation Report Storage

- **HTML Report**: Automatically generated and attached to the PR
- **Report Location**: Available through the CI/CD pipeline artifacts and linked in the PR
- **Metrics Summary**: Key metrics are also included in the PR description for quick reference

### Validation Execution

**Standard Validation (Automatic for Tier 1-2):**
- Runs automatically in the CI/CD pipeline for all Tier 1-2 PRs
- Uses the TDS golden dataset consistent across all validations
- Compares your changes against the baseline reference on the same TDS
- Produces a validation report attached to the PR as a CI artefact
- Extended and heavy TDS are not available locally: all Tier 1-2 validations run on CI servers

**Custom Dataset Validation (On Request):**
- If you need validation on a specific dataset beyond the standard TDS, request it in your PR description
- Custom validations run on CI servers with access to larger datasets
- Requires approval from ESA for dataset access
- Supplementary: does not replace the mandatory TDS validation

**Data Access:**
- **Tier 0**: Baseline tests only (no TDS required)
- **Tier 1**: Extended TDS validation (automatic via CI)
- **Tier 2**: Heavy TDS validation (automatic via CI) + custom dataset on request

---

## Coding Standards

For complete coding standards, naming conventions, formatting rules, type hints, error handling, and logging guidelines, please refer to the **[Code Standards](https://biomass-disc.info/docs/code-standards)** documentation.

**Key points:**
- Follow PEP 8 with modifications enforced by `black`
- Use type hints for all public functions
- Run `black`, `ruff`, and `mypy` before committing
- Set up pre-commit hooks for automatic code quality checks

### Backwards Compatibility

Please try to maintain backwards compatibility. BioPAL has a growing number of users with lots of existing code, so don't break it if at all possible. If you think breakage is required, clearly state why as part of the pull request.

**Principles:**

- **Avoid breaking changes**: Changes that break existing user code should be avoided unless absolutely necessary
- **Deprecation cycle**: When breaking changes are necessary, use a deprecation cycle to warn users before removing functionality
- **Clear communication**: Document breaking changes clearly in release notes and migration guides
- **Versioning**: Breaking changes should typically be reserved for major version releases

**Be especially careful when changing function and method signatures**, because any change may require a deprecation warning. Instead of simply raising an error when users pass deprecated arguments, you should catch them and emit a deprecation warning that clearly states what is deprecated, what to use instead, and when it will be removed.

**Deprecation cycle process:**

1. **Add deprecation warning**: In the current version, add a `DeprecationWarning` that clearly states:
   - What is deprecated
   - What to use instead
   - When it will be removed (typically next major version)

2. **Update documentation**: 
   - Mark deprecated features in the documentation
   - Provide migration examples
   - Update docstrings with deprecation notices

3. **Wait for next major version**: Remove the deprecated functionality only in a major version release

4. **Update release notes**: Clearly document all deprecations and breaking changes

For code examples showing how to implement deprecation warnings, see the [Code Standards](https://biomass-disc.info/docs/code-standards) documentation.

**When breaking changes are acceptable:**

- **Security vulnerabilities**: Fixing security issues may require breaking changes
- **Scientific correctness**: If the current implementation is scientifically incorrect, breaking changes may be necessary
- **Major architectural improvements**: Significant improvements that benefit the entire project
- **Removing clearly broken or unused features**: Features that are known to be broken or unused

**Always discuss breaking changes** in an issue or pull request before implementing them, and ensure they are clearly documented.

---

## Documentation Standards

If you're not the developer type, contributing to the documentation is still of huge value. You don't even have to be an expert on BioPAL to do so! In fact, there are sections of the docs that are worse off after being written by experts. If something in the docs doesn't make sense to you, updating the relevant section after you figure it out is a great way to ensure it will help the next person.

For complete information about documentation standards, docstring format (NumPy style), writing conventions, documentation types, and best practices, please refer to the **[Documentation Standards](https://biomass-disc.info/docs/documentation-standards)** documentation.

**Key points:**
- **Docstrings** should follow the NumPy Docstring Standard
- **Standalone documentation** should provide *why* and *when* to use features, with examples and tutorials
- **Keep documentation up to date**: When you change code, update the corresponding documentation
- **Write for your past self**: Document things that weren't obvious to you when you first learned them

### Documentation Updates Required

When contributing, update:
- User guides (if user-facing changes)
- Developer guides (if workflow changes)
- API documentation (if adding/modifying functions)
- Notebooks (if examples affected)
- Product definitions (if products change)

For detailed commands on how to build and preview documentation, see the "Documentation Standards - Code Examples and Commands" section in Part 2.

## Becoming a Maintainer

We value long-term contributors who demonstrate commitment, technical expertise, and community leadership. Becoming a maintainer is a natural progression for contributors who have shown consistent dedication to the project. This section outlines the pathway from contributor to maintainer, inspired by successful open-source projects like Xarray.

### What is a Maintainer?

Maintainers are trusted community members who have demonstrated deep commitment to the project and have been granted elevated permissions on the repository. They play a crucial role in:

- Reviewing and merging pull requests
- Managing releases and versioning
- Setting technical direction and architecture decisions
- Ensuring code quality and consistency
- Mentoring new contributors
- Representing the project in the community

For more details about maintainer responsibilities, see the [Governance documentation](https://biomass-disc.info/docs/governance).

### The Path to Maintainership

Becoming a maintainer is not about a specific number of contributions or a fixed timeline. Instead, it's about demonstrating consistent commitment, technical competence, and alignment with the project's values and goals. The path typically follows these stages:

The following flowchart illustrates the journey from contributor to maintainer:

```{mermaid}
flowchart TD
    Start([New Contributor]) --> Stage1[Stage 1: Active Contributor<br/>3-6 months]
    
    Stage1 --> Contrib1[Regular contributions]
    Stage1 --> Standards[Follow standards]
    Stage1 --> Community[Community engagement]
    
    Contrib1 --> Check1{Quality and<br/>consistency?}
    Standards --> Check1
    Community --> Check1
    
    Check1 -->|Yes| Stage2[Stage 2: Trusted Contributor<br/>6-12 months]
    Check1 -->|No| Stage1
    
    Stage2 --> Review[Code review]
    Stage2 --> Mentor[Mentoring]
    Stage2 --> Complex[Complex issues]
    Stage2 --> Tier12[Tier 1-2 contributions]
    
    Review --> Check2{Leadership and<br/>expertise?}
    Mentor --> Check2
    Complex --> Check2
    Tier12 --> Check2
    
    Check2 -->|Yes| Stage3[Stage 3: Maintainer Candidate]
    Check2 -->|No| Stage2
    
    Stage3 --> Express[Express interest]
    Express --> Eval[Evaluation by maintainers]
    Eval --> Discuss[Internal discussion]
    Discuss --> Nominate{Nomination?}
    
    Nominate -->|Yes| Approve[Steering Council approval]
    Nominate -->|No| Stage2
    
    Approve --> Maintainer([Maintainer])
    
    classDef defaultStyle fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,color:#333
    class Start,Stage1,Stage2,Stage3,Maintainer,Contrib1,Standards,Community,Check1,Review,Mentor,Complex,Tier12,Check2,Express,Eval,Discuss,Nominate,Approve defaultStyle
    
    style Start fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Stage1 fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Stage2 fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style Stage3 fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Maintainer fill:#a3d8b0,stroke:#a3d8b0,stroke-width:3px
    style Check1 fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Check2 fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Nominate fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
```

The following table compares the different stages:

| Criterion | Stage 1: Active Contributor | Stage 2: Trusted Contributor | Stage 3: Maintainer Candidate |
|-----------|----------------------------|-----------------------------|-------------------------------|
| **Typical duration** | 3-6 months | 6-12 months | 2-4 weeks (process) |
| **Contributions** | Regular and quality | Complex and significant | Leadership and responsibilities |
| **Code review** | Submit PRs | Review others' PRs | Advanced review and mentoring |
| **Engagement** | Discussions and questions | Mentoring and triage | Project management |
| **Validation** | Mainly Tier 0 | Tier 1-2 with validation | All tiers |
| **Permissions** | Standard contributor | Trusted contributor | Maintainer candidate |
| **Next step** | Stage 2 | Stage 3 | Maintainer |

#### Stage 1: Active Contributor

**Focus:** Build a track record of quality contributions

**What to do:**
- Make regular, meaningful contributions (code, documentation, tests, reviews)
- Follow all contribution guidelines and coding standards
- Respond promptly to review feedback
- Help improve documentation and examples
- Participate in discussions and answer questions

**What we look for:**
- Consistent contributions over time (not just one large contribution)
- High-quality code that follows our standards
- Good understanding of the codebase and project goals
- Positive interactions with the community

**Timeline:** This stage typically lasts 3-6 months of active contribution, but can vary based on the nature and quality of contributions.

#### Stage 2: Trusted Contributor

**Focus:** Demonstrate leadership and deeper engagement

**What to do:**
- Take ownership of complex issues and features
- Help review pull requests from other contributors
- Mentor new contributors and answer questions
- Propose and implement significant improvements
- Participate actively in technical discussions
- Help triage issues and improve project organization
- Contribute to Tier 1-2 changes with scientific validation

**What we look for:**
- Ability to review code effectively and provide constructive feedback
- Leadership in technical discussions and decision-making
- Mentoring and community-building skills
- Deep understanding of the project's scientific and technical aspects
- Reliability and consistency in contributions

**Timeline:** This stage typically lasts 6-12 months, during which you build trust and demonstrate your commitment.

#### Stage 3: Maintainer Candidate

**Focus:** Express interest and demonstrate readiness

**What to do:**
- Continue all activities from Stage 2
- Express interest in becoming a maintainer to current maintainers or the Open Science Lead
- Take on additional responsibilities when asked
- Help with release management and project maintenance tasks
- Participate in maintainer discussions (as invited)

**What we look for:**
- Proven track record of quality contributions and reviews
- Strong alignment with project values and goals
- Ability to work independently and make sound technical decisions
- Excellent communication and collaboration skills
- Commitment to the project's long-term success

**Process:**
1. **Expression of Interest:** You can express interest directly to current maintainers, the Open Science Lead, or by opening a thread in the [Governance category on GitHub Discussions](https://github.com/BioPAL/BPS/discussions/categories/governance)
2. **Evaluation:** Current maintainers will review your contributions, community engagement, and technical expertise
3. **Discussion:** Maintainers will discuss your candidacy internally
4. **Nomination:** If there's consensus, a maintainer will nominate you
5. **Approval:** The nomination is reviewed by the Steering Council and ESA representative
6. **Onboarding:** Once approved, you'll receive maintainer permissions and onboarding support

**Timeline:** The evaluation and approval process typically takes 2-4 weeks after expression of interest.

### Key Qualities We Value

While there's no strict checklist, successful maintainers typically demonstrate:

**Technical Excellence:**
- Deep understanding of the codebase and architecture
- Strong software engineering skills
- Knowledge of Earth observation and SAR processing (for scientific aspects)
- Ability to write clear, maintainable code
- Understanding of testing, validation, and quality assurance

**Community Leadership:**
- Positive, respectful communication
- Ability to mentor and guide others
- Constructive feedback and code review skills
- Conflict resolution abilities
- Commitment to inclusivity and diversity

**Project Commitment:**
- Long-term dedication to the project
- Availability for reviews and maintenance tasks
- Alignment with project goals and values
- Understanding of governance and decision-making processes
- Willingness to take on responsibilities

**Scientific Rigor (for scientific aspects):**
- Understanding of validation requirements
- Ability to review scientific changes
- Knowledge of reference datasets and validation methods
- Commitment to reproducibility and transparency

### Maintainer Responsibilities

Once you become a maintainer, you'll take on additional responsibilities:

**Code Review and Merging:**
- Review pull requests across all tiers
- Ensure code quality and standards compliance
- Approve and merge PRs after all requirements are met
- Help resolve conflicts and technical issues

**Project Maintenance:**
- Manage releases and versioning
- Maintain code quality and architecture consistency
- Respond to critical issues and security vulnerabilities
- Manage protected branches and repository settings

**Community Leadership:**
- Mentor new contributors
- Participate in technical and governance meetings
- Represent the project in the community
- Help set technical direction and priorities

**Governance:**
- Participate in maintainer discussions and decisions
- Contribute to governance and policy discussions
- Help ensure compliance with ESA policies and Open Science principles

### How to Get Started

If you're interested in becoming a maintainer:

1. **Start Contributing:** Focus on making quality contributions and building your track record
2. **Engage with the Community:** Participate in discussions, help others, and build relationships
3. **Take on Challenges:** Volunteer for complex issues and significant features
4. **Review Code:** Start reviewing other contributors' pull requests
5. **Express Interest:** When you feel ready, express your interest to current maintainers

**Remember:** There's no rush. Focus on making quality contributions and building trust with the community. Maintainership will come naturally as you demonstrate your commitment and capabilities.

### Questions?

If you have questions about the maintainer path or want to discuss your journey:

- Start a thread in the [Governance category on GitHub Discussions](https://github.com/BioPAL/BPS/discussions/categories/governance)
- Contact current maintainers directly
- Talk to the Open Science Lead
- Ask during Community Meetings or Office Hours

---

<div class="docs-navigation-buttons">
  <a href="https://biomass-disc.info/docs/contributing/practical" class="docs-nav-button docs-nav-button-next">
    <span>Next: Practical Instructions</span>
    <i class="fas fa-arrow-right"></i>
  </a>
</div>

