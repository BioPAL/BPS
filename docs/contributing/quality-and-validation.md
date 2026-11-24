# Quality and validation

Contributor-facing expectations for tests, scientific validation, coding
standards, and documentation. For the full reference, see
[Code standards](code-standards.md), [Documentation standards](documentation-standards.md),
and [Architecture](architecture.md).

Command examples live in [Practical workflow](practical-workflow.md).

---

## Testing requirements

For complete information about test coverage requirements, test types, writing tests, and running tests, please refer to the **[Code standards](code-standards.md)** documentation.

### Key points for contributors

- **Minimum coverage**: ≥ 60% on code touched by the PR (`--cov-fail-under=60`)
- **Test types**: Unit, baseline, extended, heavy: see [Code standards](code-standards.md) for the full marker reference
- **Use pytest**: The project uses pytest for all testing

### Test-driven development (TDD)

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

### Performance testing

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

For detailed commands on how to write tests, run tests, and use performance testing tools, see [Testing: code examples and commands](practical-workflow.md#testing-requirements---code-examples-and-commands)).

---

## Scientific validation

### When validation is required

**Tier 1-2 contributions** must include scientific validation.

### Reference datasets

Validation relies on **TDS (Test Data Sets)**: golden datasets used as a reference to compare processor outputs. Tests are organized in three levels within each module:

- `test/baseline/`: small reference outputs checked on every PR (marker: `baseline`)
- `test/extended/`: broader validation for Tier 1+ (marker: `extended`)
- `test/heavy/`: full scientific regression for Tier 2 (marker: `heavy`)

⚠ The extended and heavy TDS are not distributed with the repository. A procedure to retrieve them will be made available shortly. In the meantime, these pipelines run exclusively on CI servers.

### Validation workflow

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

### Validation report structure

The validation script automatically generates a comprehensive HTML report that includes:

- **Plots for all variables**: Visualizations showing the results for each variable processed
- **Comparison with nominal version**: Side-by-side comparisons and difference plots between the nominal (baseline) version and your new feature
- **Metrics and statistics**: RMSE, bias, correlation, and other relevant metrics for each variable
- **Summary tables**: Tabulated results for easy review

The HTML report is automatically generated by the validation script and attached to your PR. No manual notebook creation is required - the entire validation process is automated.

### Validation report storage

- **HTML Report**: Automatically generated and attached to the PR
- **Report Location**: Available through the CI/CD pipeline artifacts and linked in the PR
- **Metrics Summary**: Key metrics are also included in the PR description for quick reference

### Validation execution

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

## Coding standards

For complete coding standards, naming conventions, formatting rules, type hints, error handling, and logging guidelines, please refer to the **[Code standards](code-standards.md)** documentation.

**Key points:**
- Follow PEP 8 with modifications enforced by `black`
- Use type hints for all public functions
- Run `black`, `ruff`, and `mypy` before committing
- Set up pre-commit hooks for automatic code quality checks

### Backwards compatibility

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

For code examples showing how to implement deprecation warnings, see [Coding standards: code examples](practical-workflow.md#coding-standards---code-examples)).

**When breaking changes are acceptable:**

- **Security vulnerabilities**: Fixing security issues may require breaking changes
- **Scientific correctness**: If the current implementation is scientifically incorrect, breaking changes may be necessary
- **Major architectural improvements**: Significant improvements that benefit the entire project
- **Removing clearly broken or unused features**: Features that are known to be broken or unused

**Always discuss breaking changes** in an issue or pull request before implementing them, and ensure they are clearly documented.

---

## Documentation standards

If you're not the developer type, contributing to the documentation is still of huge value. You don't even have to be an expert on BioPAL to do so! In fact, there are sections of the docs that are worse off after being written by experts. If something in the docs doesn't make sense to you, updating the relevant section after you figure it out is a great way to ensure it will help the next person.

For complete information about documentation standards, docstring format (NumPy style), writing conventions, documentation types, and best practices, please refer to the **[Documentation standards](documentation-standards.md)** documentation.

**Key points:**
- **Docstrings** should follow the NumPy Docstring Standard
- **Standalone documentation** should provide *why* and *when* to use features, with examples and tutorials
- **Keep documentation up to date**: When you change code, update the corresponding documentation
- **Write for your past self**: Document things that weren't obvious to you when you first learned them

### Documentation updates required

When contributing, update:
- User guides (if user-facing changes)
- Developer guides (if workflow changes)
- API documentation (if adding/modifying functions)
- Notebooks (if examples affected)
- Product definitions (if products change)

For detailed commands on how to build and preview documentation, see [Documentation standards: code examples](practical-workflow.md#documentation-standards---code-examples-and-commands)).

---

**Previous:** [CI automation and contribution tiers](ci-automation-and-contribution-tiers.md) | **Next:** [Architecture](architecture.md)
