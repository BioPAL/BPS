# Contributing to BIOMASS BPS - Practical Instructions

This section provides step-by-step instructions with commands and code examples for setting up your environment and implementing contributions.

**Prerequisites:** Make sure you've read [Understanding the Project](https://biomass-disc.info/docs/contributing/overview) first to understand our processes, standards, and expectations.

---

## Table of Contents

- Where to Start?
- Bug Reports and Enhancement Requests
- Version Control, Git, and GitHub
- Getting Started - Setting Up Your Environment
- Development Workflow - Step by Step
- Coding Standards - Code Examples
- Testing Requirements - Code Examples and Commands
- Documentation Standards - Code Examples and Commands

---

## Where to Start?

If you are brand new to BIOMASS BPS or to open source development, the
right entry point is the [Issues tab](https://github.com/BioPAL/BPS/issues)
of the repository.

**Look only at issues that already carry one of the three approval labels:**

- `status:approved`: triaged, scoped, ready to be picked up.
- `good-first-issue`: approved and explicitly suitable for newcomers.
- `help-wanted`: approved and the project actively welcomes external contributions.

An open issue without one of these labels is still under triage or
discussion. Comment on it to express interest, do not start coding yet.
A maintainer will label it once the scope is settled.

If no approved issue matches what you want to work on, open a new one
using the appropriate [issue template](https://github.com/BioPAL/BPS/issues/new/choose):

- **Bug report** for a defect in a processor, the CI, or the documentation.
- **Feature or enhancement request** for a non-scientific feature or tooling improvement.
- **Algorithm proposal** for a new scientific algorithm or a methodological change.
- **Documentation issue** for an error, gap, or improvement in the documentation.
- **Security report** for a non-sensitive security concern.

For open-ended questions, ideas, or discussions that are not yet a
concrete issue, use [GitHub Discussions](https://github.com/BioPAL/BPS/discussions).

---

## Bug Reports and Enhancement Requests

Bug reports are an important part of making BIOMASS BPS more stable.
A complete bug report allows others to reproduce the bug and provides
insight into fixing it.

Trying out the bug-producing code on the `main` branch is often a
worthwhile exercise to confirm that the bug still exists. It is also
worth searching existing bug reports and pull requests to see if the
issue has already been reported and fixed.

### Submitting a Bug Report

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

## Version Control, Git, and GitHub


The code is hosted on GitHub. To contribute you will need to sign up for a [free GitHub account](https://github.com/signup/free). We use [Git](https://git-scm.com/) for version control to allow many people to work together on the project.

### Learning Git

Some great resources for learning Git:

- The [GitHub help pages](https://help.github.com/)
- The [Git documentation](https://git-scm.com/doc)
- [Atlassian's Git tutorials](https://www.atlassian.com/git/tutorials)
- [GitHub's Git Handbook](https://guides.github.com/introduction/git-handbook/)

### Getting Started with Git

[GitHub has instructions for setting up Git](https://help.github.com/set-up-git-redirect) including installing git, setting up your SSH key, and configuring git. All these steps need to be completed before you can work seamlessly between your local repository and GitHub.

**Note:** The following instructions assume you want to learn how to interact with GitHub via the git command-line utility, but contributors who are new to git may find it easier to use other tools instead such as [GitHub Desktop](https://desktop.github.com/).

### Basic Git Configuration

Before you start, make sure Git is configured with your name and email:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

For more information on Git configuration, see the [Git configuration documentation](https://git-scm.com/book/en/v2/Customizing-Git-Git-Configuration).

---

## Getting Started - Setting Up Your Environment


### Prerequisites

- Python 3.12
- Git
- Basic understanding of Earth observation and SAR processing (for scientific contributions)
- Familiarity with Python development tools (for code contributions)

### Setting Up Your Development Environment

1. **Create an account** on [GitHub](https://github.com/) if you do not already have one.

2. **Fork the repository** (if contributing externally):
   - Go to the repository page on GitHub and hit the `Fork` button near the top of the page.
   - This creates a copy of the code under your account on the GitHub server.

3. **Clone your fork** to your machine:
```bash
git clone https://github.com/your-username/biomass-bps.git
cd biomass-bps
git remote add upstream https://github.com/<org>/biomass-bps.git
```

If you're an internal contributor, clone directly:
```bash
git clone https://github.com/<org>/biomass-bps.git
cd biomass-bps
```

4. **Fetch tags** (optional but recommended):
```bash
git fetch --tags upstream
```

5. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Note:** For small changes such as fixing a typo or updating documentation, you don't necessarily need to build and test locally. Push to a feature branch, the CI will run and report any issues. For more complex changes, build and test locally first.

6. **Install a module in development mode** (repeat per module you work on):
```bash
cd bps-<module-name>
pip install -e ".[dev]"
```

7. **Set up pre-commit hooks**:
```bash
pip install pre-commit
pre-commit install --hook-type commit-msg
pre-commit install
```
The `--hook-type commit-msg` installs the DCO hook that checks for `Signed-off-by:` on every commit. To run all hooks manually:
```bash
pre-commit run --all-files
```

8. **Run tests to verify setup**:
```bash
pytest -m unit
```

### Repository Structure

See [architecture.md](../developer-guide/architecture.md#structure-of-a-typical-module) for the full structure. Each `bps-*` module follows this layout:

```
bps-<module>/
├── src/                    # Source code
├── tests/
│   ├── unit/               # Unit tests (pytest -m unit)
│   ├── baseline/           # Baseline tests (pytest -m baseline)
│   ├── extended/           # Extended tests: TDS required (pytest -m extended)
│   └── heavy/              # Heavy tests: TDS required (pytest -m heavy)
└── pyproject.toml
```

---

## Development Workflow - Step by Step

This section provides detailed step-by-step instructions with commands for the development workflow. Make sure you've read the "Development Workflow Overview" section in Part 1 first to understand the concepts.


### Branching Strategy

- **`main`**: Operational/production branch (protected, stable). Promoted to from `release` after ESA approval.
- **`release`**: Release candidate branch (protected). Pre-release validation runs here; Heavy CI is mandatory before promotion to `main`.
- **`develop`**: Main development branch (protected, latest development state).
- **Feature branches**: `feature/description`, `bugfix/issue-id`, `docs/topic`: opened from `develop`.

### Update the develop branch

Before starting a new set of changes, fetch all changes from `upstream/develop` (or `origin/develop` if working directly on the main repository), and start a new feature branch from that. From time to time you should fetch the upstream changes from GitHub:

```bash
git fetch --tags upstream
git merge upstream/develop
```

Or if you're working directly on the main repository:

```bash
git fetch --tags origin
git merge origin/develop
```

This will combine your commits with the latest BioPAL git `develop` branch. If this leads to merge conflicts, you must resolve these before submitting your pull request. If you have uncommitted changes, you will need to `git stash` them prior to updating. This will effectively store your changes, which can be reapplied after updating:

```bash
git stash
git fetch upstream
git merge upstream/develop
git stash pop  # Reapply your changes
```

### Create a new feature branch

Create a branch to save your changes, even before you start making changes. You want your `develop` branch to contain the latest development state, and your `main` branch to contain only production-ready code:

```bash
git checkout -b feature/your-feature-name
```

This changes your working directory to the `feature/your-feature-name` branch. Keep any changes in this branch specific to one bug or feature so it is clear what the branch brings to BioPAL. You can have many feature branches and switch between them using the `git checkout` command.

Generally, you will want to keep your feature branches on your public GitHub fork of BioPAL. To do this, you `git push` this new branch up to your GitHub repo. Generally (if you followed the instructions in these pages, and by default), git will have a link to your fork of the GitHub repo, called `origin`. You push up to your own fork with:

```bash
git push origin feature/your-feature-name
```

In git >= 1.7 you can ensure that the link is correctly set by using the `--set-upstream` option:

```bash
git push --set-upstream origin feature/your-feature-name
```

From now on git will know that `feature/your-feature-name` is related to the `feature/your-feature-name` branch in the GitHub repo.

### The editing workflow

1. **Make some changes** to the code or documentation.

2. **See which files have changed** with `git status`. You'll see a listing like this one:

```bash
# On branch feature/your-feature-name
# Changes not staged for commit:
#   (use "git add <file>..." to update what will be committed)
#   (use "git checkout -- <file>..." to discard changes in working directory)
#
#  modified:   src/biomass_l2_core/algorithms.py
#  modified:   tests/unit/test_algorithms.py
#
# Untracked files:
#   (use "git add <file>..." to include in what will be committed)
#
#  new_file.py
```

3. **Check what the actual changes are** with `git diff`:

```bash
git diff  # Shows unstaged changes
git diff --staged  # Shows staged changes
git diff src/biomass_l2_core/algorithms.py  # Shows changes in a specific file
```

4. **Build the documentation** (for documentation changes) - see the "Documentation Standards - Code Examples and Commands" section in Part 2 for details.

5. **Run the test suite** (for code changes) - see the "Testing Requirements - Code Examples and Commands" section in Part 2 for details.

### Step-by-Step Contribution Process

1. **Create a feature branch** (see above):
```bash
git checkout develop
git pull origin develop  # or git pull upstream develop if using fork
git checkout -b feature/your-feature-name
```

2. **Make your changes** (see "The editing workflow" above):
   - Write code following our [Code Standards](https://biomass-disc.info/docs/code-standards)
   - Add or update tests
   - Update documentation as needed
   - Use `git status` and `git diff` to review your changes

3. **Run local checks**:
   
   **Note:** If you've set up pre-commit hooks (recommended), many checks will run automatically when you commit. You can also run them manually:
   
```bash
# Run all pre-commit hooks manually
pre-commit run --all-files

# Or run individual checks:
# Format code
black src/ tests/

# Lint
ruff check --fix src/ tests/

# Type checking
mypy src/

# Run unit tests
pytest -m unit

# Build documentation (if docs/api/conf.py exists)
sphinx-build docs/api docs/_build
```

4. **Commit your changes**:
If you have created a new file, it is not being tracked by git. Add it by typing:
```bash
git add path/to/file-to-be-added.py
```

Or add all modified files:
```bash
git add .  # Adds all changes
# or
git add src/ tests/  # Adds specific directories
```

Doing `git status` again should show your files as staged for commit.

Now you can commit your changes in your local repository. The `-s` flag adds the required `Signed-off-by:` DCO trailer:
```bash
git commit -s -m "Description of your changes"
```

**Commit message guidelines:**
- A subject line with < 72 characters
- One blank line
- Optionally, a commit message body
- Use clear, descriptive messages
- Reference related issues: `Fixes #123` or `Related to #456` (you can also use `GH1234` format)
- Keep commits focused and atomic
- Separate style fixes into a different commit to make your pull request more readable

Example of a good commit message:
```
Fix biomass calculation for edge cases

This commit fixes an issue where biomass calculation would fail
for incidence angles below 20 degrees. The fix includes proper
boundary checking and adds corresponding unit tests.

Fixes #123
```

5. **Push and create a Pull Request**:
```bash
git push origin feature/your-feature-name
```

Then open a Pull Request using the template defined in [`https://github.com/BioPAL/BPS/blob/main/.github/PULL_REQUEST_TEMPLATE.md`](https://github.com/BioPAL/BPS/blob/main/.github/PULL_REQUEST_TEMPLATE.md).

### Updating Your Pull Request

If you need to make more changes after submitting your PR, you can update your branch and push the changes. The pull request will automatically be updated with the latest code and restart the Continuous Integration tests:

```bash
git push origin feature/your-feature-name
```

This will automatically update your pull request with the latest code and restart the Continuous Integration tests.

### Delete Your Merged Branch (Optional)

Once your feature branch is accepted and merged into `develop`, you'll probably want to get rid of the branch. First, update your `develop` branch to check that the merge was successful:

```bash
git fetch upstream  # or git fetch origin if working directly on main repo
git checkout develop
git merge upstream/develop  # or git merge origin/develop
```

Then you can delete the local branch:

```bash
git branch -d feature/your-feature-name
```

If the branch was squashed into a single commit before merging, you may need to use an upper-case `-D`:

```bash
git branch -D feature/your-feature-name
```

Be careful with this because `git` won't warn you if you accidentally delete an unmerged branch.

If you didn't delete your branch using GitHub's interface, then it will still exist on GitHub. To delete it there:

```bash
git push origin --delete feature/your-feature-name
```

---

## Coding Standards - Code Examples

This section provides detailed code examples for implementing coding standards. For the conceptual overview, see the "Coding Standards" section in Part 1.

### Implementing Deprecation Warnings

**Be especially careful when changing function and method signatures**, because any change may require a deprecation warning. For example, if your pull request means that the argument `old_arg` to `func` is no longer valid, instead of simply raising an error if a user passes `old_arg`, we would instead catch it and emit a deprecation warning:

```python
import warnings

def func(new_arg, old_arg=None):
    """
    Calculate biomass using new algorithm.
    
    Parameters
    ----------
    new_arg : float
        New parameter for calculation
    old_arg : float, optional
        Deprecated. Use new_arg instead. Will be removed in version 2.0.
    """
    if old_arg is not None:
        warnings.warn(
            "`old_arg` has been deprecated and will be removed in version 2.0. "
            "Please use `new_arg` from now on.",
            DeprecationWarning,
            stacklevel=2
        )
        # Still do what the user intended here for backwards compatibility
        # Or map old_arg to new_arg if possible
        new_arg = old_arg
    
    # Continue with new implementation
    return calculate_with_new_arg(new_arg)
```

**Example of a complete deprecation cycle:**

```python
# Version 1.0 - Original function
def calculate_biomass_v1(sar_data, config):
    """Calculate biomass using original algorithm."""
    # Original implementation
    pass

# Version 1.5 - Add deprecation warning
def calculate_biomass_v1(sar_data, config):
    """
    Calculate biomass using original algorithm.
    
    .. deprecated:: 1.5
        Use :func:`calculate_biomass_v2` instead. This function will be
        removed in version 2.0.
    """
    warnings.warn(
        "calculate_biomass_v1 is deprecated and will be removed in version 2.0. "
        "Use calculate_biomass_v2 instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # Original implementation still works
    pass

# Version 2.0 - Remove deprecated function
# calculate_biomass_v1 is removed, only calculate_biomass_v2 exists
```

---

## Testing Requirements - Code Examples and Commands

This section provides detailed code examples and commands for testing. For the conceptual overview, see the "Testing Requirements" section in Part 1.

### Writing Tests - Code Examples

All tests should go into the `tests/` subdirectory of the relevant `bps-*` module, under the appropriate subfolder (`unit/`, `baseline/`, `extended/`, `heavy/`). Look at existing tests for inspiration.

BIOMASS BPS uses [pytest](https://doc.pytest.org/en/latest/) with markers to classify tests. Every test must be decorated with the appropriate marker (defined in `pytest.ini`).

**Basic unit test example:**
```python
import pytest
import numpy as np

@pytest.mark.unit
def test_calculate_biomass_basic():
    """Test basic biomass calculation."""
    from bps_l2b_agb_processor.algorithms import calculate_biomass

    sar_data = np.array([0.1, 0.2, 0.3])
    incidence_angle = 30.0

    result = calculate_biomass(sar_data, incidence_angle)

    assert result.shape == sar_data.shape
    assert np.all(result >= 0)
```

**Using pytest.parametrize for multiple test cases:**
```python
import pytest
import numpy as np

@pytest.mark.unit
@pytest.mark.parametrize("incidence_angle", [20.0, 30.0, 40.0, 50.0])
def test_calculate_biomass_angles(incidence_angle):
    """Test biomass calculation with different incidence angles."""
    from bps_l2b_agb_processor.algorithms import calculate_biomass

    sar_data = np.array([0.1, 0.2, 0.3])

    result = calculate_biomass(sar_data, incidence_angle)

    assert result.shape == sar_data.shape
    assert np.all(result >= 0)
    assert np.all(np.isfinite(result))
```

**Using fixtures for reusable test data:**
```python
import pytest
import numpy as np

@pytest.fixture
def sample_sar_data():
    """Fixture providing sample SAR data for tests."""
    return np.array([0.1, 0.2, 0.3, 0.4, 0.5])

@pytest.mark.unit
def test_calculate_biomass_with_fixtures(sample_sar_data):
    """Test using fixtures for test data."""
    from bps_l2b_agb_processor.algorithms import calculate_biomass

    result = calculate_biomass(sample_sar_data, incidence_angle=30.0)

    assert result.shape == sample_sar_data.shape
    assert np.all(result >= 0)
```

**Testing with expected exceptions:**
```python
import pytest
import numpy as np

@pytest.mark.unit
def test_calculate_biomass_invalid_input():
    """Test that invalid input raises appropriate error."""
    from bps_l2b_agb_processor.algorithms import calculate_biomass

    sar_data = np.array([-0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="SAR data contains negative values"):
        calculate_biomass(sar_data, incidence_angle=30.0)
```

### Running Tests - Commands

Tests are run using pytest markers. Run from the root of the relevant `bps-*` module (after installing it with `pip install -e ".[dev]"`):

```bash
# Unit tests (no external data required)
pytest -m unit

# Baseline tests (no external data required)
pytest -m baseline

# Extended tests: requires TDS dataset
# ⚠ WARNING: TDS not distributed with the repository. A retrieval procedure will be made available shortly.
pytest -m extended

# Heavy tests: requires TDS dataset
# ⚠ WARNING: TDS not distributed with the repository. A retrieval procedure will be made available shortly.
pytest -m heavy
```

**Running specific tests:**

```bash
# Run tests matching a pattern
pytest -m unit -k "test_calculate_biomass"

# Run a specific test file
pytest tests/unit/test_algorithms.py

# Run a specific test function
pytest tests/unit/test_algorithms.py::test_calculate_biomass_basic
```

**Running tests with coverage:**

```bash
# Generate coverage report (minimum threshold: 60%)
pytest -m unit --cov=src --cov-report=html --cov-fail-under=60

# View coverage in terminal
pytest -m unit --cov=src --cov-report=term-missing
```

**Running tests in parallel:**

Using [pytest-xdist](https://pypi.python.org/pypi/pytest-xdist), one can speed up local testing on multicore machines by running pytest with the optional `-n` argument:

```bash
pytest -n 4  # Run tests using 4 parallel workers
```

This can significantly reduce the time it takes to locally run tests before submitting a pull request.

**Other useful pytest options:**

```bash
# Stop at first failure
pytest -x

# Show print statements
pytest -s

# Verbose output
pytest -v

# Show local variables in tracebacks
pytest -l

# Run only tests marked with a specific marker
pytest -m "slow"  # If you have @pytest.mark.slow markers

# Run tests and show the slowest 10 tests
pytest --durations=10
```

For more information, see the [pytest documentation](https://doc.pytest.org/en/latest/).

### Running the Performance Test Suite - Commands

**Using pytest-benchmark (if available):**

If the project uses [pytest-benchmark](https://pytest-benchmark.readthedocs.io/), you can run performance benchmarks with:

```bash
# Run all benchmarks
pytest --benchmark-only

# Run benchmarks and compare with previous runs
pytest --benchmark-compare

# Run specific benchmark tests
pytest tests/benchmarks/ -k "biomass_calculation"

# Run benchmarks and save results
pytest --benchmark-only --benchmark-json=benchmark_results.json
```

**Manual performance testing:**

For manual performance testing, you can use Python's `timeit` module or create simple timing scripts:

```python
import time
import numpy as np
from biomass_l2_core.algorithms import calculate_biomass

# Time a function call
start = time.time()
result = calculate_biomass(sar_data, angle, config)
elapsed = time.time() - start
print(f"Calculation took {elapsed:.3f} seconds")
```

**Profiling code:**

To identify performance bottlenecks, you can use profiling tools:

```bash
# Using cProfile
python -m cProfile -o profile.stats your_script.py

# Using line_profiler (if installed)
kernprof -l -v your_script.py

# Using memory_profiler (if installed)
python -m memory_profiler your_script.py
```

---

## Documentation Standards - Code Examples and Commands

For complete documentation standards, including docstring format (NumPy style), documentation types, writing conventions for Markdown and reStructuredText, code examples, and commands for building and previewing documentation, please refer to the **[Documentation Standards](https://biomass-disc.info/docs/documentation-standards)** documentation.

---

**Previous:** [Understanding the Project](process.md) | **Next:** [Help & Resources](templates.md)

