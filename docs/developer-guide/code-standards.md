# Code Standards

Complete coding standards for BioPAL, including naming conventions, formatting rules, type hints, and test requirements.

---

## Table of Contents

1. General Guidelines
2. Naming Conventions
3. Formatting Rules
4. Type Hints
5. Test Requirements
6. Documentation Standards
7. Error Handling
8. Logging

---

## General Guidelines

### Quick Reference Table

The following table provides a quick reference for code conventions:

| Element | Convention | Example |
|---------|------------|---------|
| **Variables** | snake_case | `biomass_value`, `incidence_angle` |
| **Functions** | snake_case, verb-based | `calculate_biomass()`, `process_l2_product()` |
| **Classes** | PascalCase | `BiomassProcessor`, `SARDataLoader` |
| **Constants** | UPPER_SNAKE_CASE | `MAX_ITERATIONS`, `DEFAULT_THRESHOLD` |
| **Modules** | snake_case | `biomass_processor.py`, `sar_utils.py` |
| **Type hints** | Required | `def process(data: np.ndarray) -> float:` |
| **Docstrings** | NumPy style | `"""Description.\n\nParameters\n----------\n..."""` |
| **Imports** | Grouped and sorted | `import numpy as np` then `from . import utils` |
| **Line length** | Max 120 characters | Use black for formatting |
| **Tests** | Prefix `test_` | `test_calculate_biomass()` |

### Core Principles

**Clear Naming:**
- Use descriptive names for variables, functions, and modules
- Avoid abbreviations unless widely understood
- Be consistent across the codebase

**Pure Functions:**
- Prefer functions that take explicit inputs and return outputs
- Avoid modifying global state
- Make functions testable and predictable

**Licensing Compliance:**
- All code must be compatible with Apache License 2.0
- External dependencies must be license-compatible
- Third-party code must be properly attributed
- See the [Licensing documentation](https://biomass-disc.info/docs/licensing) for complete requirements

**DRY Principle:**
- Don't Repeat Yourself
- Avoid duplication by refactoring shared logic
- Extract common patterns into utilities

**Single Responsibility:**
- Keep functions focused on one task
- Keep modules focused on one domain
- Separate concerns clearly

**Readability:**
- Code should be self-documenting
- Use comments for "why", not "what"
- Prefer clear code over clever code

---

## Naming Conventions

### Variables

**Snake Case:**
```python
# Good
biomass_value = 125.5
incidence_angle = 30.0
sar_backscatter = np.array([0.1, 0.2, 0.3])

# Avoid
biomassValue = 125.5  # camelCase
incidenceAngle = 30.0
SARBackscatter = np.array([0.1, 0.2, 0.3])  # PascalCase
```

**Descriptive Names:**
```python
# Good
above_ground_biomass = calculate_agb(sar_data)
canopy_height_map = process_height_data(lidar_data)

# Avoid
agb = calc(sd)  # Too abbreviated
chm = proc(ld)  # Unclear
```

**Constants:**
```python
# UPPER_SNAKE_CASE for constants
MAX_ITERATIONS = 100
DEFAULT_THRESHOLD = 0.5
EARTH_RADIUS_KM = 6371.0
```

### Functions

**Snake Case, Verb-Based:**
```python
# Good
def calculate_biomass(sar_data, config):
    """Calculate biomass from SAR data."""
    pass

def process_l2_product(input_file, output_file):
    """Process L2 product."""
    pass

# Avoid
def BiomassCalc(sar_data, config):  # PascalCase
    pass

def processL2(input_file, output_file):  # camelCase
    pass
```

**Action-Oriented Names:**
```python
# Good
def validate_input_data(data):
    """Validate input data."""
    pass

def load_configuration(file_path):
    """Load configuration from file."""
    pass

# Avoid
def data_validation(data):  # Noun instead of verb
    pass

def config(file_path):  # Too generic
    pass
```

### Classes

**PascalCase:**
```python
# Good
class BiomassProcessor:
    """Process biomass data."""
    pass

class ConfigurationManager:
    """Manage configuration."""
    pass

# Avoid
class biomass_processor:  # snake_case
    pass

class ConfigMgr:  # Abbreviation
    pass
```

### Modules and Packages

**Snake Case, Lowercase:**
```python
# Good
biomass_l2_core/
biomass_retrieval.py
uncertainty_quantification.py

# Avoid
BiomassL2Core/  # PascalCase
BiomassRetrieval.py  # PascalCase
```

### Private Functions and Variables

**Leading Underscore:**
```python
# Good
def _internal_helper_function():
    """Internal helper (not part of public API)."""
    pass

_internal_variable = 42

# Public API
def public_function():
    """Public function."""
    _internal_helper_function()
```

---

## Formatting Rules

### Python Style Guide

We follow **PEP 8** with modifications enforced by `black`:

**Line Length:** 120 characters (configured in `pyproject.toml` and `ruff.toml`)

```python
# Good: Break long lines appropriately
def calculate_biomass(
    sar_data: np.ndarray,
    incidence_angle: float,
    config: dict,
    aux_data: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate biomass."""
    pass

# Avoid: Lines exceeding 120 characters without breaking
def calculate_biomass(sar_data: np.ndarray, incidence_angle: float, config: dict, aux_data: Optional[np.ndarray] = None, extra_param: Optional[str] = None) -> np.ndarray:
    pass
```

**Indentation:** 4 spaces (no tabs)

```python
# Good: 4 spaces
if condition:
    do_something()
    if nested_condition:
        do_nested()

# Avoid: Tabs or inconsistent indentation
if condition:
	do_something()  # Tab
    do_other()  # Mixed
```

**Blank Lines:**
- 2 blank lines between top-level functions and classes
- 1 blank line between methods in a class
- Use blank lines to separate logical sections

```python
# Good
import numpy as np
import xarray as xr


def function_one():
    """First function."""
    pass


def function_two():
    """Second function."""
    pass


class MyClass:
    """My class."""
    
    def method_one(self):
        """First method."""
        pass
    
    def method_two(self):
        """Second method."""
        pass
```

**Imports:**
- Group imports: standard library, third-party, local
- Sort imports alphabetically within groups
- Use `ruff` for automatic import sorting

```python
# Good
import logging
from typing import Dict, Optional

import numpy as np
import xarray as xr

from biomass_l2_core.algorithms import calculate_biomass
from biomass_l2_io.readers import read_l1_product
```

### Code Formatting Tools

**black** (v24.10.0):
- Automatic code formatter: handles indentation, quotes, trailing commas, blank lines, and more
- Line length: 120 characters (read from `pyproject.toml`, no need to pass `--line-length`)
- Consistent style across codebase

```bash
# Format code
black src/ tests/

# Check formatting without modifying (CI)
black --check src/ tests/
```

**ruff** (v0.9.1):
- Fast linting and import sorting
- Style checks and auto-fix
- Docstring code formatting enabled (`docstring-code-format = true` in `ruff.toml`)

```bash
# Lint and auto-fix
ruff check --fix src/ tests/

# Check only
ruff check src/ tests/
```

**mypy** (v1.14.1):
- Static type checking
- Catches type errors before runtime
- Configuration (in `pyproject.toml`):
  - `python_version = "3.12"`
  - `explicit_package_bases = true`
  - `namespace_packages = true`

```bash
# Type check (reads config from pyproject.toml)
mypy src/
```

### Pre-commit Configuration

Pre-commit hooks automatically enforce formatting and code quality before commits. This ensures consistent code style and catches common issues early.

#### Setup

After cloning the repository and installing dependencies:

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg  # Required for the DCO check
```

This installs git hooks that run automatically on every commit.

#### Configuration

The actual pinned versions in `.pre-commit-config.yaml`:

| Hook | Version |
|------|---------|
| black | 24.10.0 |
| ruff | v0.9.1 |
| mypy | v1.14.1 |
| detect-secrets | v1.5.0 |
| reuse | v5.0.2 |

#### What Pre-commit Checks

**General checks** (pre-commit standard hooks):
- `trailing-whitespace`: removes trailing whitespace
- `end-of-file-fixer`: ensures files end with a newline
- `check-yaml`: YAML validation (excludes `recipe/meta.yaml`)
- `check-added-large-files`: blocks files larger than 10MB
- `check-merge-conflict`: detects merge conflict markers
- `detect-private-key`: blocks accidental private key commits

**Python code quality:**
- **black** (v24.10.0): Automatic code formatting
- **ruff** (v0.9.1): Fast linting and import sorting, with `--fix`
- **mypy** (v1.14.1): Type checking (excludes tests, docs, noxfile, recipe, aux_pp2_models)
- **detect-secrets** (v1.5.0): Prevents committing secrets/credentials (baseline: `.secrets.baseline`)

**License compliance:**
- **reuse** (v5.0.2): Verifies SPDX headers on all source files

**DCO (commit-msg stage):**
- `check-dco-commit-msg.sh`: Verifies `Signed-off-by:` trailer on every commit message

#### Running Pre-commit Manually

To run all hooks on all files:
```bash
pre-commit run --all-files
```

To run on staged files only:
```bash
pre-commit run
```

To run a specific hook:
```bash
pre-commit run black --all-files
pre-commit run ruff --all-files
```

---

## DCO: Developer Certificate of Origin

Every commit (excluding merge commits) must carry a `Signed-off-by:` trailer. This is verified locally by the `check-dco-commit-msg.sh` hook (commit-msg stage) and by the `baseline-dco` job in the CI pipeline.

**Sign a commit:**
```bash
git commit -s -m "feat: my feature"
```

**Enable automatic signing for all commits:**
```bash
git config format.signoff true
```

**Remediation if you forgot to sign:**
```bash
git commit --amend --signoff
git push --force-with-lease
```

For multiple unsigned commits, use interactive rebase:
```bash
git rebase --signoff HEAD~<number-of-commits>
git push --force-with-lease
```

---

## REUSE / SPDX Compliance

Every new source file must include an SPDX header. The `reuse` pre-commit hook and the `baseline-reuse` CI job enforce this: a PR with non-compliant files is **blocked**.

**Python:**
```python
# SPDX-FileCopyrightText: 2026 [Your Organisation]
#
# SPDX-License-Identifier: Apache-2.0
```

**YAML / shell:**
```yaml
# SPDX-FileCopyrightText: 2026 [Your Organisation]
# SPDX-License-Identifier: Apache-2.0
```

License texts are stored in `LICENSES/Apache-2.0.txt` and `LICENSES/MIT.txt` at the repository root.

**Check compliance locally:**
```bash
pre-commit run reuse --all-files
# or directly:
reuse lint
```

Standard: https://reuse.software/

---

#### Skipping Hooks (Not Recommended)

If you need to skip hooks for a specific commit (not recommended):
```bash
git commit --no-verify -m "Emergency fix"
```

**Note:** Skipping hooks may cause CI to fail. Always fix issues before pushing.

#### Updating Hooks

To update hook versions:
```bash
pre-commit autoupdate
```

#### Troubleshooting

If hooks fail:
1. Review the error messages
2. Most hooks auto-fix issues (black, ruff-format)
3. Fix remaining issues manually
4. Re-run: `pre-commit run --all-files`

If you encounter issues with a specific hook, you can temporarily disable it by commenting it out in `.pre-commit-config.yaml`.

---

## Type Hints

### Requirements

**All public functions must have type hints:**

```python
# Good: Complete type hints
from typing import Dict, List, Optional, Tuple
import numpy as np
import xarray as xr

def process_data(
    input_file: str,
    parameters: Dict[str, float],
    output_format: Optional[str] = None
) -> xr.Dataset:
    """Process input data."""
    pass

# Avoid: Missing type hints
def process_data(input_file, parameters, output_format=None):
    """Process input data."""
    pass
```

### Type Hint Examples

**Basic Types:**
```python
def calculate_value(x: float, y: int) -> float:
    """Calculate value."""
    return x * y
```

**Collections:**
```python
from typing import List, Dict, Tuple, Optional

def process_list(items: List[str]) -> List[int]:
    """Process list of strings."""
    pass

def get_config() -> Dict[str, float]:
    """Get configuration."""
    pass

def get_coordinates() -> Tuple[float, float]:
    """Get coordinates."""
    pass

def find_item(key: str) -> Optional[str]:
    """Find item, may return None."""
    pass
```

**NumPy Arrays:**
```python
import numpy as np
from numpy.typing import NDArray

def process_array(data: NDArray[np.float64]) -> NDArray[np.float64]:
    """Process numpy array."""
    pass
```

**xarray:**
```python
import xarray as xr

def process_dataset(data: xr.Dataset) -> xr.Dataset:
    """Process xarray dataset."""
    pass

def get_variable(ds: xr.Dataset, name: str) -> xr.DataArray:
    """Get variable from dataset."""
    pass
```

**Complex Types:**
```python
from typing import Union, Callable, Any

def flexible_input(value: Union[str, int, float]) -> str:
    """Accept multiple types."""
    pass

def apply_function(
    data: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Apply function to data."""
    pass
```

**Generic Types:**
```python
from typing import TypeVar, Generic

T = TypeVar('T')

def process_generic(item: T) -> T:
    """Process generic type."""
    pass
```

### Type Checking

**Running mypy:**
```bash
# Type checking (reads config from pyproject.toml)
mypy src/
```

**Common Issues:**
- Missing type hints: Add type hints to all public functions
- Incorrect types: Fix type annotations
- Import errors: Install type stubs (e.g. `types-PyYAML` is already included as a dependency)

---

## Test Requirements

### Test Coverage

**Minimum Coverage:** ≥ 60% on code touched by the PR.

This is enforced by the CI pipeline (`--cov-fail-under=60`).

```bash
# Run tests with coverage
pytest --cov=src --cov-report=html

# Check coverage threshold (matches CI)
pytest --cov=src --cov-fail-under=60
```

### Test Markers

All markers are declared in `pytest.ini`. Use them to classify tests and control which pipeline stage runs them:

| Marker | Pipeline stage | Purpose |
|--------|---------------|---------|
| `unit` | Baseline (every PR) | Fast, isolated tests of individual functions |
| `baseline` | Baseline (every PR) | Quick sanity checks: output must match reference |
| `smoke` | Extended (Tier 1+) | Fast end-to-end sanity run |
| `integration` | Extended (Tier 1+) | Full workflow tests with realistic data |
| `extended` | Extended (Tier 1+) | Broader coverage for code changes |
| `heavy` | Heavy (Tier 2) | Full scientific regression, requires TDS dataset |
| `public` |: | Public API surface tests (legacy compatibility) |

```python
# Example: tagging a test for the baseline pipeline
@pytest.mark.baseline
def test_output_matches_reference():
    ...

# Example: tagging a slow test for Tier 2 only
@pytest.mark.heavy
def test_full_scientific_regression():
    ...
```

### Test Types

#### Unit Tests

**Purpose:** Test individual functions and classes in isolation

**Location:** `tests/unit/`

**Example:**
```python
import pytest
import numpy as np
from biomass_l2_core.algorithms import calculate_biomass

def test_calculate_biomass_basic():
    """Test basic biomass calculation."""
    sar_data = np.array([0.1, 0.2, 0.3])
    incidence_angle = 30.0
    config = {"threshold": 0.5}
    
    result = calculate_biomass(sar_data, incidence_angle, config)
    
    assert result.shape == sar_data.shape
    assert np.all(result >= 0)
    assert np.all(result <= 500)  # Max biomass constraint

def test_calculate_biomass_edge_cases():
    """Test edge cases."""
    # Test with zero values
    sar_data = np.array([0.0, 0.1, 0.2])
    result = calculate_biomass(sar_data, 30.0, {"threshold": 0.5})
    assert np.all(result >= 0)
    
    # Test with invalid input
    with pytest.raises(ValueError):
        calculate_biomass(np.array([-1.0]), 30.0, {"threshold": 0.5})
```

**Best Practices:**
- Test one thing per test function
- Use descriptive test names
- Test edge cases and error conditions
- Use fixtures for test data
- Keep tests fast and independent

#### Integration Tests

**Purpose:** Test end-to-end workflows

**Location:** `tests/integration/`

**Example:**
```python
import pytest
from pathlib import Path
from biomass_l2_orchestration.workflows import process_l2_chain

def test_end_to_end_processing(tmp_path):
    """Test complete processing workflow."""
    input_file = "tests/data/l1_product.nc"
    output_file = tmp_path / "output.nc"
    config_file = "tests/data/config.yaml"
    
    process_l2_chain(input_file, output_file, config_file)
    
    assert output_file.exists()
    # Verify output format and content
    # Check metadata
    # Validate scientific results
```

**Best Practices:**
- Use realistic test data
- Test complete workflows
- Verify output formats
- Clean up test artifacts

#### Scientific Regression Tests

**Purpose:** Ensure no scientific regressions

**Location:** `tests/scientific_regression/`

**Example:**
```python
import pytest
import numpy as np
from biomass_l2_core.algorithms import calculate_biomass

def test_biomass_regression():
    """Test against reference results."""
    # Load reference data
    reference_results = np.load("tests/data/reference_biomass.npy")
    
    # Process with current code
    sar_data = np.load("tests/data/test_sar_data.npy")
    result = calculate_biomass(sar_data, 30.0, {"threshold": 0.5})
    
    # Compare with reference
    np.testing.assert_allclose(
        result,
        reference_results,
        rtol=1e-5,
        atol=0.1
    )
```

**Best Practices:**
- Use validated reference datasets
- Compare with previous versions
- Check for scientific accuracy
- Document reference data sources

### Writing Tests

**Test Structure:**
```python
def test_function_name_description():
    """
    Test description.
    
    What is being tested and why.
    """
    # Arrange: Set up test data
    input_data = create_test_data()
    config = {"param": 1.0}
    
    # Act: Execute function
    result = function_under_test(input_data, config)
    
    # Assert: Verify results
    assert result is not None
    assert result.shape == expected_shape
    assert np.all(result >= 0)
```

**Test Fixtures:**
```python
import pytest

@pytest.fixture
def sample_sar_data():
    """Fixture for sample SAR data."""
    return np.array([0.1, 0.2, 0.3, 0.4])

@pytest.fixture
def default_config():
    """Fixture for default configuration."""
    return {
        "threshold": 0.5,
        "max_iterations": 100
    }

def test_with_fixtures(sample_sar_data, default_config):
    """Test using fixtures."""
    result = process_data(sample_sar_data, default_config)
    assert result is not None
```

**Parametrized Tests:**
```python
@pytest.mark.parametrize("input_value,expected", [
    (0.1, 10.0),
    (0.2, 20.0),
    (0.3, 30.0),
])
def test_multiple_cases(input_value, expected):
    """Test multiple input/output pairs."""
    result = calculate_value(input_value)
    assert result == expected
```

### Running Tests

```bash
# Unit tests (run locally before pushing)
pytest -m unit

# Baseline tests (mirrors the CI baseline pipeline)
pytest -m baseline

# Extended tests (Tier 1: code changes)
# ⚠ WARNING: extended tests require a TDS dataset not distributed with the repository.
# A procedure to retrieve the TDS will be made available shortly.
pytest -m extended

# Heavy tests (Tier 2: scientific changes)
# ⚠ WARNING: heavy tests require a TDS dataset not distributed with the repository.
# A procedure to retrieve the TDS will be made available shortly.
pytest -m heavy

# Specific test file
pytest tests/unit/test_algorithms.py

# Specific test function
pytest tests/unit/test_algorithms.py::test_calculate_biomass_basic

# With coverage (matches CI threshold)
pytest -m unit --cov=src --cov-report=html --cov-fail-under=60

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Test Requirements Checklist

**Before Submitting PR:**
- [ ] Unit tests added for new functions
- [ ] Integration tests for new workflows
- [ ] Scientific regression tests (if applicable)
- [ ] Test coverage meets minimum requirements
- [ ] All tests pass locally
- [ ] Tests are fast and independent
- [ ] Test data is included or documented

---

## Documentation Standards

### Docstring Format

**NumPy Style (Required):**

```python
def calculate_biomass(
    sar_data: np.ndarray,
    incidence_angle: float,
    config: dict
) -> np.ndarray:
    """
    Calculate above-ground biomass from SAR backscatter data.
    
    This function implements the inversion algorithm described in
    [Reference Paper, 2023]. The algorithm uses P-band SAR data
    to estimate forest biomass with uncertainty quantification.
    
    Parameters
    ----------
    sar_data : np.ndarray
        Input SAR backscatter data in linear scale.
        Shape: (n_pixels,) or (n_azimuth, n_range)
        Expected range: [0.0, 1.0]
    incidence_angle : float
        Incidence angle in degrees.
        Range: [20, 60]
    config : dict
        Configuration parameters containing:
        - threshold: float, detection threshold (default: 0.5)
        - max_iterations: int, maximum iterations (default: 100)
        - biomass_range: tuple, (min, max) in Mg/ha (default: (0, 500))
        
    Returns
    -------
    np.ndarray
        Calculated biomass values in Mg/ha.
        Shape: matches sar_data shape
        Range: [0, 500] Mg/ha
        
    Raises
    ------
    ValueError
        If sar_data contains negative values or out-of-range data
    RuntimeError
        If convergence not achieved within max_iterations
        
    References
    ----------
    .. [1] Author et al. (2023). "Biomass Estimation from P-band SAR".
           Remote Sensing Journal.
    
    Examples
    --------
    >>> import numpy as np
    >>> sar_data = np.array([0.1, 0.2, 0.3])
    >>> angle = 30.0
    >>> config = {"threshold": 0.5}
    >>> biomass = calculate_biomass(sar_data, angle, config)
    >>> print(biomass)
    [12.5 25.3 38.1]
    
    Notes
    -----
    The algorithm assumes forested areas. Non-forested pixels
    should be masked before calling this function.
    """
    pass
```

### Documentation Sections

**Required Sections:**
- **Summary**: One-line description
- **Parameters**: All parameters with types and descriptions
- **Returns**: Return value with type and description

**Optional Sections:**
- **Raises**: Exceptions that may be raised
- **See Also**: Related functions
- **References**: Scientific papers or documentation
- **Examples**: Usage examples
- **Notes**: Additional information

### Inline Comments

**Use comments for "why", not "what":**

```python
# Good: Explains why
# Use iterative method because analytical solution is unstable
# for low backscatter values
result = iterative_solve(data, threshold)

# Avoid: States the obvious
# Calculate result
result = calculate(data)
```

---

## Error Handling

### Exception Types

**Use Appropriate Exception Types:**
```python
# ValueError: Invalid input values
if value < 0:
    raise ValueError(f"Value must be non-negative, got {value}")

# TypeError: Wrong type
if not isinstance(data, np.ndarray):
    raise TypeError(f"Expected numpy array, got {type(data)}")

# FileNotFoundError: Missing files
if not Path(file_path).exists():
    raise FileNotFoundError(f"File not found: {file_path}")

# RuntimeError: Algorithm failures
if not converged:
    raise RuntimeError("Algorithm did not converge")
```

### Error Messages

**Informative Error Messages:**
```python
# Good: Clear and helpful
if sar_data.min() < 0:
    raise ValueError(
        f"SAR data contains negative values: min={sar_data.min()}. "
        "Expected range: [0.0, 1.0]"
    )

# Avoid: Vague
if sar_data.min() < 0:
    raise ValueError("Invalid data")
```

### Error Handling Patterns

**Try-Except Blocks:**
```python
try:
    result = process_data(input_file)
except FileNotFoundError:
    logger.error(f"Input file not found: {input_file}")
    raise
except ValueError as e:
    logger.error(f"Invalid input data: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

**Custom Exceptions:**
```python
class BiomassCalculationError(Exception):
    """Error in biomass calculation."""
    pass

class ValidationError(Exception):
    """Error in data validation."""
    pass

# Usage
if invalid_condition:
    raise BiomassCalculationError("Detailed error message")
```

---

## Logging

### Logging Setup

```python
import logging

logger = logging.getLogger(__name__)

def process_data(data):
    """Process data with logging."""
    logger.info("Starting data processing")
    logger.debug(f"Input data shape: {data.shape}")
    
    try:
        result = process(data)
        logger.info("Data processing completed successfully")
        return result
    except Exception as e:
        logger.error(f"Data processing failed: {e}", exc_info=True)
        raise
```

### Log Levels

**Use Appropriate Log Levels:**
```python
# DEBUG: Detailed diagnostic information
logger.debug(f"Processing pixel {i}/{total}")

# INFO: General informational messages
logger.info("Processing started")
logger.info(f"Processed {n_files} files")

# WARNING: Warning messages
logger.warning("Unusual value detected, using default")
logger.warning(f"Low data quality: {quality_score}")

# ERROR: Error messages
logger.error("Processing failed")
logger.error(f"Invalid configuration: {config}")

# CRITICAL: Critical errors
logger.critical("System failure, aborting")
```

### Logging Best Practices

**Include Context:**
```python
# Good: Includes context
logger.info(f"Processing file {file_path} with config {config_name}")

# Avoid: Missing context
logger.info("Processing")
```

**Use Structured Logging:**
```python
# Good: Structured information
logger.info("Processing completed", extra={
    "file": file_path,
    "duration": duration,
    "pixels": n_pixels
})

# Avoid: String concatenation
logger.info(f"Processing completed: file={file_path}, duration={duration}")
```

---

## Code Review Checklist

### Before Submitting

- [ ] Code follows naming conventions
- [ ] Code formatted with `black`
- [ ] Code linted with `ruff`
- [ ] Type hints added to all functions
- [ ] Docstrings added (NumPy style)
- [ ] Tests added and passing
- [ ] Test coverage meets requirements
- [ ] Documentation updated
- [ ] No secrets or sensitive data
- [ ] Pre-commit hooks passing

### Code Quality

- [ ] Functions are focused and short
- [ ] No code duplication
- [ ] Error handling is appropriate
- [ ] Logging is used appropriately
- [ ] Performance considerations documented
- [ ] Code is readable and maintainable

---

## Resources

### Documentation

- [Getting Started](https://biomass-disc.info/docs) - Introduction and getting started guide
- [Licensing](https://biomass-disc.info/docs/licensing) - Apache 2.0 license requirements and legal obligations
- [Code of Conduct](https://biomass-disc.info/docs/code-of-conduct) - Community standards and expectations
- [Contributing Guide](https://biomass-disc.info/docs/contributing) - Contribution process and workflows
- [CI/CD Guide](https://biomass-disc.info/docs/ci-cd-guide) - Pipeline reference, tier detection, branch protection
- [Governance](https://biomass-disc.info/docs/governance) - Roles, responsibilities, and decision-making
- [Architecture](https://biomass-disc.info/docs/architecture) - System architecture and design
- [Architecture](https://biomass-disc.info/docs/architecture) - Monorepo layout and `bps-*` modules
- [Documentation Standards](https://biomass-disc.info/docs/documentation-standards) - Documentation writing standards and best practices
- [Communication](https://biomass-disc.info/docs/communication) - Communication channels and meeting schedules

### External Resources

- [PEP 8](https://www.python.org/dev/peps/pep-0008/) - Python style guide
- [black](https://black.readthedocs.io/) - Code formatter
- [ruff](https://docs.astral.sh/ruff/) - Fast linter
- [mypy](https://mypy.readthedocs.io/) - Type checker
- [pytest](https://docs.pytest.org/) - Testing framework

---

## Code Review Process

The following flowchart illustrates the code review process from writing code to merging:

```{mermaid}
flowchart TD
    Code[Code written] --> Lint[Pre-commit hooks<br/>black, ruff, mypy]
    Lint --> LintOK{Pass?}
    LintOK -->|No| FixLint[Fix errors]
    FixLint --> Lint
    LintOK -->|Yes| Tests[Run local tests]
    
    Tests --> TestOK{All pass?}
    TestOK -->|No| FixTests[Fix tests]
    FixTests --> Tests
    TestOK -->|Yes| Docs[Check documentation]
    
    Docs --> DocOK{Complete?}
    DocOK -->|No| AddDocs[Add documentation]
    AddDocs --> Docs
    DocOK -->|Yes| PR[Create Pull Request]
    
    PR --> CI[CI/CD Pipeline]
    CI --> CIOK{CI passes?}
    CIOK -->|No| FixCI[Fix CI issues]
    FixCI --> PR
    CIOK -->|Yes| Review[Review by maintainers]
    
    Review --> ReviewOK{Approved?}
    ReviewOK -->|No| Address[Address comments]
    Address --> PR
    ReviewOK -->|Yes| Merge[Merge into develop]
    
    classDef defaultStyle fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,color:#333
    class Code,Lint,LintOK,FixLint,Tests,TestOK,FixTests,Docs,DocOK,AddDocs,PR,CI,CIOK,FixCI,Review,ReviewOK,Address,Merge defaultStyle
    
    style Code fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style Lint fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Tests fill:#a3d8b0,stroke:#a3d8b0,stroke-width:2px
    style Docs fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style PR fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px
    style CI fill:#ef9a9a,stroke:#e57373,stroke-width:2px
    style Review fill:#e1bee7,stroke:#e1bee7,stroke-width:2px
    style Merge fill:#a3d8b0,stroke:#a3d8b0,stroke-width:3px
```

---

**Questions?** Open an issue with the `code-standards` label or contact core maintainers.

**Last Updated:** 2026

