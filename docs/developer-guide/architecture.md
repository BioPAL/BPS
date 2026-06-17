# Architecture & Development Guide

Complete guide to code structure, module interfaces, development patterns, and development practices for the BIOMASS Processing Suite (BPS).

---

## Table of Contents

1. [Repository Architecture](#repository-architecture): layout, modules, root config, CI test harness
2. [Code Structure](#code-structure): data flow between processors
3. [Module Interfaces](#module-interfaces)
4. [Development Patterns](#development-patterns)
5. [Development Guide](#development-guide)
6. [Best Practices](#best-practices)
7. [Automation Tools](#automation-tools): nox, xsdata

---

## Repository Architecture

### High-Level Repository Layout

The BIOMASS Processing Suite (BPS) is an industrial monorepo developed by **Aresys** and **ACRI-ST** under ESA contract. It hosts all processors for the ESA BIOMASS mission. Each `bps-*` subdirectory is an independent Python package with its own `pyproject.toml`, tests, and changelog.

```
biomass-bps/
├── bps-common/                   # Shared types, utilities and interfaces
├── bps-task-tables/              # Processing task table definitions
├── bps-transcoder/               # BIOMASS data format transcoder
│
├── bps-l1_pre_processor/         # L1 pre-processing
├── bps-l1_core_processor/        # L1 core processing
├── bps-l1_framing_processor/     # L1 framing
├── bps-l1_binaries/              # L1 compiled binaries
├── bps-l1_processor/             # L1 main processor
│
├── bps-l2a_processor/            # L2A processor
├── bps-l2b_agb_processor/        # L2B Above-Ground Biomass (AGB)
├── bps-l2b_fd_processor/         # L2B Forest Disturbance (FD)
├── bps-l2b_fh_processor/         # L2B Forest Height (FH)
│
├── bps-stack_pre_processor/      # SAR stack pre-processing
├── bps-stack_coreg_processor/    # SAR stack co-registration
├── bps-stack_cal_processor/      # SAR stack calibration
├── bps-stack_binaries/           # Stack compiled binaries
├── bps-stack_processor/          # SAR stack main processor
│
├── bps-dockerfiles/              # Container definitions
│
├── tests/                        # Repository-level CI harness (see below)
│   ├── baseline/
│   ├── extended/
│   └── heavy/
│
├── VERSION                       # Global suite version (format MM.PP)
├── pyproject.toml                # ruff, mypy, black configuration
├── ruff.toml                     # Linting configuration
├── pytest.ini                    # Test markers definition
├── noxfile.py                    # Automation sessions (XSD, versioning)
├── .pre-commit-config.yaml       # Local validation hooks
├── .github/
│   ├── tier-policy.yml           # Automatic tier classification policy
│   ├── CODEOWNERS                # Review routing by module
│   └── workflows/
│       └── ci.yml                # Unified CI/CD pipeline
└── LICENSES/
    ├── Apache-2.0.txt
    └── MIT.txt
```

### Modules by Family

#### Common libraries

| Module | Role |
|--------|------|
| `bps-common/` | Shared types, utilities, and interfaces used by all processors |
| `bps-task-tables/` | Processing task table definitions |
| `bps-transcoder/` | BIOMASS data format transcoder |

#### L1 processors

| Module | Role |
|--------|------|
| `bps-l1_pre_processor/` | L1 pre-processing |
| `bps-l1_core_processor/` | L1 core processing |
| `bps-l1_framing_processor/` | L1 framing |
| `bps-l1_binaries/` | L1 binary tools |
| `bps-l1_processor/` | L1 main processor |

#### L2 processors

| Module | Role |
|--------|------|
| `bps-l2a_processor/` | L2A processor |
| `bps-l2b_agb_processor/` | L2B: Above-Ground Biomass (AGB) |
| `bps-l2b_fd_processor/` | L2B: Forest Disturbance (FD) |
| `bps-l2b_fh_processor/` | L2B: Forest Height (FH) |

#### SAR stack processors

| Module | Role |
|--------|------|
| `bps-stack_pre_processor/` | SAR stack pre-processing |
| `bps-stack_coreg_processor/` | SAR stack co-registration |
| `bps-stack_cal_processor/` | SAR stack calibration |
| `bps-stack_binaries/` | Stack binary tools |
| `bps-stack_processor/` | SAR stack main processor |

#### Infrastructure

| Module | Role |
|--------|------|
| `bps-dockerfiles/` | Docker files for containerising the processors |

### Root Configuration Files

These files apply to the entire monorepo:

| File | Purpose |
|------|---------|
| `VERSION` | Global BPS version (format `MM.PP`, e.g. `5.0`) |
| `pyproject.toml` | Monorepo-level tool configuration (black, ruff, mypy) |
| `ruff.toml` | Ruff linting configuration (`line-length = 120`) |
| `pytest.ini` | Pytest markers (`unit`, `baseline`, `extended`, `heavy`, `smoke`, `integration`, `public`) |
| `noxfile.py` | Automation sessions (see [Automation Tools](#automation-tools)) |
| `.pre-commit-config.yaml` | Pre-commit hooks for local validation |
| `.github/tier-policy.yml` | Automatic CI tier classification policy |
| `.github/CODEOWNERS` | Required reviewers per path (ESA gate on `VERSION` and `CHANGELOG.md`) |
| `.github/workflows/ci.yml` | Unified CI/CD pipeline |

### Structure of a Typical Module

Every `bps-*` module follows this layout:

```
bps-<module>/
├── src/                        # Python source code
│   └── bps_<module>/
├── tests/
│   ├── unit/                   # Unit tests (pytest -m unit)
│   ├── baseline/               # Baseline regression tests (pytest -m baseline)
│   ├── extended/               # Extended tests: TDS required (pytest -m extended)
│   └── heavy/                  # Heavy tests: TDS required (pytest -m heavy)
├── pyproject.toml              # Package configuration (build, ruff, mypy, black)
└── CHANGELOG.md                # Module-level changelog
```

Install a module for development:

```bash
cd bps-<module>
pip install -e ".[dev]"
```

Install pre-commit hooks once at the repository root (applies to the whole monorepo):

```bash
pre-commit install --hook-type commit-msg
pre-commit install
```

### Repository-Level CI Test Harness

In addition to per-module `tests/`, the monorepo root provides shared suites used by [`.github/workflows/ci.yml`](https://github.com/BioPAL/BPS/blob/main/.github/workflows/ci.yml):

| Directory | Pytest marker | When CI runs it |
|-----------|---------------|-----------------|
| `test/baseline/` | `baseline` | Every PR (baseline marker signal) |
| `test/extended/` | `extended` | Tier 1+ (Extended job) |
| `test/heavy/` | `heavy` | Tier 2 (Heavy job) |

Changes to these paths (and to `pytest.ini`) are **`locked_paths`** in [`.github/tier-policy.yml`](https://github.com/BioPAL/BPS/blob/main/.github/tier-policy.yml): they escalate CI to at least **Tier 1 (Extended)**. Per-module paths matching `**/test/baseline/**`, `test/extended/**`, or `test/heavy/**` are **`sme_owned_paths`** (also Tier 1+). **Tier 2 (Heavy)** applies when policy rules require it (e.g. `VERSION` change on a PR targeting `main`, `tier_2_paths`, or manual `run_heavy` on dispatch).

### Architecture Principles

**Single Repository:**
- One ESA external Git repository for all BIOMASS L2 code
- Hosts operational code, scientific developments, validation tools, and documentation
- Avoids divergence between operational and research versions
- Ensures validation workflows apply to production code

**Stability and Clarity:**
- Stable paths to key components
- Consistent structure for documentation and tools
- Easy navigation for new contributors
- Support for long-term evolution

**Separation of Concerns:**
- Core L2 algorithms separate from I/O
- Validation tools separate from processing code
- Configuration separate from implementation
- Documentation organized by audience

---

## Code Structure

### Component Interactions

```{mermaid}
flowchart LR
    L1[L1 Products]
    AUX[Auxiliary data / XSD schemas]

    COMMON[bps-common]
    TRANSCODER[bps-transcoder]
    L1PROC[bps-l1_processor]
    STACK[bps-stack_processor]
    L2A[bps-l2a_processor]
    L2B_AGB[bps-l2b_agb_processor]
    L2B_FH[bps-l2b_fh_processor]
    L2B_FD[bps-l2b_fd_processor]

    L1 --> TRANSCODER
    AUX --> COMMON
    COMMON --> L1PROC
    COMMON --> STACK
    TRANSCODER --> L1PROC
    L1PROC --> STACK
    STACK --> L2A
    L2A --> L2B_AGB
    L2A --> L2B_FH
    L2A --> L2B_FD
    L2B_AGB --> L2[L2 Products]
    L2B_FH --> L2
    L2B_FD --> L2

    classDef defaultStyle fill:#f5f5f5,stroke:#9e9e9e,stroke-width:2px,color:#333
    class L1,AUX,COMMON,TRANSCODER,L1PROC,STACK,L2A,L2B_AGB,L2B_FH,L2B_FD,L2 defaultStyle
```

**Data Flow:**
1. **Input**: L1 products ingested and transcoded to internal format
2. **Stack processing**: SAR stack co-registration and calibration
3. **L2A**: Intermediate L2 products (polarimetric decomposition)
4. **L2B**: Final L2 products: AGB, Forest Height, Forest Disturbance
5. **Output**: L2 products in ESA-defined formats

---

## Module Interfaces

### Interface Design Principles

**Explicit Parameters:**
- Functions take explicit inputs (no hidden dependencies)
- Configuration passed as parameters or config objects
- No global state modification

**Clear Return Values:**
- Structured return types (xarray Datasets, named tuples, or typed dictionaries)
- Consistent return formats across modules
- Error conditions handled via exceptions

**Type Hints:**
- All public functions have type hints
- Use `typing` module for complex types
- Enable static type checking with `mypy`

**Documentation:**
- NumPy-style docstrings for all public functions
- Parameter descriptions with types and constraints
- Return value descriptions
- Examples where helpful

### Example Interface

```python
from typing import Dict, Optional
import numpy as np
import xarray as xr

def calculate_biomass(
    sar_data: xr.Dataset,
    incidence_angle: xr.DataArray,
    config: Dict[str, float],
    aux_data: Optional[xr.Dataset] = None
) -> xr.Dataset:
    """
    Calculate above-ground biomass from SAR data.
    
    Parameters
    ----------
    sar_data : xr.Dataset
        Input SAR backscatter data with variables:
        - sigma0: backscatter coefficients
        - coherence: interferometric coherence (if available)
    incidence_angle : xr.DataArray
        Incidence angle in degrees, shape matching sar_data
    config : Dict[str, float]
        Configuration parameters:
        - threshold: detection threshold
        - max_iterations: maximum iterations
        - biomass_range: tuple (min, max) in Mg/ha
    aux_data : xr.Dataset, optional
        Auxiliary data (DEM, land cover, etc.)
        
    Returns
    -------
    xr.Dataset
        Biomass product with variables:
        - agb: above-ground biomass in Mg/ha
        - agb_uncertainty: uncertainty estimates
        - quality_flag: quality indicators
        
    Raises
    ------
    ValueError
        If sar_data contains invalid values
    RuntimeError
        If convergence not achieved
        
    Examples
    --------
    >>> sar_data = xr.Dataset({
    ...     'sigma0': (['y', 'x'], np.array([[0.1, 0.2], [0.3, 0.4]]))
    ... })
    >>> angle = xr.DataArray([[30.0, 31.0], [32.0, 33.0]], dims=['y', 'x'])
    >>> config = {'threshold': 0.5, 'max_iterations': 100}
    >>> result = calculate_biomass(sar_data, angle, config)
    """
    pass
```

### Module Dependencies

**Dependency Rules:**
- `bps-common` has no dependencies on other BPS modules
- `bps-transcoder` depends on `bps-common`
- L1/L2/stack processors depend on `bps-common` and may depend on `bps-transcoder`
- No circular dependencies allowed between modules

**External Dependencies:**
- Scientific: numpy, scipy, xarray, numba, netcdf4
- Data models: xsdata (generated from XSD schemas)
- Testing: pytest, pytest-cov
- Development: black, ruff, mypy, pre-commit, nox

---

## Development Patterns

### Design Patterns

**Pure Functions:**
- Prefer functions that take explicit inputs and return outputs
- Avoid modifying global state
- Make functions testable and predictable

```python
# Good: Pure function
def calculate_agb(sar_data: np.ndarray, config: dict) -> np.ndarray:
    """Calculate AGB from SAR data."""
    # Process data
    result = process(sar_data, config)
    return result

# Avoid: Function with side effects
def calculate_agb(sar_data: np.ndarray):
    """Calculate AGB (modifies global state)."""
    global global_config
    # Uses global config - hard to test
    pass
```

**Separation of Concerns:**
- Keep I/O separate from processing
- Keep configuration separate from algorithms
- Keep validation separate from core code

**Error Handling:**
- Use structured error handling
- Provide informative error messages
- Allow errors to propagate with context

```python
try:
    result = process_data(input_file)
except FileNotFoundError:
    logger.error(f"Input file not found: {input_file}")
    raise
except ValueError as e:
    logger.error(f"Invalid input data: {e}")
    raise
```

**Configuration Management:**
- Use configuration files (YAML/JSON) or dictionaries
- Support configuration inheritance
- Validate configuration at startup
- Document all configuration options

```python
# Load and validate configuration
config = load_config("config.yaml")
validate_config(config)
result = process_with_config(data, config)
```

### Data Structures

**Use xarray for Geophysical Data:**
- Labeled multi-dimensional arrays
- Coordinate system support
- Metadata preservation
- CF conventions compliance

```python
import xarray as xr

# Create dataset with coordinates
data = xr.Dataset(
    {
        'agb': (['latitude', 'longitude'], biomass_array)
    },
    coords={
        'latitude': lat_coords,
        'longitude': lon_coords,
        'time': time_coords
    },
    attrs={
        'title': 'Above Ground Biomass',
        'units': 'Mg/ha'
    }
)
```

**Use NumPy for Arrays:**
- Efficient numerical operations
- Well-tested and optimized
- Standard in scientific Python

**Use Dictionaries for Configuration:**
- Flexible and readable
- Easy to serialize (YAML/JSON)
- Support nested structures

---

## Development Guide

### Setting Up Development Environment

**Requirements:** Python 3.12

1. **Clone Repository:**
   ```bash
   git clone <repository-url>
   cd biomass-bps
   ```

2. **Create Virtual Environment:**
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install a Module (example: bps-l2b_fh_processor):**
   ```bash
   cd bps-l2b_fh_processor
   pip install -e ".[dev]"
   ```

4. **Set Up Pre-commit Hooks:**
   ```bash
   pip install pre-commit
   pre-commit install
   pre-commit install --hook-type commit-msg  # Required for DCO check
   ```

5. **Verify Setup:**
   ```bash
   pytest tests/unit/
   ```

### Development Workflow

**External contributors** must fork the repository. **Internal contributors** (with write access) can branch directly.

1. **Fork and Clone (external contributors):**
   ```bash
   # Fork via GitHub UI, then:
   git clone https://github.com/<your-username>/biomass-bps.git
   cd biomass-bps
   git remote add upstream <upstream-repository-url>
   ```

   **Internal contributors** skip the fork and clone directly:
   ```bash
   git clone <repository-url>
   cd biomass-bps
   ```

2. **Create Feature Branch:**
   ```bash
   git checkout develop
   git pull upstream develop   # or origin develop for internal contributors
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes:**
   - Write code following [Code Standards](https://biomass-disc.info/docs/code-standards)
   - Add tests for new functionality
   - Update documentation

4. **Run Local Checks:**
   ```bash
   # Auto-format code 
   black src/ tests/
   
   # Lint and auto-fix
   ruff check --fix src/ tests/
   
   # Type check
   mypy src/
   
   # Run tests
   pytest tests/unit/ -m unit
   pytest test/baseline/ -m baseline
   ```

5. **Commit Changes (DCO required):**
   ```bash
   git add .
   git commit -s -m "feat: add new feature"
   # The -s flag adds the required Signed-off-by trailer
   ```

6. **Push and Open PR toward `develop`:**
   ```bash
   git push origin feature/your-feature-name
   # Then open a PR from your fork/branch toward the upstream develop branch
   ```

### Adding New Modules

**Steps:**
1. Create module in the appropriate `bps-*/` directory
2. Define clear interfaces (functions with type hints)
3. Write comprehensive docstrings
4. Add unit tests under `tests/unit/`
5. Add baseline tests under `test/baseline/` (run on every PR)
6. Update documentation
7. Add to `__init__.py` exports

**Module Template:**
```python
"""
Module description.

This module provides [functionality description].
"""

from typing import Optional
import numpy as np
import xarray as xr

def public_function(
    input_data: xr.Dataset,
    parameter: float,
    optional_param: Optional[str] = None
) -> xr.Dataset:
    """
    Function description.
    
    Parameters
    ----------
    input_data : xr.Dataset
        Description of input
    parameter : float
        Description of parameter
    optional_param : str, optional
        Description of optional parameter
        
    Returns
    -------
    xr.Dataset
        Description of output
        
    Raises
    ------
    ValueError
        When invalid input provided
    """
    # Implementation
    pass
```

### Testing New Code

**Unit Tests** (`tests/unit/`, marker: `unit`):
- Test individual functions
- Use fixtures for test data
- Test edge cases and error conditions

**Baseline Tests** (`test/baseline/`, marker: `baseline`):
- Run on every PR as part of the CI baseline pipeline (`baseline-marker-signal` job)
- Quick sanity checks that the core output hasn't regressed
- If a difference is detected, the CI elevates the PR to Tier 1 automatically. To accept the change, update the reference outputs in the same PR; SME approval flows through CODEOWNERS-required reviews (no label needed).

**Extended Tests** (`test/extended/`, marker: `extended`):
- Run on Tier 1+ PRs (code changes)
- Smoke and integration-level coverage

**Heavy Tests** (`test/heavy/`, marker: `heavy`):
- Run on Tier 2 PRs (scientific changes, or any PR targeting the `release` branch)
- Full scientific regression against reference data
- PRs to `release` additionally require `run_heavy=true` to be set via `workflow_dispatch` for the `CI gate` to pass

> See [Repository-Level CI Test Harness](#repository-level-ci-test-harness) for how root and per-module test paths affect CI tier classification.

---

## Best Practices

### Code Organization

**Module Size:**
- Keep modules focused (single responsibility)
- Split large modules into smaller ones
- Aim for 200-500 lines per module

**Function Length:**
- Keep functions short and focused
- Extract complex logic into helper functions
- Aim for < 50 lines per function

**Naming:**
- Use descriptive names
- Follow Python naming conventions
- Be consistent across codebase

### Performance

**Optimization Guidelines:**
- Profile before optimizing
- Use vectorized operations (NumPy/xarray)
- Avoid premature optimization
- Document performance considerations

**Memory Management:**
- Use chunked I/O for large datasets
- Release resources explicitly when needed
- Monitor memory usage in tests

**Parallelization:**
- Use appropriate parallelization strategies
- Consider data locality
- Balance overhead vs. benefit

### Documentation

**Code Documentation:**
- Docstrings for all public functions
- Inline comments for complex logic
- Type hints for clarity
- Examples in docstrings

**Architecture Documentation:**
- Document design decisions
- Update architecture docs when structure changes
- Include diagrams where helpful

### Error Handling

**Error Types:**
- Use appropriate exception types
- Create custom exceptions for domain-specific errors
- Provide informative error messages

**Logging:**
- Use appropriate log levels
- Include context in log messages
- Log important state changes

---

## Resources

### Documentation

- [Getting Started](https://biomass-disc.info/docs) - Introduction and getting started guide
- [Licensing](https://biomass-disc.info/docs/licensing) - Apache 2.0 license requirements and legal obligations
- [Code of Conduct](https://biomass-disc.info/docs/code-of-conduct) - Community standards and expectations
- [Contributing Guide](https://biomass-disc.info/docs/contributing) - Contribution process and workflows
- [CI/CD Guide](https://biomass-disc.info/docs/ci-cd-guide) - Pipeline reference, tier detection, branch protection
- [Release Process](https://biomass-disc.info/docs/release-process) - How releases are prepared and published
- [Governance](https://biomass-disc.info/docs/governance) - Roles, responsibilities, and decision-making
- [Code Standards](https://biomass-disc.info/docs/code-standards) - Coding conventions and best practices
- [Documentation Standards](https://biomass-disc.info/docs/documentation-standards) - Documentation writing standards and best practices
- [Communication](https://biomass-disc.info/docs/communication) - Communication channels and meeting schedules

### External Resources

- [xarray Documentation](https://docs.xarray.dev/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Python Best Practices](https://docs.python-guide.org/)

---

**Questions?** Open an issue with the `architecture` label or contact core maintainers.

## Automation Tools

### nox

`nox` at the repository root provides automation sessions:

| Session | Command | Purpose |
|---------|---------|---------|
| `align_xsd` | `nox -s align_xsd` | Aligns XSD schemas across `bps-*` submodules |
| `generate_xsd_models` | `nox -s generate_xsd_models` | Generates Python models from BIOMASS XSD schemas via xsdata |
| `version_update` | `nox -s version_update` | Bumps the version across all `.py`, `.toml`, and `.yaml` files |

### xsdata

BIOMASS XSD schemas define the input/output data formats for the processors. Python models are generated automatically:

```bash
nox -s generate_xsd_models
```

Generated models are stored in `aux_pp2_*_models/` directories. These are excluded from mypy type checking and from version control (`.gitignore`). Run `nox -s generate_xsd_models` after any XSD schema update.

---

**Last Updated:** 2026

