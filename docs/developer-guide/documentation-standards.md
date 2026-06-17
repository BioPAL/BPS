# Documentation Standards

Complete documentation standards for BioPAL, including docstring formats, writing conventions, documentation types, and best practices.

---

## Table of Contents

1. Overview
2. Documentation Types
3. Docstring Format (NumPy Style)
4. Writing Conventions
5. Documentation Structure
6. Building and Previewing Documentation
7. Documentation Updates Required
8. Best Practices

---

## Overview

If you're not the developer type, contributing to the documentation is still of huge value. You don't even have to be an expert on BioPAL to do so! In fact, there are sections of the docs that are worse off after being written by experts. If something in the docs doesn't make sense to you, updating the relevant section after you figure it out is a great way to ensure it will help the next person.

The BioPAL documentation consists of two main parts:

- **Docstrings in the code itself**: These are meant to provide a clear explanation of the usage of individual functions, classes, and methods. They follow the **NumPy Docstring Standard**, which is used widely in the Scientific Python community.
- **Standalone documentation files**: These consist of tutorial-like overviews per topic together with other information (getting started, installation, architecture, etc.). These are typically written in Markdown or reStructuredText format.

**Key principles:**

- **Docstrings** should focus on *how* to use a function or class
- **Standalone documentation** should provide *why* and *when* to use features, with examples and tutorials
- **Keep documentation up to date**: When you change code, update the corresponding documentation
- **Write for your past self**: Document things that weren't obvious to you when you first learned them

---

## Documentation Types

### 1. User Documentation

**Purpose:** How to use the processors

**Content:**
- Installation guides
- User tutorials
- Usage examples
- Configuration guides
- Troubleshooting

**Location:** `docs/user/` or similar

### 2. Developer Documentation

**Purpose:** How to contribute and extend

**Content:**
- Contributing guides
- Development workflows
- Architecture documentation
- Extension guides

**Location:** `docs/developer/` or similar

### 3. API Documentation

**Purpose:** Auto-generated from docstrings

**Content:**
- Function and class references
- Parameter descriptions
- Return value documentation
- Usage examples

**Location:** Auto-generated from code docstrings

### 4. Scientific Documentation

**Purpose:** Algorithm descriptions and validation

**Content:**
- Algorithm descriptions
- Scientific validation results
- Methodology documentation
- References to papers

**Location:** `docs/scientific/` or similar

---

## Docstring Format (NumPy Style)

All docstrings in BioPAL must follow the **NumPy Docstring Standard**. This format is widely used in the Scientific Python community and provides a clear, structured way to document functions and classes.

### Complete Docstring Template

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

### Required Sections

**Summary (first line):**
- One-line description of what the function/class does
- Should be a complete sentence ending with a period

**Parameters:**
- All parameters with types and descriptions
- Include constraints, ranges, and default values
- For complex types (dict, list), describe the structure

**Returns:**
- Return value with type and description
- Include shape, range, or other constraints

### Optional Sections

**Raises:**
- Exceptions that may be raised
- When and why they are raised

**See Also:**
- Related functions or classes
- Links to relevant documentation

**References:**
- Scientific papers or documentation
- Use numbered references format

**Examples:**
- Usage examples
- Should be executable (tested with doctest)
- Show common use cases

**Notes:**
- Additional information
- Implementation details
- Warnings or important considerations

---

## Writing Conventions

### For Markdown Files

**Headings:**
- Use clear, descriptive headings
- Follow a consistent hierarchy (#, ##, ###)
- Use title case for main headings

**Paragraphs:**
- Keep paragraphs short and focused
- One main idea per paragraph
- Use line breaks for readability

**Code Blocks:**
- Use code blocks with syntax highlighting
- Include language identifier: ` ```python `
- Make code examples executable when possible

**Lists:**
- Use lists for step-by-step instructions
- Use numbered lists for ordered steps
- Use bullet lists for unordered items

**Links:**
- Link to related documentation
- Use descriptive link text
- Check that links are valid

**Examples:**
- Include examples where helpful
- Show, don't just tell
- Test examples to ensure they work

### For reStructuredText Files (if used)

**Section Markup:**
- `=`, for main headings
- `-`, for sections
- `~`, for subsections

**Directives:**
- Use `:doc:` directive for internal document links
- Use `:ref:` for cross-references
- Use `:code:` for inline code

**Follow the [Sphinx reStructuredText documentation](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)**

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

## Documentation Structure

### Referring to Other Documents

**In Markdown:**
- Use standard markdown link syntax: `[text](/path/to/doc)`
- Use relative paths for internal links
- Use absolute paths for cross-references

**In reStructuredText:**
- Use `:doc:` directive for internal document links
- Use `:ref:` for cross-references with labels

### Including Figures and Files

**Images:**
- Use standard markdown image syntax: `![alt text](path/to/image.png)`
- Or reStructuredText image directives: `.. image:: path/to/image.png`
- Include alt text for accessibility
- Keep images optimized and appropriately sized

**Code Files:**
- Reference code files when relevant
- Use code blocks for small snippets
- Link to full files in repository for larger examples

---

## Building and Previewing Documentation

### Building Documentation Locally

You can build the documentation locally to preview your changes. The documentation build process includes HTML generation, local serving, and link checking.

**Basic commands:**
```bash
# Build HTML documentation
make docs

# Build and serve locally
make docs-serve

# Check for broken links
make docs-linkcheck
```

**Viewing built documentation locally:**

After building the documentation, you can view it in your browser. The path depends on your build system:

```bash
# For Sphinx, typically:
open docs/_build/html/index.html      # macOS
xdg-open docs/_build/html/index.html  # Linux
start docs/_build/html/index.html     # Windows
```

### View Documentation Preview in CI/CD

The **`baseline-docs`** job in the CI pipeline (`ci.yml`) automatically runs a Sphinx build if the file `docs/api/conf.py` exists in the module. This job is part of the baseline pipeline and runs on every PR: its result is visible in the PR checks list.

If you have made updates to the documentation, you can see the build result by:

1. Clicking on "Details" next to the `baseline-docs` check in the PR checks list
2. Reviewing the Sphinx build output for errors or warnings

This allows you to verify your documentation changes before they are merged.

---

## Documentation Updates Required

When contributing, update documentation as needed:

### Code Changes

- **User-facing changes**: Update user guides and tutorials
- **API changes**: Update API documentation and docstrings
- **Workflow changes**: Update developer guides

### Specific Updates

- **User guides**: If user-facing functionality changes
- **Developer guides**: If development workflow changes
- **API documentation**: If adding/modifying functions or classes
- **Notebooks**: If examples are affected
- **Product definitions**: If products change
- **Architecture docs**: If system architecture changes

### Checklist

Before submitting a pull request, verify:

- [ ] Docstrings updated for new/modified functions
- [ ] User documentation updated (if applicable)
- [ ] Developer documentation updated (if applicable)
- [ ] Examples updated and tested
- [ ] Links are valid
- [ ] Documentation builds without errors

---

## Best Practices

### General Principles

**Be consistent:**
- Follow existing documentation style
- Use the same terminology throughout
- Maintain consistent formatting

**Use examples:**
- Show, don't just tell
- Include working code examples
- Test examples to ensure they work

**Keep it simple:**
- Write for someone new to the project
- Avoid jargon when possible
- Explain acronyms on first use

**Update as you go:**
- Don't leave documentation for later
- Update docs when you update code
- Fix documentation bugs promptly

**Test your examples:**
- Make sure code examples actually work
- Run doctests to verify examples
- Update examples when code changes

**Link appropriately:**
- Help readers find related information
- Use descriptive link text
- Check that links are valid

### Writing Style

**Tone:**
- Professional but friendly
- Clear and direct
- Helpful and encouraging

**Language:**
- Use active voice when possible
- Write in present tense
- Use second person ("you") for user-facing docs

**Structure:**
- Start with the most important information
- Use headings to organize content
- Break up long sections

### Documentation Maintenance

**Regular reviews:**
- Review documentation periodically
- Update outdated information
- Remove deprecated content

**Community contributions:**
- Welcome documentation contributions
- Review documentation PRs carefully
- Provide feedback constructively

---

## Resources

### Documentation

- [Contributing Guide](https://biomass-disc.info/docs/contributing) - Contribution process and workflows
- [Code Standards](https://biomass-disc.info/docs/code-standards) - Coding conventions and best practices
- [Architecture](https://biomass-disc.info/docs/architecture) - System architecture and design

### External Resources

- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [Markdown Guide](https://www.markdownguide.org/)
- [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)

---

**Questions?** Open an issue with the `documentation` label or contact core maintainers.

**Last Updated:** 2025

