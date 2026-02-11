# Contributing to MetalNative

Thank you for your interest in contributing to MetalNative! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Building from Source](#building-from-source)
- [Running Tests](#running-tests)
- [Code Style Guide](#code-style-guide)
- [Making Changes](#making-changes)
- [Submitting Pull Requests](#submitting-pull-requests)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a new branch for your changes
5. Make your changes and test them
6. Submit a pull request

## Development Setup

### Prerequisites

- macOS 12.0 or later
- Xcode 14.0 or later with Command Line Tools
- CMake 3.20 or later
- Python 3.10 or later
- Metal-compatible GPU (Apple Silicon or Intel with discrete GPU)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/kimsejun/metal-native.git
cd metal-native

# Install Python dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

## Building from Source

### C++ Library

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DBUILD_BENCHMARKS=ON

# Build
cmake --build . -j$(sysctl -n hw.ncpu)

# Install (optional)
sudo cmake --install .
```

### Python Bindings

```bash
# Build in development mode
pip install -e .

# Or build wheel
pip install build
python -m build
```

## Running Tests

### C++ Tests

```bash
# From build directory
ctest --output-on-failure

# Run specific test
./tests/test_buffer

# Run with verbose output
ctest -V
```

### Python Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_buffer.py

# Run with coverage
pytest --cov=metal_native --cov-report=html tests/
```

### Benchmarks

```bash
# Run all benchmarks
./benchmarks/benchmark_ops

# Run specific benchmark
./benchmarks/benchmark_buffer --benchmark_filter=buffer_copy
```

## Code Style Guide

### C++ Style

- Follow the LLVM coding style (see `.clang-format`)
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use C++17 features where appropriate
- Prefer `snake_case` for functions and variables
- Use `PascalCase` for class names
- Use descriptive names; avoid abbreviations

**Format your code before committing:**

```bash
# Format all C++ files
find src include -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i

# Check formatting without modifying
clang-format --dry-run --Werror src/buffer.cpp
```

### Python Style

- Follow PEP 8 style guide
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints for function signatures
- Format with `black` and lint with `flake8`

**Format your code:**

```bash
# Format Python files
black python/

# Check with flake8
flake8 python/
```

### Metal Shading Language

- Use 4 spaces for indentation
- Follow naming conventions similar to C++
- Add comments explaining non-obvious operations
- Use descriptive kernel names

### Documentation

- Document all public APIs with clear docstrings
- Include usage examples for complex functions
- Update README.md if adding new features
- Add inline comments for complex logic

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-convolution-op` - New features
- `fix/buffer-memory-leak` - Bug fixes
- `docs/update-api-reference` - Documentation
- `perf/optimize-matmul` - Performance improvements
- `refactor/simplify-kernel-dispatch` - Code refactoring

### Commit Messages

Write clear, concise commit messages:

```
Short summary (50 chars or less)

More detailed explanation if necessary. Wrap at 72 characters.
Explain the problem this commit solves and why you chose this
approach.

- Bullet points are okay
- Use present tense ("Add feature" not "Added feature")
- Reference issues: Fixes #123
```

### Testing Requirements

- All new features must include tests
- Bug fixes should include regression tests
- Maintain or improve code coverage
- Tests must pass on all supported platforms

### Performance Considerations

- Profile performance-critical changes
- Include benchmark results for performance improvements
- Avoid premature optimization
- Document any performance trade-offs

## Submitting Pull Requests

### Before Submitting

1. **Update your branch** with the latest main:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all tests** and ensure they pass:
   ```bash
   # C++ tests
   cd build && ctest --output-on-failure

   # Python tests
   pytest tests/
   ```

3. **Format your code**:
   ```bash
   clang-format -i src/*.cpp include/*.hpp
   black python/
   ```

4. **Update documentation** if needed

5. **Commit your changes** with clear messages

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change necessary?

## Changes Made
- List key changes
- Include file modifications
- Note any breaking changes

## Testing
- [ ] Added unit tests
- [ ] Added integration tests
- [ ] Manual testing performed
- [ ] Benchmarks run (if applicable)

## Performance Impact
Describe any performance implications

## Breaking Changes
List any breaking API changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. Maintainers will review your PR
2. Address any feedback or requested changes
3. Once approved, your PR will be merged
4. Your contribution will be credited in the release notes

## Reporting Issues

### Bug Reports

Include:

- **Clear title** describing the issue
- **Steps to reproduce** the problem
- **Expected behavior** vs actual behavior
- **Environment details**: OS version, hardware, software versions
- **Code sample** demonstrating the issue (if possible)
- **Error messages** or stack traces

### Feature Requests

Include:

- **Use case**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: Other approaches you've thought about
- **Additional context**: Any relevant information

### Security Issues

**Do not** report security vulnerabilities in public issues. See [SECURITY.md](SECURITY.md) for responsible disclosure procedures.

## Development Resources

### Documentation

- [API Reference](docs/api_reference.md)
- [Architecture Overview](docs/architecture.md)
- [Metal Best Practices](docs/metal_best_practices.md)

### Communication

- GitHub Issues: Bug reports and feature requests
- GitHub Discussions: Questions and general discussion
- Pull Requests: Code review and contributions

## Recognition

Contributors will be recognized in:

- CHANGELOG.md release notes
- GitHub contributors page
- Project documentation (for significant contributions)

## License

By contributing to MetalNative, you agree that your contributions will be licensed under the same license as the project (see LICENSE file).

---

Thank you for contributing to MetalNative! 🚀
