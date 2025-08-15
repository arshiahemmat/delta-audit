# Contributing to Delta-Audit

Thank you for your interest in contributing to Delta-Audit! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .
   pip install -r requirements.txt
   pip install ruff black pytest  # Development dependencies
   ```

## Development Setup

### Code Style

Delta-Audit uses:
- **Black** for code formatting
- **Ruff** for linting
- **Type hints** throughout the codebase

Run formatting and linting:
```bash
black src/
ruff check src/
ruff check --select I --fix src/
```

### Testing

Run tests:
```bash
pytest tests/
```

### Pre-commit Hooks

Install pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

## Making Changes

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the existing code style
- Add type hints to new functions
- Write docstrings for new functions
- Add tests for new functionality

### 3. Test Your Changes

```bash
# Run the quickstart to ensure basic functionality works
delta-audit quickstart

# Run tests
pytest tests/

# Check code style
black --check src/
ruff check src/
```

### 4. Commit Your Changes

Use conventional commit messages:
```
feat: add new metric for attribution stability
fix: resolve issue with DCE calculation
docs: update API documentation
test: add tests for new functionality
```

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

## Areas for Contribution

### High Priority

- **Tests**: Add unit tests and integration tests
- **Documentation**: Improve docstrings and documentation
- **Performance**: Optimize slow functions
- **Error handling**: Improve error messages and handling

### Medium Priority

- **New metrics**: Implement additional Δ-Attribution metrics
- **New algorithms**: Add support for more ML algorithms
- **New datasets**: Add support for more datasets
- **Visualization**: Improve plotting functions

### Low Priority

- **CLI improvements**: Add more command-line options
- **Configuration**: Add more configuration options
- **Examples**: Add more usage examples

## Code Guidelines

### Python Code

- Use Python 3.9+ features
- Add type hints to all functions
- Write docstrings for all public functions
- Use meaningful variable names
- Keep functions small and focused

### Documentation

- Update docstrings when changing functions
- Update README.md for new features
- Update documentation in `docs/` for new functionality
- Add examples for new features

### Testing

- Write tests for new functionality
- Ensure tests pass before submitting PR
- Add integration tests for CLI commands
- Test with different Python versions

## Pull Request Process

1. **Fork and clone** the repository
2. **Create a feature branch** from `main`
3. **Make your changes** following the guidelines above
4. **Test your changes** thoroughly
5. **Update documentation** as needed
6. **Submit a pull request** with a clear description

### Pull Request Checklist

- [ ] Code follows the style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] No new warnings are generated
- [ ] Changes are tested with the CLI
- [ ] Type hints are added for new functions

## Reporting Issues

When reporting issues, please include:

- **Description**: Clear description of the problem
- **Steps to reproduce**: Detailed steps to reproduce the issue
- **Expected behavior**: What you expected to happen
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, Delta-Audit version
- **Error messages**: Full error traceback if applicable

## Getting Help

- Check the [FAQ](docs/faq.md) for common issues
- Review existing issues and pull requests
- Ask questions in GitHub discussions
- Join the project discussions

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

By contributing to Delta-Audit, you agree that your contributions will be licensed under the MIT License. 