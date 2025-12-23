# Contributing to eCommerce Transactions Analysis

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/eCommerce_transactions.git
   cd eCommerce_transactions
   ```
3. **Set up development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e ".[dev,notebook]"
   ```

## Development Workflow

### Code Style

We use the following tools to maintain code quality:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Before committing, run:
```bash
make format  # Format code
make lint    # Check for issues
```

### Testing

All new features should include tests. Run tests with:
```bash
make test        # Run all tests
make test-cov    # Run with coverage report
```

### Commit Messages

Follow conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Adding/updating tests
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Example: `feat: Add customer lifetime value calculation`

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Ensure all tests pass
4. Update documentation if needed
5. Submit a pull request with a clear description

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on collaboration

## Questions?

Open an issue for any questions or concerns.
