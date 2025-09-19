# Contributing to Iris Classification Tutorial..

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## 🚀 Quick Start

1. **Fork the repository**
2. **Clone your fork**
3. **Set up development environment**
4. **Make your changes**
5. **Submit a pull request**

## 🛠️ Development Setup

### Prerequisites
- Python 3.7+
- Git
- Virtual environment tool (venv, conda, etc.)

### Step-by-step Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/iris-classification-tutorial.git
   cd iris-classification-tutorial
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   make install-dev
   # or
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks** (recommended)
   ```bash
   pre-commit install
   ```

5. **Verify setup**
   ```bash
   make quick-test
   ```

## 📝 Code Style and Standards

We use several tools to maintain code quality:

### Code Formatting
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

```bash
# Format code
make format

# Check formatting
make format-check

# Run linting
make lint
```

### Type Hints
- Use type hints for function signatures
- MyPy is configured for type checking
- Example:
  ```python
  def train_model(X: np.ndarray, y: np.ndarray) -> sklearn.base.BaseEstimator:
      """Train a machine learning model."""
      # implementation
  ```

### Documentation
- Use docstrings for all functions, classes, and modules
- Follow Google/NumPy docstring style
- Example:
  ```python
  def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
      """
      Preprocess the iris dataset.
      
      Args:
          data: Raw iris dataset
          
      Returns:
          Tuple of (features, labels)
          
      Raises:
          ValueError: If data is empty or invalid
      """
  ```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
make test

# Run with verbose output
make test-verbose

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest --cov=src --cov-report=html
```

### Writing Tests
- Write tests for all new functions and classes
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)
- Example:
  ```python
  def test_data_loader_returns_correct_shape():
      # Arrange
      expected_shape = (150, 4)
      
      # Act
      X, y = load_iris_data()
      
      # Assert
      assert X.shape == expected_shape
      assert len(y) == 150
  ```

### Test Categories
- **Unit tests**: Test individual functions
- **Integration tests**: Test component interactions
- **Performance tests**: Test execution time and memory usage

## 📋 Commit Message Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Types
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that don't affect code meaning (formatting, etc.)
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to build process or auxiliary tools

### Examples
```bash
feat: add hyperparameter tuning for SVM model
fix: resolve data loading issue with custom datasets
docs: update README with installation instructions
test: add unit tests for preprocessing module
```

## 🔄 Pull Request Process

### Before Submitting
1. **Ensure all tests pass**
   ```bash
   make ci
   ```

2. **Update documentation** if needed

3. **Add tests** for new features

4. **Check code coverage** doesn't decrease significantly

### Pull Request Template
When creating a PR, please include:

- **Description**: What does this PR do?
- **Motivation**: Why is this change needed?
- **Testing**: How was this tested?
- **Checklist**: Use the provided template

### Example PR Description
```markdown
## Description
Add cross-validation support for all models

## Motivation
Users requested ability to evaluate models using cross-validation for more robust performance estimates.

## Changes
- Added `cross_validate_models()` function
- Updated model evaluation pipeline
- Added configuration options for CV folds
- Added tests for new functionality

## Testing
- All existing tests pass
- Added unit tests for cross-validation
- Tested with different CV strategies
- Performance impact: <5% increase in runtime

## Checklist
- [x] Tests added/updated
- [x] Documentation updated
- [x] Code follows style guidelines
- [x] Self-review completed
```

## 🐛 Bug Reports

### Before Reporting
1. **Search existing issues** to avoid duplicates
2. **Try the latest version** to see if it's already fixed
3. **Minimal reproduction** - create the smallest example that demonstrates the bug

### Bug Report Template
```markdown
## Bug Description
Clear and concise description of the bug.

## To Reproduce
Steps to reproduce the behavior:
1. Load data with '...'
2. Train model with '...'
3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Environment
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- Package versions: [paste pip freeze output]

## Additional Context
Any other context about the problem.
```

## 💡 Feature Requests

### Feature Request Template
```markdown
## Feature Description
Clear description of the feature you'd like to see.

## Motivation
Why would this feature be useful?

## Proposed Solution
How do you think this should be implemented?

## Alternatives Considered
What other solutions have you considered?

## Additional Context
Any other context or screenshots.
```

## 📚 Documentation Contributions

### Types of Documentation
- **Code documentation**: Docstrings and inline comments
- **User documentation**: README, tutorials, examples
- **API documentation**: Function and class references
- **Developer documentation**: Architecture, contributing guides

### Documentation Guidelines
- Use clear, concise language
- Include examples where helpful
- Update documentation with code changes
- Test documentation examples

## 🏷️ Release Process

### Versioning
We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist
1. Update version in `setup.py` and `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release notes
4. Tag the release
5. Update documentation

## 🤝 Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Celebrate contributions of all sizes

### Getting Help
- **GitHub Discussions**: For questions and general discussion
- **Issues**: For bug reports and feature requests
- **Pull Requests**: For code contributions
- **Email**: For sensitive matters

## 🎯 Areas for Contribution

### Beginner-Friendly
- Documentation improvements
- Adding tests
- Fixing typos
- Adding examples

### Intermediate
- New visualization features
- Performance optimizations
- Additional model algorithms
- CLI improvements

### Advanced
- Architecture improvements
- New data sources
- Advanced ML features
- Integration with other tools

## 📊 Contribution Recognition

We believe in recognizing all contributions:
- **Contributors list** in README
- **All-Contributors** specification
- **Release notes** mention significant contributions
- **GitHub badges** for different contribution types

## 🔍 Code Review Guidelines

### For Reviewers
- Be kind and constructive
- Explain the "why" behind suggestions
- Acknowledge good practices
- Focus on code, not the person

### For Contributors
- Respond to feedback promptly
- Ask questions if unclear
- Be open to suggestions
- Thank reviewers for their time

---

## 🙏 Thank You!

Thank you for contributing to the Iris Classification Tutorial! Every contribution, no matter how small, makes this project better for everyone.

**Happy coding!** 🎉
