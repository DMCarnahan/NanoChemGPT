# Contributing to NanoChemGPT

We welcome contributions to NanoChemGPT! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Contributing Guidelines](#contributing-guidelines)
4. [Code Standards](#code-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Submitting Changes](#submitting-changes)
8. [Review Process](#review-process)

## Getting Started

### Types of Contributions

We welcome several types of contributions:

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new functionality
- **Code Contributions**: Implement new features or fix bugs
- **Documentation**: Improve or add documentation
- **Examples**: Create tutorials or example notebooks
- **Testing**: Add or improve test coverage

### Prerequisites

- Python 3.11 or higher
- Git
- Basic understanding of nanochemistry and/or machine learning
- Familiarity with Flask, FastAPI, or similar web frameworks

## Development Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/your-username/NanoChemGPT.git
   cd NanoChemGPT
   ```

2. **Create Development Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   ```

3. **Configure Environment**:
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Run Tests**:
   ```bash
   pytest tests/
   ```

5. **Start Development Server**:
   ```bash
   python app.py
   ```

## Contributing Guidelines

### Before Starting

1. **Check Existing Issues**: Look at the [issue tracker](https://github.com/DMCarnahan/NanoChemGPT/issues) to see if your idea is already being discussed.

2. **Create an Issue**: For significant changes, create an issue to discuss your approach before starting work.

3. **Fork the Repository**: Create your own fork to work on.

### Branch Strategy

- **main**: Production-ready code
- **develop**: Integration branch for new features
- **feature/feature-name**: Individual feature branches
- **bugfix/bug-description**: Bug fix branches
- **hotfix/critical-fix**: Critical production fixes

### Commit Messages

Use clear, descriptive commit messages:

```
type(scope): description

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- test: Adding or updating tests
- refactor: Code refactoring
- style: Code style changes
- perf: Performance improvements
- ci: CI/CD changes

Examples:
feat(harvester): add enhanced relevance filtering
fix(api): resolve citation numbering issue
docs(readme): update installation instructions
test(converter): add protocol conversion tests
```

## Code Standards

### Python Code Style

We follow [PEP 8](https://pep8.org/) with some specific requirements:

- **Line Length**: 88 characters (Black formatter)
- **Import Order**: Use `isort` for consistent import ordering
- **Type Hints**: All functions must include type annotations
- **Docstrings**: Google-style docstrings for all public functions

### Code Formatting

Use these tools to maintain code quality:

```bash
# Install development tools
pip install black isort flake8 mypy

# Format code
black .
isort .

# Check style
flake8 .
mypy app.py
```

### Example Function

```python
from typing import List, Optional, Dict, Any

def process_synthesis_protocol(
    text: str, 
    temperature_range: Optional[tuple[float, float]] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Process a synthesis protocol and extract structured information.
    
    Args:
        text: Raw protocol text to process
        temperature_range: Optional temperature range filter (min, max) in Celsius
        validate: Whether to validate extracted operations
        
    Returns:
        Dictionary containing:
            - operations: List of extracted operations
            - entities: List of identified entities
            - validation: Validation results if validate=True
            
    Raises:
        ValueError: If text is empty or invalid
        ProcessingError: If protocol processing fails
        
    Example:
        >>> result = process_synthesis_protocol(
        ...     "Heat to 100°C for 2 hours",
        ...     temperature_range=(50, 200),
        ...     validate=True
        ... )
        >>> result['operations'][0]['action']
        'heat'
    """
    if not text or not text.strip():
        raise ValueError("Protocol text cannot be empty")
    
    # Implementation here
    return {"operations": [], "entities": [], "validation": {}}
```

### Directory Structure

Follow the established directory structure:

```
NanoChemGPT/
├── app.py                    # Main application
├── requirements.txt          # Dependencies
├── docs/                    # Documentation
├── tests/                   # Test suite
├── examples/               # Example notebooks
├── harvester/              # Literature mining
├── retriever/              # Vector search
├── ai_eval/               # Evaluation framework
├── static/                # Frontend assets
├── templates/             # HTML templates
└── scripts/              # Utility scripts
```

## Testing

### Test Requirements

- **Coverage**: Maintain >80% code coverage
- **Types**: Include unit, integration, and performance tests
- **Isolation**: Tests should not depend on external services (use mocks)
- **Documentation**: Document complex test scenarios

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_nanochemgpt.py

# Run with coverage
pytest --cov=app --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run tests in parallel
pytest -n auto
```

### Writing Tests

```python
import pytest
from unittest.mock import Mock, patch

class TestSynthesisProtocol:
    """Test synthesis protocol processing."""
    
    def test_basic_protocol_parsing(self):
        """Test basic protocol text parsing."""
        protocol = "Heat solution to 100°C for 2 hours"
        result = process_synthesis_protocol(protocol)
        
        assert len(result['operations']) > 0
        assert result['operations'][0]['action'] == 'heat'
        assert result['operations'][0]['temperature'] == 100
        
    @pytest.mark.slow
    def test_complex_protocol_integration(self):
        """Test complex protocol with multiple steps."""
        # Complex integration test here
        pass
        
    @patch('app.openai_client')
    def test_with_mocked_api(self, mock_client):
        """Test with mocked external API calls."""
        mock_client.embeddings.create.return_value = Mock()
        # Test implementation
```

### Test Data

- Store test data in `tests/test_data/`
- Use small, focused examples
- Don't commit large files or API keys
- Use fixtures for reusable test data

## Documentation

### Documentation Standards

- **Docstrings**: All public functions need comprehensive docstrings
- **Type Hints**: Include type information for all parameters and returns
- **Examples**: Provide usage examples in docstrings
- **API Docs**: Update API documentation for new endpoints

### Building Documentation

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme

# Build documentation
cd docs
make html
```

### Documentation Types

1. **API Documentation**: Document all endpoints, parameters, and responses
2. **Code Documentation**: Inline comments and docstrings
3. **User Guides**: Installation, configuration, and usage guides
4. **Developer Guides**: Architecture, contributing, and development setup
5. **Examples**: Jupyter notebooks and code examples

## Submitting Changes

### Pull Request Process

1. **Create Feature Branch**:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make Changes**: Implement your feature or fix

3. **Test Thoroughly**:
   ```bash
   pytest
   black .
   flake8 .
   ```

4. **Update Documentation**: Update relevant documentation

5. **Commit Changes**:
   ```bash
   git add .
   git commit -m "feat(scope): add new feature description"
   ```

6. **Push to Fork**:
   ```bash
   git push origin feature/my-new-feature
   ```

7. **Create Pull Request**: Use GitHub's interface to create a PR

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (describe)

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] No breaking changes (or documented)
```

## Review Process

### Review Criteria

- **Functionality**: Does the code work as intended?
- **Code Quality**: Is the code readable, maintainable, and well-structured?
- **Testing**: Are there adequate tests with good coverage?
- **Documentation**: Is the code and functionality properly documented?
- **Performance**: Does the change impact system performance?
- **Security**: Are there any security implications?

### Reviewer Guidelines

- Be constructive and specific in feedback
- Focus on the code, not the person
- Suggest improvements rather than just pointing out problems
- Test the changes locally when possible
- Consider the impact on existing functionality

### Response to Feedback

- Address all reviewer comments
- Ask for clarification if feedback is unclear
- Be open to suggestions and alternative approaches
- Update tests and documentation as needed

## Development Guidelines

### Adding New Features

1. **Design Document**: For major features, create a design document
2. **API Design**: Consider the API interface carefully
3. **Backward Compatibility**: Maintain compatibility when possible
4. **Performance**: Consider the impact on system performance
5. **Testing**: Include comprehensive tests
6. **Documentation**: Document new functionality thoroughly

### Bug Fixes

1. **Reproduce the Bug**: Create a test that reproduces the issue
2. **Root Cause Analysis**: Understand why the bug occurred
3. **Minimal Fix**: Make the smallest change that fixes the issue
4. **Regression Tests**: Add tests to prevent the bug from reoccurring
5. **Documentation**: Update documentation if needed

### Performance Improvements

1. **Benchmark**: Measure performance before and after changes
2. **Profiling**: Use profiling tools to identify bottlenecks
3. **Testing**: Ensure performance improvements don't break functionality
4. **Documentation**: Document performance characteristics

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Focus on constructive collaboration
- Help others learn and grow
- Follow the [Contributor Covenant](https://www.contributor-covenant.org/)

### Getting Help

- **GitHub Issues**: For bugs, feature requests, and questions
- **Discussions**: For general questions and community discussion
- **Email**: For private or sensitive matters

### Recognition

Contributors will be recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- Academic publications when appropriate

---

Thank you for contributing to NanoChemGPT! Your contributions help advance nanochemistry research and make scientific tools more accessible.