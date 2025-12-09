# Contributing to Python Implementation

This guide covers contributing to the Python implementation of the vector search library.

## Development Setup

### 1. Create and Activate Virtual Environment

**On Windows:**
```bash
cd py
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
cd py
python -m venv venv
source venv/bin/activate
```

### 2. Install Development Dependencies

```bash
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 3. Set Up Pre-commit Hooks

Pre-commit hooks will automatically run the linter before each commit:

```bash
pre-commit install
```

## Code Quality

### Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for fast Python linting and formatting.

**Run the linter:**
```bash
ruff check . --config=pyproject.toml
```

**Auto-fix issues:**
```bash
ruff check . --fix --config=pyproject.toml
```

**Check formatting:**
```bash
ruff format . --check --config=pyproject.toml
```

**Format code:**
```bash
ruff format . --config=pyproject.toml
```

### Pre-commit Hooks

The project includes pre-commit hooks that will:
- Run Ruff linter with auto-fix
- Format code with Ruff
- Trim trailing whitespace
- Fix end of file issues
- Check for merge conflicts
- Validate YAML and TOML files
- Prevent committing large files

**Manually run all hooks:**
```bash
pre-commit run --all-files
```

**Skip hooks (not recommended):**
```bash
git commit --no-verify
```

## Testing

### Run Tests

**Using unittest:**
```bash
python -m unittest discover -s . -p "test_*.py" -v
```

**Using pytest with coverage:**
```bash
pytest test_*.py -v --cov=. --cov-report=term
pytest test_*.py -v --cov=. --cov-report=html  # Generate HTML report
```

**Run specific test class:**
```bash
python -m unittest test_hnsw.TestHNSWIndex -v
```

**Run specific test method:**
```bash
python -m unittest test_hnsw.TestHNSWIndex.test_knn_search_simple -v
```

## Code Style

- **Line length**: 120 characters
- **Function names**: PascalCase (e.g., `GetHeight`, `SearchLayer`)
- **Variable names**: snake_case (e.g., `l_c`, `idx_q`)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings

The linter configuration in `pyproject.toml` enforces these conventions.

## Project Structure

```
py/
├── pyproject.toml           # Python project configuration (Ruff, etc.)
├── .pre-commit-config.yaml  # Pre-commit hooks configuration
├── requirements-dev.txt     # Development dependencies
├── CONTRIBUTING.md          # This file
├── hnsw.py                  # HNSW implementation
├── bruteforce.py           # Brute force search implementation
├── test_*.py               # Unit tests
├── pq.py                   # Product quantization
└── venv/                   # Virtual environment (not committed)
```

## Development Workflow

1. **Activate virtual environment:**
   ```bash
   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes and test:**
   ```bash
   ruff check . --fix
   python -m unittest discover -s . -p "test_*.py" -v
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Description of changes"
   # Pre-commit hooks will run automatically
   ```

5. **Deactivate virtual environment when done:**
   ```bash
   deactivate
   ```

## Adding New Features

When adding new features:

1. Write tests first (TDD approach recommended)
2. Implement the feature
3. Ensure all tests pass
4. Run the linter and fix any issues
5. Update documentation if needed

## Common Issues

### Import Errors

If you get import errors, make sure:
- Your virtual environment is activated
- All dependencies are installed: `pip install -r requirements-dev.txt`

### Pre-commit Hook Failures

If pre-commit hooks fail:
- Review the error messages
- Fix the issues (often auto-fixed by hooks)
- Stage the changes again: `git add .`
- Commit again

### Virtual Environment Not Found

If the virtual environment is missing:
```bash
python -m venv venv
```

## Questions or Issues?

- Check existing issues on GitHub
- Create a new issue with a clear description
- Include Python version and OS information
