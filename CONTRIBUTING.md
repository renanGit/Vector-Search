# Contributing to Vector Search

Thank you for your interest in contributing! This is a multi-language project with implementations in various programming languages.

Im using python here to behave as a learning material and with plans to port over to C++/rust for optimized/fast implementation.

## Language-Specific Guides

Each language implementation has its own contribution guide with specific setup instructions:

- **[Python](py/CONTRIBUTING.md)** - HNSW, PQ, and brute force implementations

## Project Structure

```
vector-search/
├── py/                      # Python implementation
│   ├── CONTRIBUTING.md     # Python-specific guide
│   ├── pyproject.toml      # Python configuration
│   └── ...
├── rust/
├── cpp/
├── .github/
│   └── workflows/          # CI/CD pipelines
└── CONTRIBUTING.md         # This file
```

## General Guidelines

### Code Quality

- Write clean, readable code (abstract where needed)
- Add tests for new features
- Keep functions focused and modular
- Document complex algorithms
- Add references to external paper/algo

### Testing

- All new features must include tests
- Ensure existing tests pass before submitting
- Aim for high test coverage

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes following the language-specific guide
4. Run tests and linters
5. Commit with clear messages
6. Push to your fork and submit a PR

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
