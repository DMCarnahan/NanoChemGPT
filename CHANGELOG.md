# Changelog

All notable changes to NanoChemGPT will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

> Release candidate: branch `pr/ci-format-checks-2025-10-03` merged into `main`. All tests passed locally and CI format/test jobs completed successfully. Ready to tag and publish.


### Added
- Comprehensive test suite with pytest framework
- Publication-ready documentation structure
- Example Jupyter notebooks with complete workflows
- Type hints and docstrings for major functions
- Contributing guidelines and code of conduct
- Citation information and academic usage guidelines

### Changed
- Enhanced README with publication-quality content
- Improved API documentation with detailed examples
- Restructured project organization for better maintainability

### Fixed
- Unicode quote handling in prompting logic
- Orphaned string literals causing syntax errors
- Verbatim mode implementation and early return logic

## [1.0.0] - 2024-01-01

### Added
- Initial release of NanoChemGPT
- Core RAG system for nanochemistry literature mining
- Flask/FastAPI dual application architecture
- Automated literature harvesting from EU-PMC and ArXiv
- Vector-based retrieval with FAISS indexing
- Custom spaCy NER model for nanochemistry entities
- Protocol conversion to robot-executable operations
- Enhanced citation management with relevance filtering
- Evaluation framework with multiple metrics
- Multi-scale synthesis protocol generation
- Verbatim mode for exact text reproduction
- File upload and analysis capabilities
- Background literature mining job queue
- MongoDB integration for persistent storage
- DuckDB support for structured data queries
- Mechanistic reasoning knowledge base

### Core Features
- **Question Answering**: Context-aware responses with automatic citation
- **Literature Mining**: Automated harvesting with enhanced relevance scoring
- **Protocol Generation**: Scale-aware synthesis planning
- **Entity Recognition**: Custom NER for materials, conditions, and equipment
- **Citation Management**: Automatic reference extraction and formatting
- **File Processing**: PDF, text, and document analysis
- **API Endpoints**: RESTful API with comprehensive documentation
- **Evaluation Tools**: Metrics for span extraction, entity recognition, and structuring

### Technical Implementation
- **Embeddings**: Support for OpenAI and sentence-transformers
- **Vector Search**: Multi-index FAISS with passage and document level retrieval
- **Web Framework**: Flask with FastAPI wrapper for production
- **Database**: MongoDB for persistence, Redis for caching
- **Processing**: Custom spaCy pipeline with domain-specific models
- **Configuration**: Environment-based configuration with fallbacks

## [0.9.0] - 2023-12-01

### Added
- Beta release with core functionality
- Basic literature mining pipeline
- Simple question answering interface
- Initial vector search implementation

### Known Issues
- Limited evaluation framework
- Basic citation management
- No file upload capabilities
- Minimal error handling

## [0.8.0] - 2023-11-01

### Added
- Alpha release for internal testing
- Proof of concept implementation
- Basic Flask application structure
- Initial literature harvesting scripts

---

## Version History Summary

| Version | Release Date | Key Features |
|---------|-------------|--------------|
| 1.0.0   | 2024-01-01  | Full production release with comprehensive features |
| 0.9.0   | 2023-12-01  | Beta release with core functionality |
| 0.8.0   | 2023-11-01  | Alpha release for internal testing |

## Migration Guides

### Upgrading to 1.0.0 from 0.9.x

1. **Environment Variables**: Update your `.env` file with new configuration options:
   ```bash
   # New in 1.0.0
   ENABLE_ENHANCED_CITATIONS=true
   CITATION_MIN_SCORE=0.25
   VERBATIM_MODE_ENABLED=true
   ```

2. **API Changes**: The `/ask` endpoint now supports file uploads:
   ```python
   # Old way (still supported)
   response = requests.post("/ask", json={"question": "..."})
   
   # New way with file upload
   files = {'file': open('protocol.pdf', 'rb')}
   data = {'question': '...'}
   response = requests.post("/ask", data=data, files=files)
   ```

3. **Dependencies**: Update your requirements:
   ```bash
   pip install -r requirements.txt
   ```

4. **Database Schema**: No database migrations required for MongoDB users.

### Breaking Changes

#### 1.0.0
- None (fully backward compatible with 0.9.x)

#### 0.9.0
- Changed API response format for enhanced citation information
- Updated vector index structure (requires reindexing)
- Modified configuration file format

## Future Roadmap

### Planned for 1.1.0
- [ ] Multi-language literature support
- [ ] Advanced visualization for synthesis pathways
- [ ] Real-time collaboration features
- [ ] Enhanced performance optimizations
- [ ] Extended chemistry domain support

### Planned for 1.2.0
- [ ] Machine learning for protocol optimization
- [ ] Integration with laboratory information systems
- [ ] Advanced statistical analysis tools
- [ ] Custom model training interfaces

### Long-term Vision
- [ ] Multi-domain scientific literature mining
- [ ] Automated experimental design
- [ ] Integration with robotics platforms
- [ ] Community-driven knowledge base
- [ ] Educational tools and interfaces

## Contributing to the Changelog

When contributing changes:

1. **Add entries**: Update the [Unreleased] section with your changes
2. **Categorize properly**: Use Added, Changed, Deprecated, Removed, Fixed, or Security
3. **Be descriptive**: Explain what changed and why it matters to users
4. **Include breaking changes**: Clearly mark any breaking changes
5. **Reference issues**: Link to relevant GitHub issues or pull requests

### Changelog Entry Format

```markdown
### Added
- New feature description with brief explanation of benefits [#123]

### Changed
- Modified behavior description and impact on users [#456]

### Fixed
- Bug fix description and affected functionality [#789]
```

---

*For more details on any release, see the corresponding GitHub release notes and documentation.*