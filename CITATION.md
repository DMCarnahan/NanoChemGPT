# Citation Information

## Software Citation

If NanoChemGPT is used in research, cite as follows:

```bibtex
@software{carnahan2024nanochemgpt,
  author = {Carnahan, D. Michael},
  title = {NanoChemGPT: Domain-Specific RAG for Nanochemistry Literature Mining and Synthesis Planning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/DMCarnahan/NanoChemGPT},
  version = {1.0.0},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

## Academic Publications

### Primary Publication

*Coming Soon*: A comprehensive paper describing NanoChemGPT's architecture, evaluation, and applications is in preparation for submission to a peer-reviewed journal.

### Conference Presentations

- *Planned*: Presentation at the American Chemical Society National Meeting
- *Planned*: Poster at the International Conference on Nanochemistry

## Methodology Citations

If specific components or methodologies from NanoChemGPT are used, also cite the relevant foundational work:

### Literature Mining and RAG
```bibtex
@article{lewis2020retrieval,
  title={Retrieval-augmented generation for knowledge-intensive nlp tasks},
  author={Lewis, Patrick and Perez, Ethan and Piktus, Aleksandra and Petroni, Fabio and Karpukhin, Vladimir and Goyal, Naman and K{\"u}ttler, Heinrich and Lewis, Mike and Yih, Wen-tau and Rockt{\"a}schel, Tim and others},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={9459--9474},
  year={2020}
}
```

### Named Entity Recognition for Chemistry
```bibtex
@article{krallinger2017overview,
  title={Overview of the BioCreative VI chemical-protein interaction track},
  author={Krallinger, Martin and Rabal, Obdulia and Akhondi, Saber A and others},
  journal={Proceedings of the sixth BioCreative challenge evaluation workshop},
  volume={1},
  pages={141--146},
  year={2017}
}
```

### Vector Similarity Search
```bibtex
@article{johnson2019billion,
  title={Billion-scale similarity search with GPUs},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  volume={7},
  number={3},
  pages={535--547},
  year={2019},
  publisher={IEEE}
}
```

## Dataset Citations

If the evaluation datasets provided with NanoChemGPT are used:

```bibtex
@dataset{carnahan2024nanochemgpt_eval,
  author = {Carnahan, D. Michael},
  title = {NanoChemGPT Evaluation Datasets},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/DMCarnahan/NanoChemGPT/tree/main/ai_eval/datasets},
  note = {Evaluation datasets for span extraction, entity recognition, and protocol structuring in nanochemistry}
}
```

## Model Citations

### spaCy NER Model
If the custom nanochemistry NER model is used:

```bibtex
@model{carnahan2024nanochemgpt_ner,
  author = {Carnahan, D. Michael},
  title = {Nanochemistry Named Entity Recognition Model},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/DMCarnahan/NanoChemGPT/tree/main/harvester/miner/ner_model},
  note = {spaCy-based NER model for nanochemistry entity extraction}
}
```

## Third-Party Dependencies

NanoChemGPT builds upon several open-source libraries. Key dependencies include:

### Core Framework
- **Flask**: Web framework for API endpoints
- **FastAPI**: ASGI wrapper for production deployment
- **spaCy**: Natural language processing and NER
- **FAISS**: Vector similarity search
- **Sentence Transformers**: Text embeddings

### Machine Learning
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **transformers**: Hugging Face transformer models

### Data Processing
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **lxml**: XML/HTML processing

These dependencies should also be cited appropriately in related publications.

## Usage Examples in Academic Writing

### In Methods Section
"Literature mining and synthesis protocol generation were performed using NanoChemGPT v1.0.0 (Carnahan, 2024), a domain-specific retrieval-augmented generation system for nanochemistry. The system employs FAISS-based vector search over automatically harvested literature from EU-PMC and ArXiv, with custom named entity recognition for material and process extraction."

### In Acknowledgments
"We thank D. Michael Carnahan for developing and maintaining NanoChemGPT, which was used for literature analysis and protocol generation in this work."

### In Figure Captions
"Figure X: Synthesis protocols generated using NanoChemGPT (Carnahan, 2024) based on literature analysis of gold nanoparticle synthesis methods."

## Contributing to Citations

When publishing work using NanoChemGPT:

1. **Notification (optional)**: An email with publication details may be sent
2. **Share results**: Consider contributing evaluation results back to the project
3. **Feedback (optional)**: Feedback can help improve the system for future research

## License and Usage Rights

NanoChemGPT is released under the MIT License, which permits:
- Commercial and non-commercial use
- Modification and distribution
- Private use

The only requirements are:
- Include the original license notice
- Include the copyright notice

## DOI and Persistent Identifiers

- **GitHub Repository**: `https://github.com/DMCarnahan/NanoChemGPT`
- **Zenodo Archive**: *Coming Soon* - Will provide persistent DOI for citation
- **ORCID**: 0000-0000-0000-0000 (D. Michael Carnahan)

## Contact for Citations

For questions about citations or academic use:
- **Email**: dcarnahan@example.com
- **GitHub Issues**: For technical questions
- **Twitter**: @NanoChemGPT (for updates and announcements)

---

*Last updated: January 2024*