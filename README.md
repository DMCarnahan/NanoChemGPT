# NanoChemGPT 
GPT-4o–powered, retrieval-augmented generation for nanomaterial synthesis.
---
Builtin database includes reference materials from ChemAgent(1) and text-mined datasets from Ceder group papers (2,3).

RAG with PDFs allows for paper PDF uploads, that are autochunked and stored in a FAISS vector database, top-K chunks are injected to the prompt.

Output JSON converts the "human-readable" answer into atomic steps suitable for inputting into an external robotic apparatus.

References.

(1) Tang, X.; Hu, T.; Ye, M.; Shao, Y.; Yin, X.; Ouyang, S.; Zhou, W.; Lu, P.; Zhang, Z.; Zhao, Y.; Cohan, A.; Gerstein, M. ChemAgent: Self-Updating Library in Large Language Models Improves Chemical Reasoning. arXiv January 11, 2025. https://doi.org/10.48550/arXiv.2501.06590.
(2) Kononova, O.; Huo, H.; He, T.; Rong, Z.; Botari, T.; Sun, W.; Tshitoyan, V.; Ceder, G. Text-Mined Dataset of Inorganic Materials Synthesis Recipes. Sci Data 2019, 6 (1), 203. https://doi.org/10.1038/s41597-019-0224-1.
(3) Wang, Z.; Kononova, O.; Cruse, K.; He, T.; Huo, H.; Fei, Y.; Zeng, Y.; Sun, Y.; Cai, Z.; Sun, W.; Ceder, G. Dataset of Solution-Based Inorganic Materials Synthesis Procedures Extracted from the Scientific Literature. Sci Data 2022, 9 (1), 231. https://doi.org/10.1038/s41597-022-01317-2.
---
