# NanoChemGPT 
GPT-4o–powered, retrieval-augmented generation for nanomaterial synthesis.
---
Builtin database includes reference materials from text-mined datasets from the Ceder group (1,2).

RAG with PDFs allows for paper PDF uploads, that are autochunked and stored in a FAISS vector database, top-K chunks are injected to the prompt.

Output JSON converts the "human-readable" answer into atomic steps suitable for inputting into an external robotic apparatus.

References.

(1) Kononova, O.; Huo, H.; He, T.; Rong, Z.; Botari, T.; Sun, W.; Tshitoyan, V.; Ceder, G. Text-Mined Dataset of Inorganic Materials Synthesis Recipes. Sci Data 2019, 6 (1), 203. https://doi.org/10.1038/s41597-019-0224-1.
(2) Wang, Z.; Kononova, O.; Cruse, K.; He, T.; Huo, H.; Fei, Y.; Zeng, Y.; Sun, Y.; Cai, Z.; Sun, W.; Ceder, G. Dataset of Solution-Based Inorganic Materials Synthesis Procedures Extracted from the Scientific Literature. Sci Data 2022, 9 (1), 231. https://doi.org/10.1038/s41597-022-01317-2.
---
