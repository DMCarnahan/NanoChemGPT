# NanoChemGPT 
GPT-4o–powered, retrieval-augmented generation for nanomaterial synthesis.
---
RAG with PDFs allows for paper PDF uploads, that are autochunked and stored in a FAISS vector database, top-K chunks are injected to the prompt. These augment the builtin database, which is populated from published datasets condensed into a parquet file (see /DuckDB/included datasets.txt for details and references). Model has a specialized dataset search module for maximum accuracy.

Output JSON converts the "human-readable" answer into atomic steps suitable for inputting into an external robotic apparatus. ONLY FOR FACILE SYNTHESIS METHODS.
---
