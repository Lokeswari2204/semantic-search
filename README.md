## Dataset & Index Files
Due to GitHub size limits, large dataset files and FAISS indexes are not included.

To reproduce:
1. Download the dataset
2. Run `python backend/index.py` to generate embeddings and index
3. Start API using `uvicorn backend.app:app --reload`
