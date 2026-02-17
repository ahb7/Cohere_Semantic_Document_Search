# Cohere_Semantic_Document_Search
Cohere Semantic Document Search
A simple semantic search system built using Cohere embeddings and FAISS for fast vector similarity search.

This project demonstrates how to build a lightweight document retrieval system using modern embedding models.

Features:
Convert documents into vector embeddings using Cohere
Store embeddings in a FAISS index
Perform fast similarity search
Retrieve the most relevant documents for a given query

🛠 Tech Stack
Python
Cohere API
FAISS
NumPy

How to run it:
pip install cohere streamlit faiss-cpu pymupdf numpy
export COHERE_API_KEY="your_key_here"
streamlit run app.py
