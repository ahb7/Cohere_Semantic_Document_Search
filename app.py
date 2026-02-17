import streamlit as st
import cohere
import os
import fitz  # PyMuPDF
import numpy as np
import faiss

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

st.title("Cohere Semantic Document Search")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    # Split into chunks
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    st.write("Generating embeddings...")

    response = co.embed(
        texts=chunks,
        model="embed-english-v3.0",
        input_type="search_document"
    )

    embeddings = np.array(response.embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    query = st.text_input("Ask a question about the document")

    if query:
        query_embed = co.embed(
            texts=[query],
            model="embed-english-v3.0",
            input_type="search_query"
        )

        query_vector = np.array(query_embed.embeddings).astype("float32")
        D, I = index.search(query_vector, k=3)

        context = "\n".join([chunks[i] for i in I[0]])

        answer = co.chat(
            model="command-a-03-2025",
            message=f"Answer the question using the context below:\n\n{context}\n\nQuestion: {query}",
            max_tokens=200
        )

        st.write("### Answer")
        st.write(answer.text)


