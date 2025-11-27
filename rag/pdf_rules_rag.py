import os
import base64
import ollama
from byaldi import RAGMultiModalModel

RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)

RAG.index(
    inputh_path = "./basic-rules-fr.pdf",
    index_name="attention",
    store_collection_with_index=True,
    overwrite=True
)

def inference (question: str):
    results = RAG.search(question, k=1)

    response = ollama.chat(
        model="llama3.2-vision",
        messages=[{
            'role': 'user',
            'content': "question",
        }]
    )

    return response ['message']['content']