import streamlit as st
import faiss
import os
from io import BytesIO
import numpy as np
from PyPDF2 import PdfReader

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from secret_api_keys import huggingface_api_key

os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingface_api_key

import torch
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

def process_input(input_type, input_data):
    pdf_reader = PdfReader(input_data)
    documents = ""
    for page in pdf_reader.pages:
        documents += page.extract_text()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(documents)

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'}
    )
    dimension = len(hf_embeddings.embed_query("sample text"))

    index = faiss.IndexFlatL2(dimension)
    vector_store = FAISS(embedding_function=hf_embeddings, index=index, docstore=InMemoryDocstore({}), index_to_docstore_id={})
    vector_store.add_texts(texts)

    return vector_store
