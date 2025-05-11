from langchain.vectorstores.base import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def create_vector_store(documents, model_name="sentence-transformers/all-MiniLM-L6-v2", save_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore

def load_vector_store(save_path="faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)


def similarity_search(query, vectorstore, k=3):
    docs = vectorstore.similarity_search(query, k=k)
    for i, doc in enumerate(docs):
        print(f"Result {i+1}: {doc.metadata}")
        print(doc.page_content[:300])
        print("-" * 50)
    return docs

def mmr_retrieve(query, vectorstore, k=3, fetch_k=10):
    return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k)

