from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from load_documents import load_documents_from_folder
from split_documents import split_documents
from embedding_faiss import create_vector_store, load_vector_store
import os

os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI-API-KEY"
os.environ["OPENAI_API_BASE"] = "YOUR-OPENAI-API-KEY-BASE"
DOCUMENTS_FOLDER = "documents"
VECTOR_STORE_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "What is a transformer in AI?"

def create_rag_pipeline(vectorstore):
    llm = ChatOpenAI(
        model_name="qwen2.5-coder:7b",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_base=os.environ["OPENAI_API_BASE"]
    )

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an AI assistant. Given the context below, answer the question.

        Context: {context}
        Question: {question}

        Answer:
        """
    )

    ragChain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )
    return ragChain

def main():
    print("\n<< Loading documents... >>")
    documents = load_documents_from_folder(DOCUMENTS_FOLDER)

    if not documents:
        print("NO documents loaded. Check your folder or loaders.")
        return

    print("<< Splitting documents into chunks...>>")
    chunks = split_documents(documents)

    if not os.path.exists(VECTOR_STORE_PATH):
        print("<< Creating FAISS vector store...>>")
        vectorstore = create_vector_store(chunks, model_name=EMBEDDING_MODEL, save_path=VECTOR_STORE_PATH)
    else:
        print("<< Loading existing vector store...>>")
        vectorstore = load_vector_store(VECTOR_STORE_PATH, model_name=EMBEDDING_MODEL)

    print("<< Creating RAG pipeline with OpenAI...>>")
    rag_pipeline = create_rag_pipeline(vectorstore)

    print(f"\n << Query: >> {QUERY}")
    result = rag_pipeline({"query": QUERY})

    print("\n << Answer: >>")
    print(result["result"])

    print("\n << Source Documents: >>")
    for doc in result["source_documents"]:
        print(f"- {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()