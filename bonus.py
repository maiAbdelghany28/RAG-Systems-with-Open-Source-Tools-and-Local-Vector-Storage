from langchain.chat_models import ChatOpenAI
import os
from prompt import create_rag_pipeline
from load_documents import load_documents_from_folder
from split_documents import split_documents
from embedding_faiss import create_vector_store, load_vector_store


def rewrite_query(query, openai_api_key):
    """Use LLM to rewrite the query for better retrieval."""
    llm = ChatOpenAI(
    model="qwen2.5-coder:7b",  
    temperature=0,
    api_key=openai_api_key,
    base_url=os.getenv("OPENAI_API_BASE")
    )
    prompt = f"Rewrite the following query to improve retrieval quality:\n{query}"
    return llm.predict(prompt)


def filter_irrelevant_docs(docs, threshold=50):
    """Post-retrieval filter to remove documents under a length threshold."""
    return [doc for doc in docs if len(doc.page_content) > threshold]


def hierarchical_retrieve(query, vectorstore):
    """Coarse then fine retrieval simulation."""
    rough_docs = vectorstore.similarity_search(query, k=10)
    return rough_docs[:3] 


def self_correct_answer(query, result, openai_api_key):
    """Ask LLM to check and regenerate if wrong."""
    llm = ChatOpenAI(
    model="qwen2.5-coder:7b",  
    temperature=0,
    api_key=openai_api_key,
    base_url=os.getenv("OPENAI_API_BASE")
    )
    check = llm.predict(f"Is this answer correct?\nQuery: {query}\nAnswer: {result}\nRespond yes or no.")
    if "no" in check.lower():
        return llm.predict(f"Regenerate the correct answer for: {query}")
    return result

if __name__ == "__main__":
    
    os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI-API-KEY"
    os.environ["OPENAI_API_BASE"] = "YOUR-OPENAI-API-KEY-BASE"

    query = "What is a transformer in AI?"
    dummy_wrong_answer = "A transformer is a type of car."

    # Step 1: Rewrite query
    print("Rewriting Query...")
    rewritten = rewrite_query(query, os.environ["OPENAI_API_KEY"])
    print("Rewritten:", rewritten)

    # Step 2: Hierarchical Retrieval
    print("\n Loading Vector Store...")
    vectorstore = load_vector_store()

    print(" Running Hierarchical Retrieval...")
    retrieved = hierarchical_retrieve(rewritten, vectorstore)
    print(f" Retrieved {len(retrieved)} docs")
    for i, doc in enumerate(retrieved):
        print(f"Doc {i+1}: {doc.metadata.get('source')} - {doc.page_content[:100]}...")

    # Step 3: Filter short documents
    print("\n Filtering Short Docs (<50 characters)...")
    filtered = filter_irrelevant_docs(retrieved)
    print(f"Remaining after filter: {len(filtered)}")

    # Step 4: Self-correct answer
    print("\n Self-Correcting Answer...")
    corrected = self_correct_answer(query, dummy_wrong_answer, os.environ["OPENAI_API_KEY"])
    print("<< Final Answer: >>", corrected)
