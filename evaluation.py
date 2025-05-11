from sklearn.metrics import precision_score, recall_score, f1_score
from embedding_faiss import create_vector_store
from prompt import create_rag_pipeline
from load_documents import load_documents_from_folder
from split_documents import split_documents
from embedding_faiss import create_vector_store, load_vector_store

import os


def evaluate_retrieval(true_docs, retrieved_docs):
    true_docs_clean = [d.lower() for d in true_docs]
    retrieved_docs_clean = [d.lower() for d in retrieved_docs]

    y_true = []
    for doc in retrieved_docs_clean:
        match = any(true_doc in doc for true_doc in true_docs_clean)
        y_true.append(1 if match else 0)

    y_pred = [1] * len(retrieved_docs_clean)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def evaluate_answer(output, expected_answer):
    """Naive answer quality check: returns 1 if expected phrase appears, else 0."""
    return 1 if expected_answer.lower() in output.lower() else 0


def test_framework(rag_pipeline, test_cases):
    """
    Run multiple test queries and calculate average performance.
    test_cases = list of dicts with 'query', 'expected_answer', 'true_sources'
    """
    scores = []
    correct_answers = 0
    
    for case in test_cases:
        result = rag_pipeline({"query": case["query"]})
        answer = result["result"]
        retrieved_sources = [doc.metadata.get("source") for doc in result["source_documents"]]
        
        retrieval_metrics = evaluate_retrieval(case["true_sources"], retrieved_sources)
        answer_correct = evaluate_answer(answer, case["expected_answer"])

        scores.append({**retrieval_metrics, "answer_correct": answer_correct})
        
    return scores

def compare_rag_configs(document_path, embedding_models, prompt_templates, queries):
    """
    Compare RAG pipelines with different embeddings and prompts.
    """
    documents = load_documents_from_folder(document_path)
    chunks = split_documents(documents)

    results = []

    for model_name in embedding_models:
        vectorstore = create_vector_store(chunks, model_name=model_name, save_path=f"faiss_{model_name.split('/')[-1]}")

        for prompt in prompt_templates:
            rag = create_rag_pipeline(vectorstore, os.getenv("OPENAI_API_KEY"), custom_prompt=prompt)
            for q in queries:
                res = rag({"query": q})
                results.append({
                    "embedding_model": model_name,
                    "prompt": prompt.template[:30] + "...",
                    "query": q,
                    "response": res["result"]
                })
    return results

if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI-API-KEY"
    os.environ["OPENAI_API_BASE"] = "YOUR-OPENAI-API-KEY-BASE"

    test_cases = [
        {
            "query": "Machine Learning in Artificial Intelligence?",
            "expected_answer": "machine learning",
            "true_sources": ["0520.pdf"] # Example source
        },
        {
            "query": "What is BERT in NLP?",
            "expected_answer": "Bidirectional Encoder Representations from Transformers",
            "true_sources": ["1810.04805v2.pdf"]
        },
        {
            "query": "What is the RAG model?",
            "expected_answer": "Retrieval-Augmented Generation",
            "true_sources": ["2005.11401v4.pdf"]
        },
        {
            "query": "How is AI used in education?",
            "expected_answer": "personalization",
            "true_sources": ["ai-report.pdf"]
        },
        {
            "query": "What is an AI vision?",
            "expected_answer": "guiding document",
            "true_sources": ["aivision_eng-1.pdf"]
        }
        
    ]


    print(" Loading vector store...")
    vectorstore = load_vector_store()

    print(" Creating RAG pipeline...")
    rag = create_rag_pipeline(vectorstore, os.environ["OPENAI_API_KEY"])

    print(" Running test cases...")
    results = test_framework(rag, test_cases)

    print("\n Evaluation Results:")
    for res in results:
        print(res)

