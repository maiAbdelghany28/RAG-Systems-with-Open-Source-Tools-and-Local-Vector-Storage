##################################################################################
##     Building RAG Systems with Open-Source Tools and Local Vector Storage     ##
##################################################################################


## Author
Mai Waheed
New Giza University – Applied Data Mining – Spring 2025


## Overview

This project uses a Retrieval-Augmented Generation (RAG) system for domain-specific question answering with LangChain and an OpenAI-compatible custom LLM endpoint. The system retrieves relevant information from a local corpus of documents using semantic similarity search, and then generates context-aware answers through a language model.
It is evaluated in terms of precision, recall, F1-score, and answer correctness on diverse test cases and retrieval approaches.
This solution has been developed for New Giza University Applied Data Mining course (Spring 2025).


## Learning Objectives

   • Build a complete RAG pipeline using open-source components

   • Implement document processing, chunking, and embedding generation

   • Configure and use local vector stores for efficient similarity search

   • Design effective retrieval strategies for different types of queries

   • Apply prompt engineering techniques to improve RAG performance

   • Evaluate and compare different RAG configurations

## Prerequisites

 • Python 3.10+

 • Basic understanding of vector embeddings and semantic search

 • Familiarity with LLMs and prompt engineering

 • Access to LLM API (OpenAI, Anthropic, etc.) for the generation component


## Setup Instructions

1. Clone or download the repository

   • Place all files in one directory:

      main.py, evaluation.py, bonus.py, prompt.py, embedding_faiss.py, split_documents.py, load_documents.py

   • Your documents inside a folder called documents/

2. Create and activate a virtual environment (optional but recommended)

3. Install dependencies

   pip install -r requirements.txt

   If you don’t have a requirements.txt, install manually:

   pip install langchain langchain-openai langchain-community openai faiss-cpu sentence-transformers

4. Run the system

   • Run the full pipeline:

   python main.py

   • Run evaluation with metrics:

   python evaluation.py

   • Test bonus features:

   python test_bonus_all.py


## RAG Architecture Overview

The Retrieval-Augmented Generation (RAG) pipeline includes:

   1. Document Loading

      Loads .pdf, .txt, and .docx files from the /documents directory.

   2. Chunk Splitting

      Documents are split into smaller chunks for better retrieval using recursive character splitters.

   3. Embedding + Vector Store

      Embeds chunks with HuggingFaceEmbeddings and stores them in a FAISS index.

   4. Retriever

      Default: similarity search (top-k). Bonus: hierarchical retrieval and filtering.

   5. LLM Integration

      Connects to a custom OpenAI-compatible endpoint:

      API Key: "YOUR-OPENAI-API-KEY"  

      Base URL: "YOUR-OPENAI-API-KEY-BASE"

## Evaluation Metrics

We evaluate each test case using:

   > Precision: Correct retrieved docs / total retrieved docs

   > Recall: Correct retrieved docs / total relevant docs

   > F1 Score: Harmonic mean of precision and recall

   Answer Correctness: 1 if answer contains expected keyword, else 0


## Strengths

   • Modular, extensible design

   • Bonus features like query rewriting and answer validation

   • Quantitative evaluation with multiple metrics

   • Works with custom LLM API endpoints

## Weaknesses

   • Filters may discard helpful short snippets

   • Exact match checking for answers is simplistic

   • Sensitive to document structure and metadata

   • LLM may still hallucinate despite retrieved context


## Challenges and Solutions

   1| "Model not found" errors

      Solution: Switched to a supported model using /v1/models listing

   2| Evaluation showed 0.0 precision/recall

      Solution: Fixed filename mismatch

   3| LangChain deprecation warnings

      Solution: Switched to langchain-openai

   4| Missing answers despite good retrieval

      Solution: Added self-correction using bonus.py

   5| Setup issues with custom API
   
      Solution: Verified model support and used full base_url and api_key


## References
   - Chapter 9: LLM Workflows from the textbook
   - “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”
   Lewis et al.
   - LangChainDocumentation: https://python.langchain.com/docs/get_started/introduction
   - Sentence Transformers Documentation: https://www.sbert.net/
   - FAISS Documentation: https://github.com/facebookresearch/faiss
   - “The Illustrated RAG: Retrieval-Augmented Generation” (blog post)
   - Chapter 9: LLM Workflows from the textbook
