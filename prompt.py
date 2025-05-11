from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

def create_rag_pipeline(vectorstore, openai_api_key):
    llm = ChatOpenAI(
    model="qwen2.5-coder:7b",  
    temperature=0,
    api_key=openai_api_key,
    base_url=os.getenv("OPENAI_API_BASE")
    )

    prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant. Given the context below, answer the question.
    
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
