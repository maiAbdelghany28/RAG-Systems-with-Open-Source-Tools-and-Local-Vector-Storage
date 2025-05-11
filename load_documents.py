from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
import os

def load_documents_from_folder(folder_path):
    docs = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        try:
            if file.endswith(".pdf"):
                docs.extend(PyPDFLoader(file_path).load())
            elif file.endswith(".txt"):
                docs.extend(TextLoader(file_path).load())
            elif file.endswith(".docx"):
                docs.extend(Docx2txtLoader(file_path).load())
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return docs
