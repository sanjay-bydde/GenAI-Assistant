import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

DOCS_PATH = "data/documents"
INDEX_PATH = "faiss_index"

def ingest_documents():
    texts = []

    # Read all PDF files in the folder
    for file in os.listdir(DOCS_PATH):
        if file.lower() == "medical_book.pdf":  # specifically your PDF
            file_path = os.path.join(DOCS_PATH, file)
            reader = PdfReader(file_path)
            full_text = ""
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
            texts.append(full_text)
            print(f"Read {file} with {len(reader.pages)} pages.")

    # Split documents into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # adjust if needed
        chunk_overlap=200
    )

    documents = splitter.create_documents(texts)

    # Create embeddings using OpenAI API
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Build FAISS index
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(INDEX_PATH)

    print(f"Ingested {len(documents)} chunks into FAISS index.")

if __name__ == "__main__":
    ingest_documents()
