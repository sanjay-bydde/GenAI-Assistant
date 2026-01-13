import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

load_dotenv()

INDEX_PATH = "faiss_index"
TOP_K = 5

def get_retriever(k=TOP_K):
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    db = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db.as_retriever(search_kwargs={"k": k})


if __name__ == "__main__":
    retriever = get_retriever()

    query = "What is diabetes?"
    docs = retriever.get_relevant_documents(query)

    print(f"\nTop {TOP_K} chunks for query: {query}\n")
    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}:\n{doc.page_content}\n{'-'*80}")
