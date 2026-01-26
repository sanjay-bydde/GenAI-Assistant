from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

INDEX_PATH = "faiss_index"

def is_valid_source(text: str) -> bool:
    """
    Filters out noisy or non-informative chunks
    """
    bad_patterns = [
        "GALE ENCYCLOPEDIA",
        "Page",
        "PERIODICALS",
        "Philadelphia",
        "19th ed.",
        "GEM -"
    ]
    return not any(p.lower() in text.lower() for p in bad_patterns)


def get_retriever(k: int = 4):
    """
    Loads FAISS index and returns a retriever
    """
    embeddings = OpenAIEmbeddings()

    db = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # Safe because YOU created it
    )

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever


def retrieve_context(query: str, retriever):
    """
    Retrieves relevant documents and prepares
    context + source list
    """
    docs = retriever.get_relevant_documents(query)

    # Filter noisy chunks
    docs = [doc for doc in docs if is_valid_source(doc.page_content)]

    # For definition-type questions, top-1 is enough
    if query.lower().startswith("what is"):
        docs = docs[:1]

    context = ""
    sources = []

    for i, doc in enumerate(docs, start=1):
        context += f"[{i}] {doc.page_content}\n\n"
        sources.append(doc.page_content[:300].strip())

    return context, sources
