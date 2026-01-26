from langchain_openai import ChatOpenAI
from app.prompts import RAG_PROMPT
from app.rag import retrieve_context


def run_agent(question: str, retriever):
    """
    Orchestrates RAG + LLM to produce
    grounded answers with citations
    """
    context, sources = retrieve_context(question, retriever)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=200
    )

    chain = RAG_PROMPT | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    answer = response.content.strip()

    return answer, sources
