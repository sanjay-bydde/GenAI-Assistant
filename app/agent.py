import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from rag import get_retriever

load_dotenv()

def build_agent():
    # Load LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # cost-effective and good for RAG
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Prompt that enforces grounded answers
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a medical assistant.

Use ONLY the information provided in the context below to answer the question.
If the answer is not present in the context, say:
"I don't have enough information from the document to answer this."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    retriever = get_retriever(k=5)

    return chain, retriever


def ask_question(question: str):
    chain, retriever = build_agent()

    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    # Combine retrieved chunks into a single context
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generate answer
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response["text"]


if __name__ == "__main__":
    question = "What is diabetes?"
    answer = ask_question(question)
    print("\nAnswer:\n")
    print(answer)
