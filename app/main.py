from dotenv import load_dotenv
import os

load_dotenv()

from .rag import get_retriever
from .agent import run_agent


def main():
    print("Medical GenAI Assistant")
    print("Type 'exit' to quit\n")

    retriever = get_retriever()

    while True:
        question = input("Question: ").strip()
        if question.lower() == "exit":
            break

        answer, sources = run_agent(question, retriever)

        print("\nAnswer:\n")
        print(answer)

        if sources:
            print("\nSources:\n")
            for i, src in enumerate(sources, start=1):
                print(f"[{i}] {src}")

        print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
