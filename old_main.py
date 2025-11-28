# def main():
#     print("Hello from ir-project-rag!")


# if __name__ == "__main__":
#     main()

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector2 import retriever

model = OllamaLLM(model = "llama3")

template = """
You are an expert in bicycle safety questions
Here is the question to answer: {question}
"""


# Example usage of ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-----------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    answ_list = retriever.invoke(question)
    result = chain.invoke({"answ_list": answ_list, "question": question})
    print(result)