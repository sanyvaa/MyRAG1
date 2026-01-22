import os
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from get_models import get_main_model_name
import streamlit as st

def main():
    query_text = input("Enter your question: ")
    ai_response = query_rag_and_ask_AI(query_text)
    print(ai_response)

@st.cache_data
def get_context_from_RAG_DB(user_question: str):
    
    CHROMA_PATH = "chroma"
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(user_question, k=20)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    return context_text


def query_rag_and_ask_AI(query_text: str):

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    context_text = get_context_from_RAG_DB(query_text)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = OllamaLLM(model = get_main_model_name())
    response_text = model.invoke(prompt)

    return response_text


if __name__ == "__main__":
    main()
