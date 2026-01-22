import os
import time
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#from QueryRAG_Data import get_context_from_RAG_DB
from get_models import get_main_model_name
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

CHROMA_PATH = "chroma"
embedding_function = get_embedding_function()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

def get_first_db_dir_name():
    doc = db.get()["metadatas"][0]
    doc_name = doc["source"]
    path_name = os.path.basename(os.path.dirname(doc_name))
    chunks_num = len(db.get()["ids"])
    return path_name, chunks_num

def get_context_from_RAG_DB(user_question: str, depth):
    results = db.similarity_search_with_score(user_question, k=depth)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return context_text, sources

def get_AI_response(user_question, context, modelname):
    PROMPT_TEMPLATE = """
        User Question: {user_question}

        Answer the user question based only on the following context:
     
        --
        {context}
        ---

        Answer the question based on the above context: {user_question}
        """
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    llm = init_chat_model(modelname, model_provider=os.getenv("MODEL_PROVIDER"), temperature=0)

    chain = prompt_template | llm | StrOutputParser()
    return chain.stream({
        "user_question" : user_question,
        "context" : context
        })

st.set_page_config(layout="wide", page_title="TestSuite AI Chat", page_icon="./MtsIcon.ico")
st.title("TestSuite AI Chat")
path_name, chunks_num = get_first_db_dir_name()
st.markdown(":yellow[TestSuite documentation bundle name: " + str(path_name) +
            ". Number of chunks in database: "  + str(chunks_num) + ".]")

with st.container():
    col1, col2, col3, col4 = st.columns([1, 2, 3, 4])

    with col1:
        model_names = [ 
            "llama3.2", 
            "granite3.2-vision",
            "deepseek-r1:1.5b",
            "qwen3-vl:4b",  ]
        model_name = st.selectbox( "Select LLM:",  model_names, width=400)

    with col2:
       depth = st.slider( "Search depth",  min_value=1,  max_value=100,  value=10,  step=1, width=200) 

    with col3:
       show_refs = st.checkbox("Show references in response", width=200)
       
    with col4:
        clearButton = st.button("Clear the chat window")
        if clearButton:
            st.session_state.messages=[]
            st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are helpful assistant.")]

for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)


user_question = st.chat_input("Ask me anything about TestSuite")

if user_question:

    with st.chat_message("User"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(content=user_question))

    with st.chat_message("AI", avatar='💭'):

        context, sources = get_context_from_RAG_DB(user_question, depth)
        
        st.markdown(model_name + "  (depth " + str(depth) + ")" + "  is answering...")
        start_time = time.time()
        ai_response = st.write_stream(get_AI_response(user_question, context, model_name))
        st.session_state.messages.append(AIMessage(content= ai_response))

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        st.markdown(":yellow[Response time: " + str(duration) + " sec]")

        if show_refs:
            st.markdown("\n\n References: \n\n")
            st.markdown(sources)
        






