from langchain.chains import LLMChain
import streamlit as st
from decouple import config
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.evaluation.qa import QAGenerateChain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import time
from htmlTemplates import css, bot_template, user_template

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Start time
        result = func(*args, **kwargs)  # Function execution
        end_time = time.time()  # End time
        print(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")
        return result
    return wrapper


@timeit
def get_llm():
    return OpenAI(temperature=0.0)

@timeit
def get_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

@timeit
def generate_response(question, vectordb, llm, memory,chat_history):

    prompt = ChatPromptTemplate.from_template(
        "You are a petroleum engineer specialist in hydralic fracture stimulation \
    , please answer the question that surounded between the triple backtick \
    ```{question}```"
    )
    
    question_template = prompt.format_messages(question=question)
    final_qa = question_template[0].content

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_type="mmr",k=5, fetch_k=10),
        memory=memory,
    )

    handle_userinput((qa({"question": question, "chat_history": chat_history})))

@timeit
def create_embeding_function():
    embedding_func_all_mpnet_base_v2 = SentenceTransformerEmbeddings(
        model_name="all-mpnet-base-v2")
    # embedding_func_all_MiniLM_L6_v2 = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    return embedding_func_all_mpnet_base_v2

@timeit
def get_vector_db(embedding_function):
    vector_db = Chroma(persist_directory="./final_db",
                       embedding_function=embedding_function)
    return vector_db

def handle_userinput(user_question):
    response = user_question
    if chat_history not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
if __name__ == "__main__":

    st.set_page_config(page_title = "Hydraulic Fracture Stimulation Chat",page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.title("Hydraulic Fracture Stimulation Chat")
    st.write(
        "This is a chatbot that can answer questions related to petroleum engineering specially in hydraulic fracture stimulation.")

    # get embeding function   
    embeding_function = create_embeding_function()
    # get vector db
    vector_db = get_vector_db(embeding_function)
    # get llm
    llm = get_llm()

    # get memory
    if 'memory' not in st.session_state:
        st.session_state['memory'] = get_memory()
    memory = st.session_state['memory']

    # chat history
    chat_history = []
   
    prompt_question = st.chat_input("Please ask a question:")
    if prompt_question:
        generate_response(question= prompt_question, vectordb=vector_db, llm=llm, memory=memory,chat_history=chat_history)
