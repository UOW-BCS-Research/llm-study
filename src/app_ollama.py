from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings, OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools.retriever import create_retriever_tool

import tempfile
import os
from timeit import default_timer as timer

import streamlit as st

# load .env file
from dotenv import dotenv_values
config = dotenv_values(".env")

# vector storage
client = MongoClient(config["MONGODB_ATLAS_CLUSTER_URI"], server_api=ServerApi('1'))
DB_NAME = config["MONGODB_ATLAS_DB_NAME"]
# COLLECTION_NAME = config["MONGODB_ATLAS_COLLECTION_NAME"]
# ATLAS_VECTOR_SEARCH_INDEX_NAME = config["MONGODB_ATLAS_VECTOR_SEARCH_INDEX_NAME"]
# MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# for chat
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

st.set_page_config(page_title="UOWCHK LLM Research Demo", page_icon="")
st.title("UOWCHK LLM Research Demo")

# global variables
openai_api_key = ""
embedding_model_name = ""
embedding_model = None
chat_model_name = ""
chat_model = None

with st.sidebar:
    st.markdown("### Configuration")
    platform = "OpenAI"
    chat_model_name = st.selectbox(
        "Model",
        [
            "gpt-4o-2024-05-13",
            "gpt-4-1106-preview",
            "gpt-3-ada",
            "gpt-3-babbage",
            "gpt-3-curie",
            "gpt-3-davinci",
            "gpt-3-instruct-beta",
            "gpt-3-j1",
            "gpt-3-j6",
            "gpt-3-jumbo",
            "gpt-3-pet",
            "gpt-3-t",
            "gpt-3",
            "davinci-instruct-beta",
            "davinci",
            "curie-instruct-beta",
            "curie",
            "babbage-instruct-beta",
            "babbage",
            "ada-instruct-beta",
            "ada",
        ],
    )

    if config["OPEN_AI_API_KEY"] == "":
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    else:
        openai_api_key = config["OPEN_AI_API_KEY"]

    chat_model = ChatOpenAI(model_name=chat_model_name, openai_api_key=openai_api_key, streaming=True)

    # upload training data
    st.markdown("---")
    st.markdown("### Training Data")
    embedding_model_name = st.selectbox(
        "Embedding model",
        [
            "llama3",                       # dimension: 4096
            "nomic-embed-text",             # dimension: 768
            "mxbai-embed-large",            # dimension: 1024
            "snowflake-arctic-embed",       # dimension: 1024
        ],
    )

    MONGODB_COLLECTION = client[DB_NAME][embedding_model_name]

    # embedding
    embedding_model = OllamaEmbeddings(
        base_url=config["OLLAMA_BASE_URL"],
        model=embedding_model_name,
    )

    training_data = st.file_uploader("Upload training data", type=["pdf"])
    if training_data:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, training_data.name)
        with open(path, "wb") as f:
            f.write(training_data.getvalue())
        st.write(f"Training data uploaded: {training_data.name}, {path}")
        
        loader = PyPDFLoader(path, extract_images=False)
        timer_start = timer()
        pages = loader.load_and_split()
        st.write(f"Time taken to load and split PDF: {timer() - timer_start:.2f} seconds")
        st.write(f"Number of pages: {len(pages)}")

        timer_start = timer()
        # insert the documents in MongoDB Atlas with their embedding
        vector_search = MongoDBAtlasVectorSearch.from_documents(
            documents=pages,
            embedding=embedding_model,
            collection=MONGODB_COLLECTION,
            # index_name=f"search_{embedding_model_name.replace('/', '_')}",
        )
        st.write(f"Time taken to insert documents in MongoDB Atlas: {timer() - timer_start:.2f} seconds")
        # reset training_data
        training_data.close()

        # st.write(pdf_loader.text)
    st.markdown("---")
    st.markdown("### Reset")
    if st.sidebar.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    print('msg', msg, idx)
    with st.chat_message(avatars[msg.type]):
        st.write(msg.content)

if prompt := st.chat_input(placeholder="Type something..."):
    st.chat_message("user").write(prompt)
    msgs.add_user_message(prompt)
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]

        vector_search = MongoDBAtlasVectorSearch.from_connection_string(
            config["MONGODB_ATLAS_CLUSTER_URI"],
            config["MONGODB_ATLAS_DB_NAME"] + "." + embedding_model_name,
            embedding_model,
            index_name=f"search_{embedding_model_name.replace('/', '_')}"
        )

        qa_retriever = vector_search.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        tools = [
            create_retriever_tool(
                qa_retriever,
                "search_knowledge_retriever",
                "Search any knowledge",
            )   
        ]
        chat_agent = ConversationalChatAgent.from_llm_and_tools(llm=chat_model, tools=tools)
        executor = AgentExecutor.from_agent_and_tools(
            agent=chat_agent,
            tools=tools,
            memory=memory,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
        )

        response = executor.invoke(
            {"input": prompt}, {"callbacks": [st_cb]}
        )
        st.write(response["output"])


