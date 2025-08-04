# unified_methodology.py

import os
import pandas as pd
from dotenv import load_dotenv

# Langchain, Chroma, Streamlit, etc.
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma as CommunityChroma
from langchain_groq import ChatGroq
from langchain.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
class Config:
    YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=uFhDGagZzjs"
    YOUTUBE_AUDIO_SAVE_DIRECTORY = "docs/youtube/"
    PDF_SOURCE_DIRECTORY = "data"
    CHROMA_PERSIST_DIRECTORY = "docs/chroma"
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
    CHUNK_SIZE = 2028
    CHUNK_OVERLAP = 250

    def __init__(self):
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        os.makedirs(self.YOUTUBE_AUDIO_SAVE_DIRECTORY, exist_ok=True)

config = Config()

# --- Ingestion Functions ---

def load_youtube_content(url, save_dir):
    loader = GenericLoader(
        YoutubeAudioLoader([url], save_dir),
        FasterWhisperParser()
    )
    youtube_docs = loader.load()
    return youtube_docs

def load_pdf_content(pdf_directory):
    all_pdf_docs = []
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_directory, filename)
            loader = PyPDFLoader(filepath)
            pages = loader.load()
            all_pdf_docs.extend(pages)
    return all_pdf_docs

def ingest_all_documents():
    youtube_docs = load_youtube_content(config.YOUTUBE_VIDEO_URL, config.YOUTUBE_AUDIO_SAVE_DIRECTORY)
    pdf_docs = load_pdf_content(config.PDF_SOURCE_DIRECTORY)
    combined_docs = youtube_docs + pdf_docs
    if not combined_docs:
        print("No documents loaded!")
        return
    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunked_docs = splitter.split_documents(combined_docs)
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    # Persist Chroma DB
    os.makedirs(config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )
    vectordb.persist()
    print(f"Ingested {len(chunked_docs)} chunks.")

# --- Retrieval QA Functionality ---

def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

def get_vector_store(embed_func):
    return CommunityChroma(
        persist_directory=config.CHROMA_PERSIST_DIRECTORY, 
        embedding_function=embed_func
    )

def get_chat_model():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=400
    )

def call_model(state: MessagesState):
    system_prompt = (
        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. "
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = get_chat_model().invoke(messages)
    return {"messages": response}

def build_langgraph_app():
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_node("model", call_model)
    workflow.add_edge(START, "model")
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# --- Example Usage ---

if __name__ == "__main__":
    # 1. Ingest documents (YouTube + PDF) and build Chroma vector DB
    ingest_all_documents()
    # 2. Initialize embeddings and vectorstore for retrieval
    embeddings = get_embeddings_model()
    vectordb = get_vector_store(embeddings)
    # 3. Chat interface (for demonstration, a simple console QA loop)
    app = build_langgraph_app()
    thread_id = "demo_thread"

    while True:
        prompt = input("Ask a question (or type 'exit'): ")
        if prompt.strip().lower() == "exit":
            break
        docs = vectordb.similarity_search_with_score(prompt, k=3)
        df = pd.DataFrame(
            [(prompt, doc[0].page_content, doc[0].metadata.get('source'), doc[0].metadata.get('page'), doc[1]) for doc in docs],
            columns=['query', 'paragraph', 'document', 'page_number', 'relevant_score']
        )
        context = "\n\n".join(df['paragraph'])
        message = HumanMessage(content=f"Context: {context}\n\nQuestion: {prompt}")
        result = app.invoke(
            {"messages": [message]},
            config={"configurable": {"thread_id": thread_id}},
        )
        ai_response = result['messages'][-1].content
        print("\nAssistant:", ai_response)
        print("-" * 40)
