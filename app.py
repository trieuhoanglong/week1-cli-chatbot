import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os

# --- CONFIG ---
st.set_page_config(page_title="Week 1 RAG Chatbot", layout="wide")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- SIDEBAR ---
st.sidebar.header("Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# --- CHOOSE PERSONALITY ---
st.sidebar.header("Assistant Personality")
personality = st.sidebar.selectbox(
    "Choose a style:",
    [
        "Helpful Teacher",
        "Sarcastic Genius",
        "Friendly Travel Guide",
        "Corporate Consultant",
        "Storyteller"
    ]
)

# --- SESSION STATE ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- LOAD PDF ---
if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_documents(docs, embeddings)
    st.session_state.vector_store = vector_store
    st.sidebar.success("Knowledge base loaded!")

# --- CHAT ---
st.title("🤖 RAG Chatbot with Personality")

if st.session_state.vector_store:
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)
    retriever = st.session_state.vector_store.as_retriever()

    # Persona instruction
    personality_prompts = {
        "Helpful Teacher": "You are a patient and clear teacher who explains answers simply.",
        "Sarcastic Genius": "You are witty and sarcastic, but still accurate and informative.",
        "Friendly Travel Guide": "You are warm, cheerful, and speak like a travel guide.",
        "Corporate Consultant": "You answer in a professional, business-oriented tone.",
        "Storyteller": "You explain everything as if telling a story, adding vivid imagery."
    }
    persona_instruction = personality_prompts.get(personality, "")

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        verbose=False
    )

    query = st.text_input("Ask me anything about your document:")
    if query:
        full_prompt = f"{persona_instruction}\n\nUser question: {query}"
        with st.spinner("Thinking..."):
            result = chain({"question": full_prompt, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((query, result["answer"]))
        st.markdown(f"**{personality} says:** {result['answer']}")

# --- RESET BUTTON ---
if st.sidebar.button("Reset Chat"):
    st.session_state.vector_store = None
    st.session_state.chat_history = []
    st.rerun()