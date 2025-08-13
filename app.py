import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os

load_dotenv()

# --- CONFIG ---
st.set_page_config(page_title="Week 1 RAG Chatbot", layout="wide")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- SIDEBAR ---
st.sidebar.header("Knowledge Base")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])

# --- ASSISTANT PERSONALITY ---
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

# --- PERSONALITY PROMPTS ---
personality_prompts = {
    "Helpful Teacher": "You are a patient and clear teacher who explains answers simply.",
    "Sarcastic Genius": "You are witty and sarcastic, but still accurate and informative.",
    "Friendly Travel Guide": "You are warm, cheerful, and speak like a travel guide.",
    "Corporate Consultant": "You answer in a professional, business-oriented tone.",
    "Storyteller": "You explain everything as if telling a story, adding vivid imagery."
}
persona_instruction = personality_prompts.get(personality, "")

# --- TITLE ---
st.title("🤖 RAG Chatbot with Personality")

# --- DISPLAY CHAT HISTORY ---
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

# --- CHAT INPUT ---
if prompt := st.chat_input("Type your message here..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.chat_history.append(("user", prompt))

    # Prepare LLM
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY)

    if st.session_state.vector_store:
        retriever = st.session_state.vector_store.as_retriever()
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            verbose=False
        )
        with st.spinner("Thinking..."):
            result = chain({
                "question": f"{persona_instruction}\n\nUser question: {prompt}",
                "chat_history": [
                    (q, a) for role, msg in st.session_state.chat_history if role == "user" or role == "assistant"
                    for q, a in [(msg, "")]
                ]
            })
        answer = result["answer"]
    else:
        # Fallback to normal LLM without RAG
        with st.spinner("Thinking..."):
            response = llm.invoke(f"{persona_instruction}\n\nUser question: {prompt}")
        answer = response.content

    # Display assistant message
    st.chat_message("assistant").markdown(answer)
    st.session_state.chat_history.append(("assistant", answer))

# --- RESET BUTTON ---
if st.sidebar.button("Reset Chat"):
    st.session_state.vector_store = None
    st.session_state.chat_history = []
    st.rerun()