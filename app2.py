import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Page Configuration
st.set_page_config(
    page_title="PDF Document Assistant",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>

    .main {
        background-color: #0E1117;
    }

    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)


# Extract PDF Text
def get_pdf_text(pdf_docs):

    text = ""

    for pdf in pdf_docs:

        try:

            pdf_reader = PdfReader(pdf)

            for page in pdf_reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text

        except Exception as e:
            st.error(f"Error reading PDF: {e}")

    return text


# Split Text into Chunks
def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_text(text)


# Create Vector Store
def create_vector_store(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# Load Vector Store
def load_local_vector_store():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


# Create QA Chain
def get_qa_chain():

    prompt_template = """
    You are a professional assistant.

    Use ONLY the provided context to answer the question.

    If the answer is not available in the context,
    say:
    "Answer not available in the uploaded documents."

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # GROQ MODEL
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    # Load FAISS DB
    db = load_local_vector_store()

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Retrieval Chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )


# Handle User Query
def handle_query(user_question):

    qa_chain = get_qa_chain()

    response = qa_chain({"query": user_question})

    # Store chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_question
    })

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response["result"]
    })

    return response


# Main App
def main():

    st.title("📄 AI Document Analysis Bot")

    st.subheader(
        "Ask questions from your PDF documents using RAG + Groq AI"
    )

    # Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # SIDEBAR
    with st.sidebar:

        st.header("📂 Document Management")

        uploaded_files = st.file_uploader(
            "Upload PDF Files",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Process Documents"):

            if uploaded_files:

                with st.spinner("Processing documents..."):

                    raw_text = get_pdf_text(uploaded_files)

                    chunks = get_text_chunks(raw_text)

                    create_vector_store(chunks)

                    st.success("✅ Documents Indexed Successfully")

            else:
                st.warning("Please upload PDF files.")

    # CHAT INPUT
    query = st.chat_input(
        "Ask something from the uploaded documents..."
    )

    if query:

        if os.path.exists("faiss_index"):

            response = handle_query(query)

            # Display Chat History
            for message in st.session_state.chat_history:

                with st.chat_message(message["role"]):

                    st.markdown(message["content"])

            # Show Source Chunks
            with st.expander("📚 Reference Chunks"):

                for i, doc in enumerate(
                    response["source_documents"]
                ):

                    st.markdown(f"### Reference {i+1}")

                    st.info(doc.page_content)

        else:
            st.error(
                "Please upload and process documents first."
            )


# Run App
if __name__ == "__main__":
    main()