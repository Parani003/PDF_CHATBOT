from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from langchain_groq import ChatGroq

from src.vector_store import load_vector_store
from src.config import GROQ_API_KEY


def get_qa_chain():

    prompt_template = """
    You are an intelligent PDF assistant.

    Answer the question only from the provided context.

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

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    db = load_vector_store()

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain