from PyPDF2 import PdfReader
import streamlit as st


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

            st.error(
                f"Error reading PDF: {e}"
            )

    return text