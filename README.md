# 📄 AI Document Analysis Bot using RAG + Groq

An intelligent PDF chatbot built using Retrieval-Augmented Generation (RAG), LangChain, FAISS Vector Database, HuggingFace Embeddings, and Groq LLMs.

This application allows users to upload PDF documents and ask questions based on the uploaded content.  
The chatbot retrieves relevant document chunks using semantic similarity search and generates accurate responses using Groq AI models.

---

# 🚀 Features

- 📄 Upload multiple PDF documents
- 🤖 Ask questions from uploaded PDFs
- 🧠 Retrieval-Augmented Generation (RAG)
- 🔍 Semantic similarity search
- 📚 FAISS Vector Database
- ⚡ Fast inference using Groq
- 💬 Chat-style interface
- 📖 Reference chunk viewer
- 🔐 Secure API key management using `.env`

---

# 🏗️ System Architecture

```text
PDF Files
   ↓
Text Extraction
   ↓
Text Chunking
   ↓
Embedding Generation
   ↓
FAISS Vector Database
   ↓
Similarity Search
   ↓
Relevant Context Retrieval
   ↓
Groq LLM Response Generation
