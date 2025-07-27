# Emma Watson UN Speech Summarizer using LangChain + OpenAI + FAISS

This project demonstrates how to create a simple question-answering system using LangChain, OpenAI embeddings, and FAISS vector storage, based on the transcript of Emma Watson’s UN speech.

---

## 🧠 Overview

We:
- Extracted the transcript of Emma Watson’s UN speech from a PDF
- Preprocessed and split the text into chunks
- Embedded these chunks using OpenAI’s embeddings
- Stored the vectors in a FAISS vector store
- Used LangChain’s RetrievalQA to build a Q&A system over the document

---

## 📦 Setup Instructions

```bash
pip install langchain openai langchain_community pdfplumber faiss-cpu
```

---

## 📁 Code Structure

### 1. **Download and Extract PDF**
We fetch the speech PDF from the UN website and extract the full text using `pdfplumber`.

```python
import requests, io, pdfplumber

url = "https://www.un.int/sites/www.un.int/files/IAPR/full-transcript-of-emma-watson.pdf"
response = requests.get(url)
with pdfplumber.open(io.BytesIO(response.content)) as pdf:
    all_text = "\n\n".join(page.extract_text() for page in pdf.pages)
with open("emma_speech_un_transcript.txt", "w", encoding="utf-8") as f:
    f.write(all_text)
```

---

### 2. **Load and Split Text**
We load the transcript and split it into chunks of 1000 characters using LangChain’s `CharacterTextSplitter`.

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

documents = TextLoader('emma_speech_un_transcript.txt').load()
docs = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(documents)
```

---

### 3. **Embed Text using OpenAI**
We embed each chunk using OpenAI’s `text-embedding-ada-002` model.

```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()
```

---

### 4. **Store Embeddings using FAISS**
We store the embedded vectors into a FAISS vector store.

```python
from langchain.vectorstores import FAISS
db = FAISS.from_documents(docs, embeddings)
```

We can inspect:
```python
print(db.index.ntotal)           # total vectors
print(db.index.reconstruct(0))   # first vector
print(db.docstore._dict)         # document mapping
```

---

### 5. **Perform Similarity Search**
We can now search the vector store for semantically similar text chunks.

```python
query = "What is the name of Emma's campaign?"
docs_result = db.similarity_search(query)
print(docs_result[0].page_content)
```

---

### 6. **Build a Q&A System**
We use `RetrievalQA` from LangChain to query the vector store through OpenAI's LLM.

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=db.as_retriever(),
    return_source_documents=True
)

result = qa_chain({"query": "What did Emma tell herself firmly?"})
print("Answer:", result["result"])
```

---

## 🔍 Sample Questions

- What is the name of Emma's campaign?
- How many girls will be married in the next 16 years?
- What did Emma tell herself firmly?

---

## 🧪 Tech Stack

- LangChain
- OpenAI (Embeddings + LLM)
- FAISS
- pdfplumber
- Python

---

## ✅ Learnings

- How to ingest PDF data and create embeddings
- How to use FAISS as a vector store
- How LangChain simplifies chaining LLMs with retrievers
- A great intro to Retrieval-Augmented Generation (RAG)

---

## ⚠️ Notes

- You must set your OpenAI API key using `os.environ["OPENAI_API_KEY"] = "sk-..."`

---

## 📁 License

MIT
