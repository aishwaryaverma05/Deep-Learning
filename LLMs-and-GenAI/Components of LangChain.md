# LangChain — Document Loaders (Data Ingestion)

## 1. What is Data Ingestion?

**Data ingestion** is the process of collecting and loading data from different sources into an application so it can be processed further.

In LLM applications, data can come from:
- PDF and text files
- Word documents
- Websites
- CSV files
- Databases

**Example:** Loading a placement preparation PDF into a chatbot so users can ask questions about its contents.

## 2. What are Document Loaders?

**Document Loaders** are LangChain components that load data from different sources and convert it into a standard `Document` format.

A LangChain `Document` generally contains:

- `page_content`: The actual text content.
- `metadata`: Additional information about the content, such as source, page number, or file name.

### Example

```python
Document(
    page_content="LangChain helps build LLM applications.",
    metadata={"source": "notes.pdf", "page": 1}
)
```

This is an illustrative representation of a loaded document.

## 3. How Document Loaders Work

1. Connect to or access a data source.
2. Extract its content.
3. Convert the extracted content into `Document` objects.
4. Pass the documents to the next processing stage, such as text splitting.

**Workflow:**

`Data Source → Document Loader → Document Objects → Text Splitter → Embeddings → Vector Store`

Document loaders do not automatically perform all the later steps.

## 4. Important Document Loaders

| Loader | Purpose |
|---|---|
| `PyPDFLoader` | Loads text from PDF files |
| `TextLoader` | Loads plain-text files |
| `CSVLoader` | Loads CSV data |
| `WebBaseLoader` | Loads content from web pages |
| `DirectoryLoader` | Loads files from a directory |
| `Docx2txtLoader` | Loads text from Word documents |

Availability and dependencies can vary by LangChain integration package.

## 5. Practical Example — Loading a PDF

Install the required packages:

```bash
pip install langchain-community pypdf
```

Python code:

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("placement_notes.pdf")

documents = loader.load()

print("Total pages:", len(documents))
print(documents[0].page_content)
print(documents[0].metadata)
```

**Explanation:**
- `PyPDFLoader`: Creates a loader for the PDF.
- `load()`: Loads the PDF and returns a list of documents, typically one per page.
- `page_content`: Contains extracted text.
- `metadata`: Contains information such as the source file and page number.

**Note:** The PDF must exist at the specified path. Scanned PDFs may require OCR if they do not contain extractable text.

## 6. Important Methods

- `load()`: Loads the data and returns documents.
- `lazy_load()`: Yields documents progressively, useful for larger sources when supported.
- `load_and_split()`: Loads and splits documents using a text splitter; explicit splitting is often preferable when you need control over chunk size and overlap.

## 7. Why Are Document Loaders Important in RAG?

In **Retrieval-Augmented Generation (RAG)**, an LLM needs relevant information from external documents to answer questions.

Document loaders provide the initial content that can later be:
1. Split into smaller chunks.
2. Converted into embeddings.
3. Stored in a vector database.
4. Retrieved to answer a user's query.

Without accessible and correctly extracted source content, the retrieval pipeline cannot work reliably.

## 8. Important Interview Questions

**Q1. What is a document loader in LangChain?**

A component that loads data from a source and converts it into LangChain `Document` objects.

**Q2. What is a LangChain Document?**

An object containing `page_content` and `metadata`.

**Q3. What is the difference between a document loader and a text splitter?**

A loader extracts and loads content; a text splitter divides that content into smaller chunks.

**Q4. Which loader can be used for PDFs?**

`PyPDFLoader`, available through `langchain-community`.

**Q5. What is metadata, and why is it useful?**

Metadata stores information such as source, file name, or page number. It helps with filtering, traceability, and identifying retrieved content.

**Q6. What is the difference between `load()` and `lazy_load()`?**

`load()` returns the loaded documents as a collection, while `lazy_load()` yields them progressively.

**Q7. Do document loaders create embeddings?**

No. Loading documents and generating embeddings are separate steps.

---

## Quick Revision

- **Data ingestion:** Bringing data into an application.
- **Document loader:** Extracts data and converts it into `Document` objects.
- **`page_content`:** Actual document text.
- **`metadata`:** Information about the document.
- **`load()`:** Loads documents.
- **`lazy_load()`:** Yields documents progressively.
- **Next step in RAG:** Text splitting / chunking.
