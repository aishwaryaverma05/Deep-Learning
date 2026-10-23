# Week 6: RAG Improvement

## 1. Retrieval Quality

Retrieval quality measures how well a RAG system retrieves relevant information for a user's question.

### Why is it important?
- Helps the LLM generate accurate answers.
- Reduces irrelevant information.
- Improves source-based answering.

### How to improve retrieval quality?
- Use meaningful chunking.
- Use suitable embedding models.
- Apply metadata filtering.
- Tune the top-k value.
- Use reranking.
- Combine keyword and semantic search using hybrid search.

**Example:** For a question about refund policies, retrieving the refund section is better than retrieving shipping information.

---

## 2. Top-k Retrieval

Top-k specifies how many of the highest-ranked search results are returned by the retriever.

**Example:** If `k = 3`, the retriever returns the top 3 chunks.

### Effect of top-k

- **Low top-k:** Important information may be missed.
- **High top-k:** More irrelevant information may be included, increasing token usage and latency.
- **Suitable top-k:** Provides enough relevant information without unnecessary context.

### How to choose top-k?
1. Test different values such as 3, 5 and 10.
2. Check the relevance of retrieved chunks.
3. Compare answer quality, latency and token usage.
4. Choose the value that works best for your documents and questions.

**Remember:** Higher top-k does not always mean better retrieval.

---

## 3. Reranking

Reranking reorders initially retrieved chunks according to their relevance to the user's question.

### How does it work?
1. Retrieve candidate chunks using the initial search.
2. Pass the question and candidate chunks to a reranker.
3. Score or rank the chunks according to relevance.
4. Send the most relevant chunks to the LLM.

**Example:** Initial retrieval returns a refund policy at rank 4. Reranking moves it to rank 1 because it directly answers the question.

### Retrieval vs Reranking

- **Retrieval:** Finds relevant candidate chunks.
- **Reranking:** Improves the order of those candidates.

### Limitation
A reranker usually cannot recover a relevant chunk that was never retrieved in the first place.

**Remember:** Retrieval finds candidates; reranking improves their order.

---

## 4. Hybrid Search

Hybrid search combines keyword search and semantic search.

### Keyword Search
Finds documents using matching words or terms.

Example: Searching for `ERR-404` to find documentation about that exact error code.

A common keyword-search method is BM25.

### Semantic Search
Finds documents based on meaning and context rather than only exact word matches.

Example: Searching for "battery draining quickly" may retrieve a document about "rapid battery discharge."

Semantic search commonly uses embeddings and vector similarity.

### How Hybrid Search Works
1. Run keyword search.
2. Run semantic search.
3. Combine their results.
4. Rank the combined results.
5. Send relevant chunks to the LLM.

A common result-combination method is **Reciprocal Rank Fusion (RRF)**.

### When is hybrid search useful?
- Technical documentation
- Company knowledge bases
- Customer support applications
- Documents containing exact identifiers and natural-language descriptions

**Remember:** Keyword search focuses on matching terms; semantic search focuses on meaning; hybrid search combines both.

---

## 5. Evaluating RAG Answers

RAG evaluation checks whether the system retrieves the right information and generates a useful, supported answer.

### A. Retrieval Quality
Checks whether the correct information was retrieved.

Example: The retriever returns the refund policy when asked about refunds.

### B. Answer Relevance
Checks whether the generated answer addresses the user's question.

Example: A question about refunds receives an answer about the refund period, not company history.

### C. Faithfulness
Checks whether the generated answer is supported by the retrieved context.

Example:

Context: "Eligible products can be returned within 30 days."

Faithful answer: "Eligible products can be returned within 30 days."

Unfaithful answer: "All products can be returned within 90 days."

The second answer introduces unsupported information.

### Difference Between the Three

| Metric | What it checks |
|---|---|
| Retrieval quality | Was the correct information retrieved? |
| Answer relevance | Does the answer address the question? |
| Faithfulness | Is the answer supported by the retrieved context? |

### Basic Evaluation Process
1. Prepare a set of representative questions.
2. Identify the expected answers and supporting documents.
3. Run the questions through the RAG application.
4. Inspect the retrieved chunks.
5. Evaluate answer relevance and faithfulness.
6. Record failures and improve the system.
7. Run the same tests again to compare results.

Tools such as **Ragas** and **DeepEval** can help automate parts of RAG evaluation.

---

## 6. Build a RAG Application Using Your Own Documents

### Project: Personal Document Q&A Assistant

Build an application that answers questions using your own PDFs, notes or text documents.

### Example Questions
- What is hybrid search?
- Explain the difference between retrieval and reranking.
- Summarize the key points from my notes.
- Which document explains faithfulness?

### Technology Stack

| Component | Suggested Technology |
|---|---|
| Programming language | Python |
| Document loading | PyMuPDF |
| Chunking | Python / LangChain |
| Embeddings | Sentence Transformers or an embedding API |
| Vector database | ChromaDB |
| Retrieval | ChromaDB similarity search |
| Reranking | Cross-encoder model |
| LLM | API-based or local model |
| Interface | Streamlit |
| Evaluation | Manual test set, optionally Ragas |

### Application Workflow

1. **Document loading:** Extract text from PDFs or text files.
2. **Chunking:** Split text into meaningful chunks.
3. **Embeddings:** Convert chunks into numerical vectors.
4. **Storage:** Store vectors and metadata in ChromaDB.
5. **Retrieval:** Search for relevant chunks using the user's question.
6. **Reranking:** Reorder retrieved chunks by relevance.
7. **Answer generation:** Pass relevant context to the LLM.
8. **Source citations:** Display source filenames and page numbers.
9. **Evaluation:** Measure retrieval quality, relevance and faithfulness.

### Improving the Application

Start with basic vector similarity search. Then:

- Experiment with different top-k values.
- Add reranking.
- Implement hybrid search.
- Compare results before and after each improvement.
- Test whether the answer is supported by the retrieved sources.

### Project Completion Checklist

- [ ] Upload and process personal documents.
- [ ] Retrieve relevant document chunks.
- [ ] Generate context-based answers.
- [ ] Display source filenames and page numbers.
- [ ] Test multiple top-k values.
- [ ] Implement and compare reranking.
- [ ] Implement and compare hybrid search.
- [ ] Evaluate retrieval quality, answer relevance and faithfulness.
- [ ] Handle questions for which the documents contain no answer.

---

## 7. Quick Revision

| Concept | One-line meaning |
|---|---|
| Retrieval quality | How well relevant information is retrieved |
| Top-k | Number of top search results returned |
| Reranking | Reordering retrieved chunks by relevance |
| Keyword search | Finds matching words or terms |
| Semantic search | Finds information by meaning |
| Hybrid search | Combines keyword and semantic search |
| RRF | Combines ranked results from multiple searches |
| Answer relevance | Checks whether the answer addresses the question |
| Faithfulness | Checks whether the answer is supported by context |
| RAG evaluation | Measures the quality of retrieval and generated answers |

## Final Takeaway

A good RAG application must retrieve relevant information, provide context-supported answers and identify its sources.

Improving retrieval with suitable top-k settings, reranking and hybrid search can improve answer quality, but the actual improvement must be verified through evaluation.
