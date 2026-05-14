# RAG Adversarial Document Injection — Experimental Evaluation

A self-contained research framework for measuring how adversarial document
injection degrades the reliability of Retrieval-Augmented Generation (RAG) systems,
and for evaluating candidate mitigations.

---

## Project Structure

```
COMS4507/
├── rag_pipeline.py           # Core RAG system implementation
├── adversarial_eval.py       # Evaluation harness and mitigation strategies  
├── eval_results.json         # Experiment results and metrics
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## File Descriptions

| File | Purpose |
|------|---------|
| **rag_pipeline.py** | Core RAG implementation with document storage, semantic search using FAISS, and LLM integration via Ollama |
| **adversarial_eval.py** | Evaluation framework for running experiments with test cases, multiple mitigation strategies, and comprehensive metrics |
| **eval_results.json** | Experimental results containing accuracy, attack success rates, and detailed per-query analysis |
| **requirements.txt** | Python package dependencies |

---

## Architecture Overview

### RAG Pipeline (`rag_pipeline.py`)

The core RAG system with the following components:

- **Embedding Model**: Uses `sentence-transformers` (default: `all-MiniLM-L6-v2`) to convert documents and queries into semantic embeddings
- **Vector Index**: FAISS `IndexFlatIP` for efficient dense retrieval using cosine similarity
- **LLM Backend**: Local model inference via Ollama (default: `llama3.2`) with deterministic generation (temperature=0.0)
- **Document Management**: Tracks documents with metadata, source labels (clean/adversarial), and unique IDs
- **Retrieval Pipeline**: Supports filtering and reranking functions for implementing mitigations

Key classes:
- `Document`: Represents a document with text, ID, source label, and metadata
- `RAGPipeline`: Main class orchestrating embedding, retrieval, and generation
- `OllamaClient`: Wrapper for local LLM inference
- `RAGResponse`: Complete output with answer, retrieved documents, and relevance scores

### Evaluation Framework (`adversarial_eval.py`)

Implements comprehensive evaluation with:

- **Test Cases**: Structured queries with ground truth, target attack answers, and keyword-based judgment
- **Evaluation Metrics**: Accuracy, attack success rate, adversarial document retrieval rate, and answer corruption rate
- **Mitigation Strategies**: Keyword blocklist and source-based filtering
- **EvalReport**: Aggregates results with detailed statistics

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Ollama
Dependencies

The project requires:

- `sentence-transformers >= 3.0.0` — For semantic text embeddings
- `faiss-cpu >= 1.8.0` — For efficient vector similarity search
- `numpy >= 1.26.0` — Numerical computing support

Additionally, Ollama must be running locally for LLM inference.

### 3. Run the experiment

```bash
python adversarial_eval.py
```

---

## Extending the framework

### Adding your own documents

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.add_documents(["Your factual document ..."], source="clean")
rag.add_documents(["Misleading content ..."],   source="adversarial")
answer = rag.query("Your question?")
print(answer.answer)
```

### Adding your own test cases

```python
from adversarial_eval import TestCase

my_case = TestCase(
    query             = "What is X?",
    ground_truth      = "X is ...",
    target_answer     = "X is actually ...",   # what a successful attack produces
    keywords_clean    = ["correct", "keywords"],
    keywords_attack   = ["misleading", "keywords"],
    category          = "your_category",
)
```

### Implementing a new mitigation

Create a filter function with signature `filter_fn(docs: list[RetrievedDocument]) -> list[RetrievedDocument]` and pass it to `rag.retrieve()` as shown in `adversarial_eval.py
| Metric | Description |
|---|---|
| **Accuracy** | Fraction of queries answered correctly (ground-truth keywords found). |
| **Attack success rate (ASR)** | Fraction of queries where the adversarial target answer was produced. |
| **Avg adv docs retrieved** | Mean number of adversarial documents appearing in the top-k context window per query. |

---

## Extending the framework

### Adding your own documents

```python
from rag_pipeline import RAGPipeline

rag = RAGPipeline()
rag.add_documents(["Your factual document ..."], source="clean")
rag.add_documents(["Misleading content ..."],   source="adversarial")
answer = rag.query("Your question?")
print(answer.answer)
```

### Adding your own test cases

```python
from adversarial_eval import TestCase

my_case = TestCase(
    query            = "What is X?",
    ground_truth     = "X is ...",
    target_answer    = "X is actually ...",   # what a successful attack produces
    keywords_clean   = ["correct", "keywords"],
    keywords_attack  = ["misleading", "keywords"],
)
```

### Implementing a new mitigation

---

## Experiment Output

The `eval_results.json` file contains:

- **Per-report metrics**: Accuracy, attack success rate, adversarial retrieval rate, and corruption rate
- **Per-query results**: Individual query performance, retrieved documents, relevance scores, and classification
- **Retrieval analysis**: Document IDs, sources, and ranking information for debugging and analysis

Example report structure:
```json
{
  "label": "Clean baseline",
  "accuracy": 0.9,
  "attack_success_rate": 0.1,
  "adversarial_retrieval_rate": 0.0,
  "avg_adv_docs_retrieved": 0.0,
  "corrupted_answer_rate": 0.1,
  "results": [...]
}
```

Implement a class with a `filter_docs(docs) -> list[Document]` method and wrap
`rag.retrieve` as shown in `adversarial_eval.py → main()`.

---

## Mitigation strategies implemented

| Strategy | Implementation | Limitation |
|---|---|---|
| **Keyword blocklist** | `KeywordBlocklist` — strips docs containing known adversarial phrases | Requires prior knowledge of attack vocabulary |
| **Source score penalty** | `SourceScoreFilter` — adds L2 distance penalty to adversarial-labelled docs | Requires document provenance metadata |

### Recommended further mitigations (not yet implemented)

- **LLM-based document credibility scoring** — ask a second model to rate each retrieved document for factual plausibility before passing to the generator.
- **Cross-encoder re-ranking** — replace FAISS flat L2 with a cross-encoder that scores (query, document) pairs more precisely, making adversarial documents harder to surface.
- **Ensemble retrieval with source diversity constraints** — require retrieved docs to come from multiple distinct trusted sources.
- **Instruction-tuned scepticism** — add system-prompt instructions telling the LLM to flag contradictions across retrieved documents.

---

## Alignment with project methodology

| Step | Covered by |
|---|---|
| 1. Literature review | See references below |
| 2. RAG framework selection | `rag_pipeline.py` (sentence-transformers + FAISS + Claude) |
| 3. Adversarial document design | `ADVERSARIAL_DOCUMENTS` in `adversarial_eval.py` |
| 4. Injection + standardised queries | `main()` in `adversarial_eval.py` |
| 5. Attack success rate measurement | `EvalReport.attack_success_rate` |
| 6. Mitigations | `KeywordBlocklist`, `SourceScoreFilter` |
| 7. Results + recommendations | `eval_results.json` + this README |

---

## References

- Shafran et al. (2025). *[Adversarial attacks on RAG systems]* — key motivation for this project.
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
- Gao et al. (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey.*
