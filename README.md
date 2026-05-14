# RAG Adversarial Document Injection — Experimental Evaluation

A self-contained research framework for measuring how adversarial document
injection degrades the reliability of Retrieval-Augmented Generation (RAG) systems,
and for evaluating candidate mitigations.

---

## Project Structure

```
rag_adversarial/
├── rag_pipeline.py       # Core RAG system (embeddings + FAISS + Claude)
├── adversarial_eval.py   # Evaluation harness, mitigations, demo experiment
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the experiment

```bash
python adversarial_eval.py
```

---

## What the experiment does

| Phase | Description |
|---|---|
| **Clean baseline** | Knowledge base contains only factual documents. Test queries are run and answers evaluated for correctness. |
| **Poisoned** | Adversarial documents are injected. The same queries are run and the increase in incorrect / biased answers is measured. |
| **Mitigated** | A keyword blocklist mitigation is applied to demonstrate how filtering can partially restore accuracy. |

Results are saved to `eval_results.json` for further analysis.

---

## Key Metrics

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
