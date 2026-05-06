"""
rag_pipeline.py
---------------
Core Retrieval-Augmented Generation (RAG) pipeline.

Uses:
  - sentence-transformers  → dense embeddings (fully local, free)
  - faiss-cpu              → fast approximate nearest-neighbour search (free)
  - Ollama                 → local LLM generation, no API key needed (free)

Prerequisites
-------------
1. Install Ollama from https://ollama.com
2. Pull a model, e.g.:
       ollama pull llama3.2        (4 GB, recommended)
       ollama pull mistral         (4 GB, alternative)
       ollama pull phi3            (2 GB, lightweight option)
3. Ollama runs automatically in the background once installed.

Usage
-----
    from rag_pipeline import RAGPipeline

    rag = RAGPipeline()                         # uses llama3.2 by default
    rag.add_documents(["Doc 1 text ...", ...])
    answer = rag.query("Your question here")
    print(answer.answer)
"""

from __future__ import annotations

import textwrap
import urllib.request
import json
from dataclasses import dataclass, field
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A single document in the knowledge base."""
    doc_id: int
    text: str
    source: str = "clean"          # "clean" | "adversarial"
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Output of a retrieval call."""
    query: str
    retrieved_docs: list[Document]
    scores: list[float]


@dataclass
class RAGResponse:
    """Full RAG pipeline output."""
    query: str
    answer: str
    retrieved_docs: list[Document]
    scores: list[float]
    adversarial_docs_retrieved: int   # how many retrieved docs were adversarial


# ---------------------------------------------------------------------------
# Ollama client (no external library needed — uses the built-in urllib)
# ---------------------------------------------------------------------------

class OllamaClient:
    """
    Thin wrapper around Ollama's local REST API.
    Ollama must be running (it starts automatically after installation).

    Parameters
    ----------
    model   : Ollama model tag, e.g. "llama3.2", "mistral", "phi3"
    host    : Ollama server address (default: localhost:11434)
    timeout : request timeout in seconds
    """

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.url = f"{host}/api/generate"
        self.timeout = timeout
        self._check_connection(host)

    def _check_connection(self, host: str) -> None:
        """Warn if Ollama is not reachable."""
        try:
            urllib.request.urlopen(host, timeout=3)
        except Exception:
            print(
                f"\n  Could not reach Ollama at {host}.\n"
                "   Make sure Ollama is installed and running:\n"
                "     Download: https://ollama.com\n"
                f"     Then run:  ollama pull {self.model}\n"
            )

    def generate(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the generated text."""
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    A minimal but complete RAG pipeline with support for
    adversarial document injection and retrieval tracking.

    Parameters
    ----------
    embedding_model : HuggingFace model name for sentence embeddings.
    ollama_model    : Ollama model tag to use for generation.
    top_k           : Number of documents to retrieve per query.
    embed_dim       : Dimensionality of the embedding model's output.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        ollama_model: str = "llama3.2",
        top_k: int = 3,
        embed_dim: int = 384,
    ):
        print(f"[RAGPipeline] Loading embedding model '{embedding_model}' ...")
        self.encoder = SentenceTransformer(embedding_model)
        self.top_k = top_k
        self.embed_dim = embed_dim

        # FAISS index (flat L2 — exact search, suitable for small corpora)
        self.index = faiss.IndexFlatL2(embed_dim)
        self.documents: list[Document] = []

        # Local LLM via Ollama
        print(f"[RAGPipeline] Connecting to Ollama model '{ollama_model}' ...")
        self.llm = OllamaClient(model=ollama_model)

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def add_documents(
        self,
        texts: list[str],
        source: str = "clean",
        metadata: Optional[list[dict]] = None,
    ) -> None:
        """
        Embed and index a list of documents.

        Parameters
        ----------
        texts    : list of raw document strings
        source   : label attached to every document ("clean" or "adversarial")
        metadata : optional per-document dicts
        """
        if not texts:
            return

        metadata = metadata or [{} for _ in texts]
        start_id = len(self.documents)

        embeddings = self.encoder.encode(texts, convert_to_numpy=True).astype("float32")
        self.index.add(embeddings)

        for i, (text, meta) in enumerate(zip(texts, metadata)):
            self.documents.append(
                Document(doc_id=start_id + i, text=text, source=source, metadata=meta)
            )

        print(
            f"[RAGPipeline] Added {len(texts)} '{source}' document(s). "
            f"Total corpus size: {len(self.documents)}"
        )

    def remove_adversarial_documents(self) -> None:
        """Reset the index to contain only clean documents."""
        clean_docs = [d for d in self.documents if d.source == "clean"]
        clean_texts = [d.text for d in clean_docs]

        self.index = faiss.IndexFlatL2(self.embed_dim)
        self.documents = []

        if clean_texts:
            self.add_documents(clean_texts, source="clean")

        print("[RAGPipeline] Adversarial documents removed. Clean baseline restored.")

    @property
    def corpus_size(self) -> int:
        return len(self.documents)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        """Return the top-k most relevant documents for *query*."""
        if self.corpus_size == 0:
            raise ValueError("Knowledge base is empty — call add_documents() first.")

        k = min(self.top_k, self.corpus_size)
        q_emb = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        distances, indices = self.index.search(q_emb, k)

        retrieved = [self.documents[i] for i in indices[0]]
        scores = distances[0].tolist()

        return RetrievalResult(query=query, retrieved_docs=retrieved, scores=scores)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _build_prompt(self, query: str, context_docs: list[Document]) -> str:
        """Format a RAG prompt with retrieved context."""
        context_block = "\n\n".join(
            f"[Document {i+1} | source={doc.source}]\n{doc.text}"
            for i, doc in enumerate(context_docs)
        )
        return textwrap.dedent(f"""
            You are a helpful assistant. Use ONLY the information in the
            documents below to answer the question. If the documents do not
            contain enough information, say so clearly. Be concise.

            --- RETRIEVED DOCUMENTS ---
            {context_block}
            --- END OF DOCUMENTS ---

            Question: {query}

            Answer:
        """).strip()

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def query(self, question: str) -> RAGResponse:
        """
        Run the full RAG pipeline:
          1. Retrieve relevant documents
          2. Build a context-grounded prompt
          3. Generate an answer via Ollama (local, free)
          4. Return a structured RAGResponse
        """
        retrieval = self.retrieve(question)
        prompt = self._build_prompt(question, retrieval.retrieved_docs)
        answer = self.llm.generate(prompt)

        adv_count = sum(
            1 for d in retrieval.retrieved_docs if d.source == "adversarial"
        )

        return RAGResponse(
            query=question,
            answer=answer,
            retrieved_docs=retrieval.retrieved_docs,
            scores=retrieval.scores,
            adversarial_docs_retrieved=adv_count,
        )