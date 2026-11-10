"""Small local document retrieval for the Session 3 learning experiment."""

import hashlib
import json
import math
import os
import re
from pathlib import Path

from openai import OpenAI


def _safe_error(error):
    message = f"{type(error).__name__}: {error}"
    secret = os.environ.get("OPENAI_API_KEY", "").strip()
    return message.replace(secret, "[REDACTED]") if secret else message


class EmbeddingModel:
    """Create vectors with the configured OpenAI embeddings model."""

    def __init__(self, client=None, model=None):
        self.model = (model or os.environ.get("TINYAGENT_EMBEDDING_MODEL", "text-embedding-3-small")).strip()
        if not self.model:
            raise ValueError("TINYAGENT_EMBEDDING_MODEL must be non-empty.")
        self.client = client

    def embed(self, texts):
        values = [texts] if isinstance(texts, str) else list(texts)
        if not values or any(not isinstance(text, str) or not text.strip() for text in values):
            raise ValueError("Embedding inputs must be non-empty text.")
        client = self.client
        owned_client = False
        if client is None:
            api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise ValueError("Set OPENAI_API_KEY before using document retrieval.")
            client = OpenAI(api_key=api_key, timeout=60.0, max_retries=0)
            owned_client = True
        try:
            response = client.embeddings.create(model=self.model, input=values)
        finally:
            if owned_client:
                client.close()
        vectors = [list(item.embedding) for item in response.data]
        if len(vectors) != len(values):
            raise ValueError("Embedding response count did not match input count.")
        return vectors


class RAGMemory:
    """Load a small Markdown corpus and cache embeddings by content and model."""

    def __init__(self, corpus_dir=None, embedding_model=None, cache_path=None,
                 top_k=3, threshold=0.30):
        self.corpus_dir = Path(corpus_dir or Path(__file__).with_name("documents"))
        self.embedding_model = embedding_model or EmbeddingModel()
        self.cache_path = Path(cache_path or Path(__file__).with_name(".cache") / "embeddings.json")
        self.top_k = max(1, int(os.environ.get("TINYAGENT_RETRIEVAL_TOP_K", top_k)))
        self.threshold = float(os.environ.get("TINYAGENT_RETRIEVAL_THRESHOLD", threshold))
        self.corpus_errors = []
        self.documents = self._load_documents()
        self.last_error = ""

    def _load_documents(self):
        documents = []
        seen_ids = set()
        if not self.corpus_dir.exists():
            return documents
        for path in sorted(self.corpus_dir.glob("*.md")):
            try:
                text = path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError) as error:
                self.corpus_errors.append({"file": path.name, "error": _safe_error(error)})
                continue
            match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
            if not match:
                self.corpus_errors.append({"file": path.name, "error": "Missing or malformed front matter."})
                continue
            metadata, body = {}, match.group(2).strip()
            for line in match.group(1).splitlines():
                key, sep, value = line.partition(":")
                if sep:
                    metadata[key.strip()] = value.strip()
            if metadata.get("id") and metadata.get("title") and metadata.get("source") and body:
                if metadata["id"] in seen_ids:
                    self.corpus_errors.append({"file": path.name, "error": f"Duplicate document id: {metadata['id']}"})
                    continue
                documents.append({"id": metadata["id"], "title": metadata["title"],
                                  "source": metadata["source"], "text": body})
                seen_ids.add(metadata["id"])
            else:
                self.corpus_errors.append({"file": path.name, "error": "Document needs id, title, source, and non-empty text."})
        return documents

    def _cache_key(self, document):
        material = self.embedding_model.model + "\0" + document["text"]
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _read_cache(self):
        try:
            value = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def rebuild_cache(self):
        self.cache_path.unlink(missing_ok=True)
        return self.index(force=True)

    def index(self, force=False):
        if not self.documents:
            return {}
        cache = {} if force else self._read_cache()
        missing = [doc for doc in self.documents if self._cache_key(doc) not in cache]
        if missing:
            vectors = self.embedding_model.embed([doc["text"] for doc in missing])
            for doc, vector in zip(missing, vectors):
                cache[self._cache_key(doc)] = vector
        valid_keys = {self._cache_key(doc) for doc in self.documents}
        trimmed = {key: vector for key, vector in cache.items() if key in valid_keys}
        if missing or len(trimmed) != len(cache):
            cache = trimmed
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(cache), encoding="utf-8")
        return {doc["id"]: cache[self._cache_key(doc)] for doc in self.documents}

    @staticmethod
    def _cosine(left, right):
        if not left or len(left) != len(right):
            return 0.0
        denom = math.sqrt(sum(x*x for x in left) * sum(y*y for y in right))
        return sum(x*y for x, y in zip(left, right)) / denom if denom else 0.0

    def search(self, query, top_k=None):
        self.last_error = ""
        query = (query or "").strip()
        if not query or not self.documents:
            return []
        try:
            vectors = self.index()
            query_vector = self.embedding_model.embed([query])[0]
            ranked = sorted(((self._cosine(query_vector, vectors[doc["id"]]), doc)
                             for doc in self.documents), key=lambda item: (-item[0], item[1]["id"]))
            return [{"document_id": doc["id"], "title": doc["title"], "source": doc["source"],
                     "score": float(score), "text": doc["text"]}
                    for score, doc in ranked[:min(self.top_k, max(1, int(top_k or self.top_k)))]
                    if score >= self.threshold]
        except Exception as error:
            self.last_error = _safe_error(error)
            return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect or rebuild the Session 3 local embedding cache.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Re-embed every current local document.")
    args = parser.parse_args()
    rag = RAGMemory()
    if args.rebuild_cache:
        rag.rebuild_cache()
        print(f"Rebuilt local embedding cache for {len(rag.documents)} documents using {rag.embedding_model.model}.")
    else:
        print(f"Loaded {len(rag.documents)} local documents; model={rag.embedding_model.model}, top_k={rag.top_k}, threshold={rag.threshold}.")
