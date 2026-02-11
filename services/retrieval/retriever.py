from typing import List, Optional, Any, Dict, Tuple
import time
import logging

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Filter
from sentence_transformers import SentenceTransformer, CrossEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Retriever:
    """
    Retriever for Qdrant + SentenceTransformer embeddings + CrossEncoder reranker.

    Returns a list of dicts with standardized fields:
      {
        "id": <point id>,
        "payload": <payload dict or None>,
        "vector_score": <float or None>,
        "rerank_score": <float or None>,
        "combined_score": <float or None>
      }
    """

    def __init__(
        self,
        vectorDB_url: str = "http://qdrant:6333",
        embedding_model_name_path: str = "./models/embedding/multilingual-e5-base",
        rerank_model_name_path: str = "./models/reranker/ms-marco-MiniLM-L-6-v2",
        vectorDB_collection_name: str = "internal_docs",
        timeout: int = 10,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load embedding model
        self.embedding_model = SentenceTransformer(
            embedding_model_name_path,
            device=self.device,
        )

        # Load reranker
        self.rerank_model = CrossEncoder(
            rerank_model_name_path,
            device=self.device,
            max_length=256,
        )

        self.collection = vectorDB_collection_name
        self.client = QdrantClient(url=vectorDB_url, timeout=timeout)

        # Verify collection exists
        try:
            collections = [c.name for c in self.client.get_collections().collections]
        except Exception as e:
            logger.exception("Failed to fetch collections from Qdrant")
            raise

        if self.collection not in collections:
            raise ValueError(f"Collection '{self.collection}' not found!")

    def preprocess_query(self, query: str) -> str:
        q = query.strip().replace("\n", " ")
        q = " ".join(q.split())
        return q  # do not lower by default; let caller decide if needed

    def _safe_extract_text(self, hit: Any) -> Optional[str]:
        """
        Extract text payload safely. Qdrant ScoredPoint uses .payload (dict) in Python client.
        """
        payload = getattr(hit, "payload", None) or getattr(hit, "payload", {})
        if isinstance(payload, dict):
            # common key names: "text", "content", "page_content"
            for key in ("text", "content", "page_content"):
                if key in payload:
                    return payload.get(key)
        return None

    def _ensure_points_list(self, qdrant_resp: Any) -> List[Any]:
        """
        qdrant_resp might be a QueryResponse with attribute .points or might already be a list.
        Normalize to a list of ScoredPoint-like objects.
        """
        if qdrant_resp is None:
            return []
        if hasattr(qdrant_resp, "points"):
            return list(qdrant_resp.points)
        if isinstance(qdrant_resp, list):
            return qdrant_resp
        # fallback: try iterable
        try:
            return list(qdrant_resp)
        except Exception:
            return []

    def rerank(
        self,
        query: str,
        hits: List[Any],
        top_n: int = 5,
        batch_size: int = 8,
        combine_with_vector_score: bool = True,
        alpha: float = 0.7,
    ) -> List[Dict]:
        """
        hits: list of Qdrant ScoredPoint-like objects (with .payload, .id, .score)
        Returns: list of dicts with rerank_score and optionally combined_score.
        """
        hits = self._ensure_points_list(hits)
        if not hits:
            return []

        pairs = []
        meta = []
        for hit in hits:
            text = self._safe_extract_text(hit)
            if not text:
                # skip hits without usable payload text
                continue
            pairs.append((query, text))
            meta.append(hit)

        if not pairs:
            return []

        # Cross-encoder predict
        try:
            # predict returns array-like scores (float)
            rerank_scores = self.rerank_model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        except Exception as e:
            logger.exception("Reranker failed")
            # fallback: return original vector-ranked results mapped to None rerank_score
            return [
                {
                    "id": getattr(hit, "id", None),
                    "payload": getattr(hit, "payload", None),
                    "vector_score": float(getattr(hit, "score", None)) if getattr(hit, "score", None) is not None else None,
                    "rerank_score": None,
                    "combined_score": float(getattr(hit, "score", None)) if getattr(hit, "score", None) is not None else None,
                }
                for hit in hits
            ]

        # Build results with both scores
        results = []
        # Normalize vector scores (if present) to [0,1] for combining
        vec_scores = [float(getattr(h, "score", 0.0) or 0.0) for h in meta]
        # simple min-max normalization (avoid div by zero)
        min_s, max_s = (min(vec_scores), max(vec_scores)) if vec_scores else (0.0, 0.0)
        vec_norm = []
        for s in vec_scores:
            if max_s - min_s > 1e-12:
                vec_norm.append((s - min_s) / (max_s - min_s))
            else:
                vec_norm.append(0.0)

        for hit_obj, rerank_score, vnorm in zip(meta, rerank_scores, vec_norm):
            vid = getattr(hit_obj, "id", None)
            payload = getattr(hit_obj, "payload", None)
            vscore = float(getattr(hit_obj, "score", None)) if getattr(hit_obj, "score", None) is not None else None
            rerank_f = float(rerank_score)
            combined = None
            if combine_with_vector_score and vscore is not None:
                combined = float(alpha * rerank_f + (1 - alpha) * vnorm)
            results.append(
                {
                    "id": vid,
                    "payload": payload,
                    "vector_score": vscore,
                    "rerank_score": rerank_f,
                    "combined_score": combined,
                }
            )

        # sort by combined_score if present, else by rerank_score
        results.sort(key=lambda r: (r["combined_score"] is not None, r["combined_score"] or r["rerank_score"]), reverse=True)
        return results[:top_n]

    def search(self, query: str, top_k: int = 20, filters: Optional[Filter] = None, top_n_rerank: int = 5):
        """
        End-to-end: preprocess -> embed -> qdrant -> rerank -> return standardized list.
        """
        if not query or not query.strip():
            return []

        query = self.preprocess_query(query)

        try:
            t0 = time.time()
            # optionally add a prefix if your embedding model benefits from it — keep configurable
            q_text = f"query: {query}"
            q_emb = self.embedding_model.encode(q_text, normalize_embeddings=True)
            # encode returns numpy array; convert to list for qdrant
            q_emb_list = q_emb.tolist()
            t1 = time.time()
        except Exception as e:
            logger.exception("Failed to create embedding for query")
            return []

        try:
            # call qdrant
            resp = self.client.query_points(
                collection_name=self.collection,
                query=q_emb_list,
                limit=top_k,
                query_filter=filters,
                with_payload=True,
                score_threshold=0.2,
            )
            t2 = time.time()
        except Exception as e:
            logger.exception("Qdrant query failed")
            return []

        # normalize points list
        points = self._ensure_points_list(resp)

        # rerank top candidates
        reranked = self.rerank(query=query, hits=points, top_n=top_n_rerank)

        # attach some retrieval timings (optional)
        retrieval_ms = {
            "embed_ms": int((t1 - t0) * 1000),
            "qdrant_ms": int((t2 - t1) * 1000),
            "total_ms": int((t2 - t0) * 1000),
        }

        return {"results": reranked, "timings_ms": retrieval_ms}

