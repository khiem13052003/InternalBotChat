# prompt_template.py
"""
Prompt builder OOP for Vietnamese RAG production (LLaMA 3.2 1B via MLC-llm).
Provides:
 - sorting by configurable priority keys
 - deduplication (content_hash -> normalized text fallback)
 - merge contiguous chunks (same source_file & adjacent pages)
 - token budget control with truncation at sentence boundaries
 - build Vietnamese prompt production-ready with citations
Usage:
    from prompt_template import PromptBuilder
    pb = PromptBuilder()
    out = pb.build_prompt_from_retrieval(retrieval_json, "Câu hỏi của user", token_budget=2048)
    print(out["prompt"])
"""
from __future__ import annotations
import math
import datetime
import re
import copy
from typing import List, Dict, Any, Optional, Callable, Tuple


# --------------------------
# Defaults / Configurable
# --------------------------
DEFAULT_CHAR_PER_TOKEN = 4.0     # default: 4 chars ~= 1 token; override for your tokenizer
DEFAULT_RESERVE_TOKENS = 512     # tokens to reserve for system + model response
DEFAULT_TOKEN_BUDGET = 4096

# Default sort order: list of (field_name, mode)
# mode: "auth" special for boolean authoritative; "asc"/"desc" for numeric; "date_desc" for datetime newest first
DEFAULT_SORT_KEYS = [
    ("is_authoritative", "auth"),
    ("source_priority", "asc"),
    ("combined_score", "desc"),
    ("rerank_score", "desc"),
    ("vector_score", "desc"),
    ("created_at", "date_desc"),
]

SYS_INST = (
    "Bạn là một trợ lý nội bộ chuyên nghiệp, trả lời bằng Tiếng Việt. "
    "Chỉ sử dụng thông tin trong phần CONTEXT; nếu không đủ, trả lời 'Tôi không biết' và nêu lý do ngắn gọn. "
    "Không tự bịa thông tin. Luôn trích dẫn nguồn theo định dạng [<tên file>:<trang>]."
)
GEN_INST = (
    "Trả lời ngắn gọn, rõ ràng; nếu cần, dùng gạch đầu dòng. "
    "PHẦN NGUỒN: liệt kê các citation đã dùng, mỗi dòng kèm 1 câu tóm tắt ngắn (1-2 câu). "
    "Nêu mức độ tự tin 0-100% dựa trên mức độ phủ thông tin trong ngữ cảnh."
)

# --------------------------
# Utility functions
# --------------------------
def _safe_get_score(hit: Dict[str, Any], key: str) -> Optional[float]:
    """Try to fetch score-like values from payload or top-level."""
    # prefer top-level if present (e.g., combined_score sometimes at root)
    if key in hit and isinstance(hit.get(key), (int, float)):
        return float(hit.get(key))
    p = hit.get("payload", {})
    v = p.get(key)
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def _parse_iso_datetime(s: Optional[str]) -> Optional[datetime.datetime]:
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        # fallback: try common formats
        try:
            return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")
        except Exception:
            return None


def _normalize_text_key(text: str, max_len: int = 200) -> str:
    if not text:
        return ""
    norm = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE).lower()
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm[:max_len]


# --------------------------
# PromptBuilder class
# --------------------------
class PromptBuilder:
    def __init__(
        self,
        sort_keys: Optional[List[Tuple[str, str]]] = None,
        char_per_token: float = DEFAULT_CHAR_PER_TOKEN,
        reserve_tokens: int = DEFAULT_RESERVE_TOKENS,
        token_estimator: Optional[Callable[[str], int]] = None,
        citation_template: str = "[{file}:{page}]",
    ):
        """
        Args:
            sort_keys: list of (field, mode). See DEFAULT_SORT_KEYS.
            char_per_token: chars per token approximation (used if token_estimator not provided)
            reserve_tokens: number of tokens reserved for response/system
            token_estimator: optional callable(text)->int to compute tokens exactly (preferred)
            citation_template: format string used for citations
        """
        self.sort_keys = sort_keys or DEFAULT_SORT_KEYS
        self.char_per_token = float(char_per_token)
        self.reserve_tokens = int(reserve_tokens)
        self.token_estimator = token_estimator or self._estimate_tokens_fallback
        self.citation_template = citation_template

    # ---------- token estimation ----------
    def _estimate_tokens_fallback(self, text: str) -> int:
        """Fallback estimator: ceil(len(normalized_text) / char_per_token)."""
        if not text:
            return 0
        norm = re.sub(r"\s+", " ", text.strip())
        chars = len(norm)
        return max(1, math.ceil(chars / max(1.0, self.char_per_token)))

    def estimate_tokens(self, text: str) -> int:
        return int(self.token_estimator(text))

    # ---------- sorting ----------
    def sort_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sorts hits according to self.sort_keys.
        Each key considered on payload first, then root.
        """
        def key_fn(hit: Dict[str, Any]):
            p = hit.get("payload", {})
            key_list = []
            for field, mode in self.sort_keys:
                if mode == "auth":
                    val = bool(p.get(field, False))
                    # want True first -> use 0 for True, 1 for False
                    key_list.append(0 if val else 1)
                elif mode == "asc":
                    v = p.get(field)
                    key_list.append(v if v is not None else float('inf'))
                elif mode == "desc":
                    # prefer numeric: try root then payload
                    v = _safe_get_score(hit, field)
                    key_list.append(-v if v is not None else float('inf'))
                elif mode == "date_desc":
                    dt = _parse_iso_datetime(p.get(field) or hit.get(field))
                    key_list.append(-dt.timestamp() if dt else float('inf'))
                else:
                    # fallback
                    v = p.get(field)
                    key_list.append(v if v is not None else 0)
            return tuple(key_list)

        return sorted(hits, key=key_fn)

    # ---------- deduplication ----------
    def deduplicate(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate by content_hash if present, else by normalized text prefix.
        Keep the highest combined_score among duplicates.
        """
        buckets: Dict[str, Dict[str, Any]] = {}
        for h in hits:
            p = h.get("payload", {})
            key = p.get("content_hash")
            if not key:
                key = _normalize_text_key(p.get("text", "") or "")
            existing = buckets.get(key)
            if existing is None:
                buckets[key] = h
            else:
                existing_score = _safe_get_score(existing, "combined_score") or -1e9
                new_score = _safe_get_score(h, "combined_score") or -1e9
                if new_score > existing_score:
                    buckets[key] = h
        return list(buckets.values())

    # ---------- merge contiguous ----------
    def merge_contiguous(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge chunks that share the same source_file and chunk_version when pages are contiguous.
        The merged payload will include 'pages' list and concatenated text.
        """
        groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for h in hits:
            p = h.get("payload", {})
            key = (p.get("source_file", ""), p.get("chunk_version", ""))
            groups.setdefault(key, []).append(h)

        merged: List[Dict[str, Any]] = []
        for key, group in groups.items():
            group_sorted = sorted(group, key=lambda x: (x.get("payload", {}).get("page_number") or 0))
            buffer: Optional[Dict[str, Any]] = None
            for h in group_sorted:
                p = h.get("payload", {})
                page = p.get("page_number")
                if buffer is None:
                    buffer = copy.deepcopy(h)
                    buffer_payload = buffer.setdefault("payload", {})
                    buffer_payload["pages"] = [page] if page is not None else []
                else:
                    prev_pages = buffer["payload"].get("pages", [])
                    last = prev_pages[-1] if prev_pages else None
                    if page is not None and last is not None and page == last + 1:
                        # merge text with separator
                        buffer["payload"]["text"] = buffer["payload"].get("text", "") + "\n\n" + (p.get("text", "") or "")
                        buffer["payload"]["pages"].append(page)
                        # update scores conservatively (take max)
                        cs_existing = _safe_get_score({"payload": buffer["payload"]}, "combined_score") or -1e9
                        cs_new = _safe_get_score(h, "combined_score") or -1e9
                        buffer["payload"]["combined_score"] = max(cs_existing, cs_new)
                        # propagate other metadata if absent
                        if not buffer["payload"].get("created_at"):
                            buffer["payload"]["created_at"] = p.get("created_at")
                    else:
                        merged.append(buffer)
                        buffer = copy.deepcopy(h)
                        buffer_payload = buffer.setdefault("payload", {})
                        buffer_payload["pages"] = [page] if page is not None else []
            if buffer is not None:
                merged.append(buffer)
        # After merging, we can sort merged list by combined_score desc to prioritize
        return sorted(merged, key=lambda x: -(_safe_get_score(x, "combined_score") or _safe_get_score(x, "vector_score") or 0.0))

    # ---------- truncation ----------
    def truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate 'text' so estimated tokens <= max_tokens.
        Prefer to cut at sentence boundary (. ? ! or newline) near the limit.
        """
        if max_tokens <= 0:
            return ""
        if self.estimate_tokens(text) <= max_tokens:
            return text
        allowed_chars = int(max_tokens * self.char_per_token)
        if allowed_chars <= 0:
            return ""
        candidate = text[:allowed_chars]
        # cut at last sentence end
        m = re.search(r"(.+?[\.\?\!])([^\.\?\!]*)\s*$", candidate, re.DOTALL)
        if m:
            return m.group(1).strip()
        # else cut at last newline if meaningful
        idx = candidate.rfind("\n")
        if idx != -1 and idx > int(allowed_chars * 0.4):
            return candidate[:idx].strip()
        # fallback: hard cut
        return candidate.strip()

    # ---------- token budget application ----------
    def apply_token_budget(self, hits: List[Dict[str, Any]], token_budget: int) -> Tuple[List[Dict[str, Any]], int]:
        """
        Select hits in order to fit token_budget - reserve_tokens.
        Returns (selected_hits, used_tokens).
        """
        allowed = max(token_budget - self.reserve_tokens, 0)
        selected: List[Dict[str, Any]] = []
        used = 0
        for h in hits:
            txt = (h.get("payload", {}) or {}).get("text", "") or ""
            est = self.estimate_tokens(txt)
            if used + est <= allowed:
                selected.append(h)
                used += est
            else:
                remaining = allowed - used
                if remaining <= 0:
                    break
                truncated = self.truncate_text_to_tokens(txt, remaining)
                if truncated:
                    new_hit = copy.deepcopy(h)
                    new_hit["payload"] = dict(new_hit.get("payload", {}))
                    new_hit["payload"]["text"] = truncated
                    # keep pages if present but note truncated
                    new_hit["payload"]["truncated"] = True
                    selected.append(new_hit)
                    used += self.estimate_tokens(truncated)
                break
        return selected, used

    # ---------- prompt building ----------
    def format_citation(self, payload: Dict[str, Any]) -> str:
        file = payload.get("source_file", "unknown")
        pages = payload.get("pages") or ([payload.get("page_number")] if payload.get("page_number") is not None else [])
        page_str = pages[0] if isinstance(pages, list) and len(pages) == 1 else (f"{pages[0]}-{pages[-1]}" if pages else "?")
        return self.citation_template.format(file=file, page=page_str)

    def build_vietnamese_prompt(self, user_query: str, selected_hits: List[Dict[str, Any]]) -> str:
        """
        Build final Vietnamese prompt text to feed into LLM.
        """

        parts = []
        # parts.append("[SYSTEM]\n" + sys_inst + "\n")
        parts.append("[CONTEXT]\n")
        if not selected_hits:
            parts.append("(Không có ngữ cảnh được chọn)\n")
        else:
            for idx, h in enumerate(selected_hits, start=1):
                p = h.get("payload", {}) or {}
                citation = self.format_citation(p)
                src = p.get("source_file", "<unknown>")
                pages = p.get("pages") or ([p.get("page_number")] if p.get("page_number") is not None else [])
                pages_str = ",".join(str(x) for x in pages) if pages else "?"
                auth = " (authoritative)" if p.get("is_authoritative") else ""
                sp = p.get("source_priority", "-")
                cs = _safe_get_score(h, "combined_score") or p.get("combined_score", "-")
                header = f"[{idx}] {citation} | file={src} pages={pages_str}{auth} priority={sp} score={cs}"
                text = (p.get("text") or "").strip()
                # keep the full selected text (we already applied token budget)
                parts.append(header + "\n" + text + "\n")
        parts.append("[END CONTEXT]\n")
        parts.append("[USER QUESTION]\n" + user_query + "\n")
        parts.append("[HƯỚNG DẪN]\n" + GEN_INST + "\n")
        return "\n".join(parts)

    # ---------- high-level pipeline ----------
    def build_prompt_from_retrieval(self,
                                   retrieval: Dict[str, Any],
                                   user_query: str,
                                   token_budget: int = DEFAULT_TOKEN_BUDGET) -> Dict[str, Any]:
        """
        Full pipeline:
         1. extract hits
         2. sort
         3. deduplicate
         4. merge contiguous
         5. apply token budget (reserving self.reserve_tokens)
         6. build prompt
        Returns dict: {prompt, selected_hits, used_tokens, token_budget, reserve_tokens}
        """
        hits = retrieval.get("hits", []) if isinstance(retrieval, dict) else retrieval
        # 1. sort
        sorted_hits = self.sort_hits(hits)
        # 2. dedupe
        deduped = self.deduplicate(sorted_hits)
        # 3. merge contiguous
        merged = self.merge_contiguous(deduped)
        # 4. final resort to respect priority
        merged_sorted = self.sort_hits(merged)
        # 5. apply token budget
        selected, used = self.apply_token_budget(merged_sorted, token_budget)
        # 6. build prompt string
        prompt = self.build_vietnamese_prompt(user_query, selected)
        return {
            "sys_prompt": SYS_INST,
            "user_prompt": prompt,
            "selected_hits": selected,
            "used_tokens": used,
            "token_budget": token_budget,
            "reserve_tokens": self.reserve_tokens
        }


# --------------------------
# Example / quick test
# --------------------------
if __name__ == "__main__":
    sample_retrieval = {
        "hits": [
            {
                "id": "15038e18-005d-5ad0-9fe4-ad5a9f1a33ef",
                "payload": {
                    "text": "Thực hiện Nghị quyết Trung ương 4 khóa XI, XII, XIII về tăng cường xây dựng, chỉnh đốn Đảng và hệ thống chính trị",
                    "source_file": "1-HopChiBoThang2-2026.pdf",
                    "page_number": 3,
                    "language": "vi",
                    "content_hash": "6ce05e79a98637c...",
                    "num_tokens": 462,
                    "chunk_version": "semantic_e5_v2",
                    "embedding_model": "intfloat/multilingual-e5-base",
                    "index_name": "internal_docs_v1",
                    "access_level": "internal",
                    "is_authoritative": True,
                    "source_priority": 1,
                    "created_at": "2026-02-13T03:51:59.704676Z",
                    "combined_score": 5.342692681761931
                },
                "vector_score": 0.8635467,
                "rerank_score": 7.277629852294922,
                "combined_score": 5.342692681761931
            }
        ],
        "timings_ms": {"embed_ms": 717, "qdrant_ms": 23, "total_ms": 741}
    }

    builder = PromptBuilder()
    out = builder.build_prompt_from_retrieval(sample_retrieval,
                                              "Cho tôi tóm tắt nội dung trên và nêu 3 ý chính.",
                                              token_budget=2048)
    print("===== PROMPT =====\n")
    print(out["user_prompt"])
    print("\n===== METADATA =====")
    print("used_tokens:", out["used_tokens"], "token_budget:", out["token_budget"], "reserve:", out["reserve_tokens"])
