import os
import spacy
import hashlib
import uuid
import numpy as np
from datetime import datetime, timezone
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText, Title, ListItem
import torch
from transformers import AutoTokenizer

dir = r"./storage/data"
#dir = r"D:\DaiHoc\folderrr"

class PdfChunks:
    CHUNK_VERSION = "semantic_e5_v2"

    def __init__(
        self,
        llm_tokenizor_file_path="./models/llm/Llama-3.2-1B-q4f16_1-MLC",
        embedding_model_path="./models/embedding/multilingual-e5-base",
        similarity_threshold=0.85,
        max_chars=900,
        overlap_sentences=2,
        min_overlap_sim=0.6,
        lang="vi",
        vectorDB_url="http://192.168.1.107:6333",
        vectorDB_collection_name="internal_docs",
        vectorDB_size=768,
        upsert_batch_size=20,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.filter_elements = (NarrativeText, Title, ListItem)
        # embedding model
        self.embedding_model = SentenceTransformer(embedding_model_path,device=self.device)

        self.similarity_threshold = similarity_threshold
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences
        self.min_overlap_sim = min_overlap_sim

        self.nlp = spacy.blank(lang)
        self.nlp.add_pipe("sentencizer")

        self.tokenizer = AutoTokenizer.from_pretrained(llm_tokenizor_file_path)

        self.collection = vectorDB_collection_name
        self.client = QdrantClient(url=vectorDB_url)
        self.upsert_batch_size = upsert_batch_size

        if self.collection not in [c.name for c in self.client.get_collections().collections]:
            print(f"Not found collection named {self.collection}, creating new database...")
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vectorDB_size, distance=Distance.COSINE),
            )
            print("OK!")

    def _sentence_overlap(self, text):
        sents = [s.text.strip() for s in self.nlp(text).sents if s.text.strip()]
        return "\n".join(sents[-self.overlap_sentences:])

    def _hash(self, text):
        # use normalized text for stable hashing
        normalized = " ".join(text.strip().split()).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _deterministic_chunk_id(self, document_id, chunk_index, content_hash):
        raw = f"{document_id}-{chunk_index}-{content_hash}"
        u = uuid.uuid5(uuid.NAMESPACE_URL, raw)
        return str(u)

    def upsert_chunks(self, chunks):
        if not chunks:
            return
        points = [
            PointStruct(
                id=c["chunk_id"],
                vector=c["embedding"],
                payload={"text": c["content"], **c["metadata"]},
            )
            for c in chunks
        ]
        # consider adding retry/backoff in production
        self.client.upsert(collection_name=self.collection, points=points)

    def _flush_current(self, chunks, filename, document_id, current_text, current_emb, chunk_index, page_number=None):
        # finalize current chunk (assumes current_emb is numpy array already normalized)
        content_hash = self._hash(current_text)
        chunk_id = self._deterministic_chunk_id(document_id, chunk_index, content_hash)
        meta = {
            # Identity
            "document_id": document_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,

            # Source
            "source_file": filename,
            "source_type": "None",
            "page_number": page_number,

            # Content
            "language": "vi",
            "content_hash": content_hash,
            "num_tokens": len(self.tokenizer.encode(current_text, out_type=int, add_bos=False, add_eos=False)),

            # Versioning
            "chunk_version": self.CHUNK_VERSION,
            "embedding_model": "intfloat/multilingual-e5-base",
            "embedding_version": "v1.0",
            "index_name": "internal_docs_v1",

            # Security / Access control
            "access_level": "internal",
            "departments": ["it", "security"],
            "roles": ["employee"],

            # Temporal
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),

            # Trust & ranking signals
            "is_authoritative": True,
            "source_priority": 1,

            # Ops
            # "ingestion_job_id": job_id,
            "ingested_by": "ingestion_service",
        }

        if page_number is not None:
            meta["page_number"] = page_number

        chunks.append({
            "chunk_id": chunk_id,
            "content": current_text,
            "embedding": current_emb.tolist(),
            "metadata": meta
        })

    def generation(self, folder_dir):
        for filename in os.listdir(folder_dir):
            if not filename.lower().endswith(".pdf"):
                continue
            print(f"---Processing {filename}---")
            file_path = os.path.join(folder_dir, filename)
            document_id = os.path.splitext(filename)[0]

            elements = partition_pdf(
                filename=file_path,
                languages=["vie"],
                strategy="hi_res",
                infer_table_structure=True
            )

            chunks = []
            current_text = ""
            chunk_index = 0
            current_emb = None
            emb_count = 0

            for e in elements:
                if not isinstance(e, self.filter_elements):
                    continue
                if not e.text or len(e.text.strip()) < 30:
                    continue

                # batch flush to Qdrant to keep RAM bounded
                if len(chunks) >= self.upsert_batch_size:
                    print(f"Upserting {len(chunks)} chunks ...")
                    self.upsert_chunks(chunks)
                    chunks = []

                text = e.text.strip()

                # create embedding with passage: prefix
                passage_input = f"passage: {text}"
                emb = self.embedding_model.encode(passage_input, normalize_embeddings=True)
                emb = np.array(emb, dtype=np.float32)

                # Optional: force new chunk on Title elements
                if isinstance(e, Title) and current_text:
                    # flush current chunk
                    page_num = getattr(e.metadata, "page_number", None) if hasattr(e, "metadata") else None
                    self._flush_current(chunks, filename, document_id, current_text, current_emb, chunk_index, page_num)
                    chunk_index += 1
                    # start new chunk with title text
                    current_text = text
                    current_emb = emb
                    emb_count = 1
                    continue

                if not current_text:
                    current_text = text
                    current_emb = emb
                    emb_count = 1
                    continue

                # compute similarity between current_emb and new emb (both numpy arrays, normalized)
                sim = float(np.dot(current_emb, emb))
                if (sim >= self.similarity_threshold and len(current_text) + len(text) <= self.max_chars):
                    # incremental mean: update current_emb and emb_count, then renormalize
                    summed = current_emb * emb_count + emb
                    emb_count += 1
                    current_emb = summed / emb_count
                    # normalize
                    norm = np.linalg.norm(current_emb)
                    if norm > 0:
                        current_emb = current_emb / norm
                    current_text += "\n" + text
                else:
                    # flush old chunk (include page number if available)
                    page_num = getattr(e.metadata, "page_number", None) if hasattr(e, "metadata") else None
                    self._flush_current(chunks, filename, document_id, current_text, current_emb, chunk_index, page_num)
                    chunk_index += 1
                    # build new chunk, include overlap if present
                    overlap = self._sentence_overlap(current_text)
                    if overlap:
                        # prepare overlap input with passage: prefix
                        overlap_input = f"passage: {overlap}"
                        overlap_emb = self.embedding_model.encode(overlap_input, normalize_embeddings=True)
                        overlap_emb = np.array(overlap_emb, dtype=np.float32)
                        # compute overlap similarity to new emb
                        overlap_sim = float(np.dot(overlap_emb, emb))
                        if overlap_sim >= self.min_overlap_sim:
                            current_text = overlap + "\n" + text
                            summed = overlap_emb + emb
                            current_emb = summed / 2.0
                            norm = np.linalg.norm(current_emb)
                            if norm > 0:
                                current_emb = current_emb / norm
                            emb_count = 2
                        else:
                            current_text = text
                            current_emb = emb
                            emb_count = 1
                    else:
                        current_text = text
                        current_emb = emb
                        emb_count = 1

            # flush final current chunk
            if current_text:
                # no page number for final chunk (could be last element's page if desired)
                self._flush_current(chunks, filename, document_id, current_text, current_emb, chunk_index)

            # final upsert for document
            if chunks:
                print(f"Upserting remaining {len(chunks)} chunks ...")
                self.upsert_chunks(chunks)
            print("Done!")

if __name__ == "__main__":
    app = PdfChunks()
    app.generation(dir)

