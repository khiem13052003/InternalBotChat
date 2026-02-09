import os
import spacy
import hashlib
import uuid
import numpy as np
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer, util
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText, Title, ListItem

dir = r"./storage/data"
#dir = r"D:\DaiHoc\folderrr"

class PdfChunks:
    CHUNK_VERSION = "semantic_e5_v2"

    def __init__(
        self,
        model_name="intfloat/multilingual-e5-base",
        similarity_threshold=0.85,
        max_chars=900,
        overlap_sentences=2,
        min_overlap_sim=0.6,
        lang="vi",
        vectorDB_url="http://qdrant:6333",
        vectorDB_collection_name="internal_docs",
        vectorDB_size=768,
        upsert_batch_size=128,
    ):
        self.filter_elements = (NarrativeText, Title, ListItem)
        self.embedding_model = SentenceTransformer(model_name, cache_folder="./models/embedding")

        self.similarity_threshold = similarity_threshold
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences
        self.min_overlap_sim = min_overlap_sim

        self.nlp = spacy.blank(lang)
        self.nlp.add_pipe("sentencizer")

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
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


    def _deterministic_chunk_id(self, document_id, chunk_index, content_hash):
        # raw string to base the UUID on
        raw = f"{document_id}-{chunk_index}-{content_hash}"
        # create a deterministic UUID (v5) using a namespace (NAMESPACE_URL chosen arbitrarily)
        u = uuid.uuid5(uuid.NAMESPACE_URL, raw)
        return str(u)  # returns canonical UUID string

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
        self.client.upsert(collection_name=self.collection, points=points)

    def _flush_current(self, chunks, document_id, current_text, current_emb, chunk_index):
        # finalize current chunk (assumes current_emb is numpy array already normalized)
        content_hash = self._hash(current_text)
        chunk_id = self._deterministic_chunk_id(document_id, chunk_index, content_hash)
        chunks.append({
            "chunk_id": chunk_id,
            "content": current_text,
            "embedding": current_emb.tolist(),
            "metadata": {
                "source_file": document_id + ".pdf",
                "document_id": document_id,
                "chunk_index": chunk_index,
                "language": "vi",
                "chunk_version": self.CHUNK_VERSION,
                "created_at": datetime.now().isoformat() + "Z",
                "content_hash": content_hash,
            }
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
                emb = self.embedding_model.encode(text, normalize_embeddings=True)
                emb = np.array(emb, dtype=np.float32)

                # Optional: force new chunk on Title elements
                if isinstance(e, Title) and current_text:
                    # flush current chunk
                    self._flush_current(chunks, document_id, current_text, current_emb, chunk_index)
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

                # compute similarity between current_emb and new emb
                # util.cos_sim returns torch tensor; convert to float
                sim = util.cos_sim(current_emb, emb).item()

                if (sim >= self.similarity_threshold and len(current_text) + len(text) <= self.max_chars):
                    # incremental mean: update current_emb and emb_count
                    # current_emb = (current_emb * emb_count + emb) / (emb_count + 1)
                    # but re-normalize after averaging
                    summed = current_emb * emb_count + emb
                    emb_count += 1
                    current_emb = summed / emb_count
                    # normalize
                    norm = np.linalg.norm(current_emb)
                    if norm > 0:
                        current_emb = current_emb / norm
                    current_text += "\n" + text
                else:
                    # flush old chunk
                    self._flush_current(chunks, document_id, current_text, current_emb, chunk_index)
                    chunk_index += 1
                    # build new chunk, include overlap if present
                    overlap = self._sentence_overlap(current_text)
                    if overlap:
                        current_text = overlap + "\n" + text
                        # recompute emb for overlap + new text: simple approach = mean(emb_overlap, emb)
                        overlap_emb = self.embedding_model.encode(overlap, normalize_embeddings=True)
                        overlap_emb = np.array(overlap_emb, dtype=np.float32)
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

            # flush final current chunk
            if current_text:
                self._flush_current(chunks, document_id, current_text, current_emb, chunk_index)

            # final upsert for document
            if chunks:
                print(f"Upserting remaining {len(chunks)} chunks ...")
                self.upsert_chunks(chunks)
            print("Done!")

if __name__ == "__main__":
    app = PdfChunks()
    app.generation(dir)

