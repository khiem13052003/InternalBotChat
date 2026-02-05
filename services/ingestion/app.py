import os
import spacy
from sentence_transformers import SentenceTransformer, util
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText, Title, ListItem

dir = r"./storage/data"
#dir = r"D:\DaiHoc\folderrr"

class PdfChunks:
    def __init__(
        self,
        model_name="intfloat/multilingual-e5-base",
        similarity_threshold=0.75,
        max_chars=900,
        overlap_sentences=2,      # ⬅ số câu overlap
        min_overlap_sim=0.6,
        lang="vi",                # vi / en
    ):
        self.filter_elements = (NarrativeText, Title, ListItem)
        self.embedding_model = SentenceTransformer(model_name, cache_folder="./models/embedding")

        self.similarity_threshold = similarity_threshold
        self.max_chars = max_chars
        self.overlap_sentences = overlap_sentences
        self.min_overlap_sim = min_overlap_sim

        # spaCy sentence splitter (nhẹ)
        self.nlp = spacy.blank(lang)
        self.nlp.add_pipe("sentencizer")

    def _get_sentence_overlap(self, text):
        doc = self.nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        if not sents:
            return ""
        return "\n".join(sents[-self.overlap_sentences :])

    def generation(self, folder_dir):
        print("-----------------------generation---------------------")
        for filename in os.listdir(folder_dir):
            file_path = os.path.join(folder_dir, filename)

            if not (os.path.isfile(file_path) and filename.lower().endswith(".pdf")):
                continue

            print(f"\n=== Processing: {filename} ===")

            elements = partition_pdf(
                filename=file_path,
                languages=["vie"],
                strategy="hi_res",
                infer_table_structure=True,
            )

            chunks = []
            current_chunk = None
            current_emb = None

            for e in elements:
                if not isinstance(e, self.filter_elements):
                    continue
                if not e.text or len(e.text.strip()) < 30:
                    continue

                text = e.text.strip()
                emb = self.embedding_model.encode(text, normalize_embeddings=True)

                # First element
                if current_chunk is None:
                    current_chunk = text
                    current_emb = emb
                    continue

                # Title → force new chunk
                if e.category == "Title":
                    chunks.append(current_chunk)
                    current_chunk = text
                    current_emb = emb
                    continue

                sim = util.cos_sim(current_emb, emb).item()

                if (
                    sim >= self.similarity_threshold
                    and len(current_chunk) + len(text) <= self.max_chars
                ):
                    current_chunk += "\n" + text
                    current_emb = (current_emb + emb) / 2
                else:
                    chunks.append(current_chunk)

                    overlap_text = self._get_sentence_overlap(current_chunk)
                    if overlap_text:
                        overlap_emb = self.embedding_model.encode(
                            overlap_text, normalize_embeddings=True
                        )
                        overlap_sim = util.cos_sim(overlap_emb, emb).item()
                    else:
                        overlap_sim = 0

                    if overlap_sim >= self.min_overlap_sim:
                        current_chunk = overlap_text + "\n" + text
                        current_emb = (overlap_emb + emb) / 2
                    else:
                        current_chunk = text
                        current_emb = emb

            # flush last chunk
            if current_chunk:
                chunks.append(current_chunk)

            print(f"✅ {len(chunks)} chunks created")
            for c in chunks:
                print("_______________")
                print(c)


pdfc = PdfChunks()
pdfc.generation(dir)

