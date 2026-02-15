from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="Internal RAG Service")

retriever = Retriever()
builder = ContextBuilder()

class SearchRequest(BaseModel):
    query: str

@app.post("/retrieval")
def retrieval(req: SearchRequest):
    retrieved_results = retriever.search(req.query)
    return retrieved_results
@app.get("/health")
def health():
    return {"status": "ok"}

