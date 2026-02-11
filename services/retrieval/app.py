from fastapi import FastAPI
from pydantic import BaseModel
from retriever import Retriever
import os

app = FastAPI(title="Internal RAG Retrieval Service")

retriever = Retriever()

class SearchRequest(BaseModel):
    query: str

@app.post("/search")
def search(req: SearchRequest):
    results = retriever.search(req.query)
    return {
        "query": req.query,
        "results": results
    }

@app.get("/health")
def health():
    return {"status": "ok"}

