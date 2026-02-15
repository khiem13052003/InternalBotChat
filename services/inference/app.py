import os
from typing import List, Dict, Any, Optional, Callable, Tuple
from fastapi import FastAPI
from pydantic import BaseModel
from prompt_template import PromptBuilder
from llm_client import MLCClient

app = FastAPI(title="Internal RAG Inference Service")

builder = PromptBuilder()
llmClient = MLCClient()

class SearchRequest(BaseModel):
    retrieved_data_json : Dict[str, Any]
    query: str
    token_budget: int = 2048


@app.post("/inference")
def inference(req: SearchRequest):
    prompt_result = builder.build_prompt_from_retrieval(req.retrieved_data_json, req.query, req.token_budget)
    messages = [
        {"role": "system", "content": prompt_result["sys_prompt"]},
        {"role": "user", "content": prompt_result["user_prompt"]},
    ]
    #return {"result":messages}
    result = llmClient.generate(messages, temperature=0.2)
    return {"result":result}
@app.get("/health")
def health():
    return {"status": "ok"}
