# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

from hcms.core import HCMS
from hcms.agent_bridge import HCMSAgentBridge

# Configuração
DSN = "dbname=hcms user=felipe"
hcms = HCMS(DSN)
agent = HCMSAgentBridge(hcms)

app = FastAPI(title="HCMS Agent API")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class MemoryItem(BaseModel):
    id: str
    content: str
    importance: float

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint principal de conversa"""
    try:
        response = agent.chat(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories", response_model=List[MemoryItem])
async def list_memories(limit: int = 10):
    """Retorna as memórias mais recentes para exibição no Dashboard"""
    rows = hcms.storage.fetch_all(
        "SELECT id, content, importance FROM memories ORDER BY creation_time DESC LIMIT %s",
        (limit,)
    )
    return rows

@app.delete("/memories/clear")
async def clear_database():
    """Limpa o lixo de testes (Use com cautela)"""
    hcms.storage.execute("TRUNCATE memories, edges, access_logs CASCADE;")
    return {"message": "Banco de dados resetado."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)