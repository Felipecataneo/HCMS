# server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from hcms.core import HCMS
from hcms.agent_bridge import HCMSAgentBridge

# 1. Inicialização ÚNICA do App
app = FastAPI(title="HCMS Agent API v2")

# 2. Configuração do CORS (DEVE ser logo após a criação do app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Configuração do HCMS
DSN = "dbname=hcms user=felipe"
hcms = HCMS(DSN)
agent = HCMSAgentBridge(hcms)

# 4. Modelos de Dados
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str  # Ajustado para 'reply' para bater com o Next.js

class MemoryItem(BaseModel):
    id: str
    content: str
    importance: float

# 5. Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Endpoint principal de conversa"""
    try:
        response = agent.chat(request.message)
        # Retornamos 'reply' para o frontend ler corretamente
        return {"reply": response}
    except Exception as e:
        print(f"Erro no chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories", response_model=List[MemoryItem])
async def list_memories(limit: int = 20):
    """Retorna as memórias mais recentes para o Dashboard"""
    rows = hcms.storage.fetch_all(
        "SELECT id, content, importance FROM memories ORDER BY creation_time DESC LIMIT %s",
        (limit,)
    )
    return rows

@app.delete("/memories/{mem_id}")
async def delete_memory(mem_id: str):
    """Deleta uma memória específica via UI"""
    hcms.storage.execute("DELETE FROM memories WHERE id = %s", (mem_id,))
    return {"status": "deleted"}

@app.delete("/memories/clear/all")
async def clear_database():
    """Limpa o banco de dados (CUIDADO)"""
    hcms.storage.execute("TRUNCATE memories, edges, access_logs CASCADE;")
    return {"message": "Banco de dados resetado."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)