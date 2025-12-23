from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from hcms.core import RAGCore
from hcms.agent_bridge import HCMSAgentBridge

app = FastAPI(title="HCMS v3 Cognitive API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Certifique-se de que o DSN está correto para seu ambiente
DSN = "dbname=hcms user=felipe"
core = RAGCore(DSN)
agent = HCMSAgentBridge(core)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

class MemoryItem(BaseModel):
    id: str
    content: str
    importance: float
    access_count: int

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # O AgentBridge aciona o core.recall que agora gerencia o campo de ativação
        response = agent.chat(request.message)
        return {"reply": response}
    except Exception as e:
        print(f"Erro detalhado: {e}")
        raise HTTPException(status_code=500, detail=f"Erro no Agente: {str(e)}")

@app.get("/memories", response_model=List[MemoryItem])
async def list_memories(limit: int = 50):
    # ATUALIZADO: Agora busca da coluna real 'importance' e inclui 'access_count'
    rows = core.storage.fetch_all(
        """
        SELECT id, content, importance, access_count
        FROM memories
        ORDER BY last_accessed DESC NULLS LAST, creation_time DESC
        LIMIT %s
        """,
        (limit,)
    )
    return rows

@app.delete("/memories/{mem_id}")
async def delete_memory(mem_id: str):
    core.storage.execute("DELETE FROM memories WHERE id = %s", (mem_id,))
    return {"status": "deleted"}

@app.post("/maintenance")
async def run_maintenance():
    # O metabolismo agora é focado em limpeza de baixo acesso e importância
    deleted = core.storage.fetch_all("""
        DELETE FROM memories 
        WHERE importance < 0.3 AND access_count < 2
        RETURNING id
    """)
    return {"status": "success", "purged_count": len(deleted)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)