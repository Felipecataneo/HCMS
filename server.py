# server.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from hcms.core import HCMSOptimized
from hcms.background_gc import create_background_gc
from hcms.agent_bridge import HCMSAgentBridge

# =============================================================================
# 1. Inicialização do App
# =============================================================================
app = FastAPI(title="HCMS Agent API v2.1")

# =============================================================================
# 2. CORS
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 3. Inicialização do HCMS
# =============================================================================
DSN = "dbname=hcms user=felipe"

# Motor principal (Auto-GC desligado → GC via Worker)
hcms = HCMSOptimized(DSN, enable_auto_gc=False)

# Bridge do agente
agent = HCMSAgentBridge(hcms)

# -----------------------------------------------------------------------------
# Background GC — Metabolismo Cognitivo Assíncrono
# -----------------------------------------------------------------------------
gc_worker = create_background_gc(
    hcms,
    mode="simple",
    interval=1800  # 30 minutos (v2.1 benchmarked)
)

# =============================================================================
# 4. Ciclo de Vida
# =============================================================================
@app.on_event("startup")
async def startup_event():
    gc_worker.start()
    print("🧠 HCMS v2.1: Sistema de Memória e Background GC ativos.")

@app.on_event("shutdown")
async def shutdown_event():
    gc_worker.stop()
    print("🧠 HCMS v2.1: Background GC finalizado com segurança.")

# =============================================================================
# 5. Modelos de Dados
# =============================================================================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

class MemoryItem(BaseModel):
    id: str
    content: str
    importance: float

# =============================================================================
# 6. Endpoints
# =============================================================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = agent.chat(request.message)
        return {"reply": response}
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        raise HTTPException(status_code=500, detail="Erro interno no chat")

@app.get("/memories", response_model=List[MemoryItem])
async def list_memories(limit: int = 20):
    return hcms.storage.fetch_all(
        """
        SELECT id, content, importance
        FROM memories
        ORDER BY creation_time DESC
        LIMIT %s
        """,
        (limit,)
    )

@app.delete("/memories/{mem_id}")
async def delete_memory(mem_id: str):
    hcms.storage.execute(
        "DELETE FROM memories WHERE id = %s",
        (mem_id,)
    )
    return {"status": "deleted"}

@app.delete("/memories/clear/all")
async def clear_database():
    hcms.storage.execute(
        "TRUNCATE memories, edges, access_logs CASCADE;"
    )
    return {"message": "Banco de dados resetado"}

# =============================================================================
# 7. Entry Point
# =============================================================================
if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
