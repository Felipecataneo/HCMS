import asyncio
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from hcms.core import RAGCore
from hcms.agent_bridge import HCMSAgentBridge

app = FastAPI(title="HCMS v4.2 Crystallized Memory")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DSN = "dbname=hcms user=felipe"
core = RAGCore(DSN)
agent = HCMSAgentBridge(core)
_message_counter = 0

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

class MemoryItem(BaseModel):
    id: str
    content: str
    importance: float
    access_count: int
    is_permanent: bool

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    global _message_counter
    try:
        response = agent.chat(request.message)
        # Extração em background para não travar o usuário
        background_tasks.add_task(agent.analyze_and_remember, request.message)
        
        _message_counter += 1
        if _message_counter % 50 == 0:
            background_tasks.add_task(_light_garbage_collection)
            
        return {"reply": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories", response_model=List[MemoryItem])
async def list_memories(limit: int = 50):
    rows = core.storage.fetch_all("""
        SELECT id, content, importance, access_count, is_permanent
        FROM memories
        ORDER BY is_permanent DESC, last_accessed DESC NULLS LAST
        LIMIT %s
    """, (limit,))
    return rows

@app.post("/memories/{mem_id}/toggle-permanent")
async def toggle_permanent(mem_id: str):
    core.storage.execute("""
        UPDATE memories SET is_permanent = NOT is_permanent WHERE id = %s
    """, (mem_id,))
    return {"status": "updated"}

@app.delete("/memories/{mem_id}")
async def delete_memory(mem_id: str):
    core.storage.execute("DELETE FROM coactivations WHERE id_a = %s OR id_b = %s", (mem_id, mem_id))
    core.storage.execute("DELETE FROM memories WHERE id = %s", (mem_id,))
    return {"status": "deleted"}

# METABOLISMO
async def periodic_maintenance():
    while True:
        await asyncio.sleep(3600)
        try:
            print("🧹 [Metabolismo] Iniciando limpeza...")
            _run_decay_cleanup()
        except Exception as e:
            print(f"Erro: {e}")

def _run_decay_cleanup():
    now = time.time()
    to_delete = core.storage.fetch_all("""
        SELECT id FROM memories
        WHERE is_permanent = FALSE
        AND EXP(-0.693 * ((%s - COALESCE(last_accessed, creation_time)) / 3600.0) / 
                  (24.0 * (1 + 10 * importance))) < 0.05
    """, (now, now))
    if not to_delete: return
    ids = [r['id'] for r in to_delete]
    core.storage.execute("DELETE FROM coactivations WHERE id_a = ANY(%s) OR id_b = ANY(%s)", (ids, ids))
    core.storage.execute("DELETE FROM memories WHERE id = ANY(%s)", (ids,))

def _light_garbage_collection():
    now = time.time()
    core.storage.execute("""
        DELETE FROM memories 
        WHERE is_permanent = FALSE 
        AND access_count = 0 AND importance < 0.2 AND creation_time < %s
    """, (now - 3600,))

@app.on_event("startup")
async def on_startup():
    asyncio.create_task(periodic_maintenance())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)