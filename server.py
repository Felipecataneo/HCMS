import asyncio
import time
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from hcms.core import RAGCore
from hcms.agent_bridge import LlamaNotebookBridge # Bridge atualizada

app = FastAPI(title="HCMS v4.3 – Llama 1B Local")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurações de Inicialização
DSN = "dbname=hcms user=felipe"
MODEL_PATH = "train_model/hcms_personal_llm/checkpoint-500" # Pasta onde você salvou o modelo após o treino

core = RAGCore(DSN)
# O modelo é carregado na RAM/VRAM aqui
agent = LlamaNotebookBridge(core, model_path=MODEL_PATH)

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
    try:
        # 1. Resposta principal (usa RAG + LLM)
        response = agent.chat(request.message)
        
        # 2. Extração de fatos técnica ocorre em Background para não atrasar o usuário
        background_tasks.add_task(agent.analyze_and_remember, request.message)
        
        return {"reply": response}
    except Exception as e:
        print(f"Erro no endpoint de chat: {e}")
        return {"reply": "Erro interno no processamento da mensagem."}

@app.get("/memories", response_model=List[MemoryItem])
async def list_memories(limit: int = 50):
    rows = core.storage.fetch_all("""
        SELECT id, content, importance, access_count, is_permanent
        FROM memories
        ORDER BY is_permanent DESC, last_accessed DESC NULLS LAST
        LIMIT %s
    """, (limit,))
    return rows

@app.delete("/memories/{mem_id}")
async def delete_memory(mem_id: str):
    core.storage.execute("DELETE FROM coactivations WHERE id_a = %s OR id_b = %s", (mem_id, mem_id))
    core.storage.execute("DELETE FROM memories WHERE id = %s", (mem_id,))
    return {"status": "deleted"}

# MANTIDO: Lógica de Metabolismo (Decaimento)
async def periodic_maintenance():
    while True:
        await asyncio.sleep(3600) # 1 hora
        try:
            print("🧹 [Metabolismo] Executando limpeza de memórias fracas...")
            _run_decay_cleanup()
        except Exception as e:
            print(f"Erro no metabolismo: {e}")




@app.post("/memories/{mem_id}/toggle-permanent")
async def toggle_permanent(mem_id: str):
    """Toggle permanência de uma memória"""
    result = core.storage.fetch_all(
        "SELECT is_permanent FROM memories WHERE id = %s", 
        (mem_id,)
    )
    
    if not result:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    current = result[0]  # ← pega primeiro resultado
    new_state = not current['is_permanent']
    
    core.storage.execute(
        "UPDATE memories SET is_permanent = %s WHERE id = %s",
        (new_state, mem_id)
    )
    
    return {"status": "toggled", "is_permanent": new_state}

def _run_decay_cleanup():
    now = time.time()
    # Deleta o que já decaiu abaixo do limiar de 5% de ativação
    to_delete = core.storage.fetch_all("""
        SELECT id FROM memories
        WHERE is_permanent = FALSE
        AND EXP(-0.693 * ((%s - COALESCE(last_accessed, creation_time)) / 3600.0) / 
                  (24.0 * (1 + 10 * importance))) < 0.05
    """, (now,))
    
    if not to_delete: return
    ids = [r['id'] for r in to_delete]
    core.storage.execute("DELETE FROM coactivations WHERE id_a = ANY(%s) OR id_b = ANY(%s)", (ids, ids))
    core.storage.execute("DELETE FROM memories WHERE id = ANY(%s)", (ids,))

@app.on_event("startup")
async def on_startup():
    # Inicia a tarefa de manutenção em background
    asyncio.create_task(periodic_maintenance())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)