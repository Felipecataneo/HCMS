# HCMS — Hierarchical Compressed Memory System (v2)

**HCMS v2** é uma engine de memória autônoma projetada para agentes de IA. Ele supera implementações tradicionais de RAG ao combinar busca semântica, busca exata por texto e um sistema de "esquecimento cognitivo" que mantém o contexto do agente limpo e relevante.

## 🚀 O que há de novo na V2?

-   **Busca Híbrida + RRF:** Integração entre `pgvector` e `tsvector` usando *Reciprocal Rank Fusion* para precisão total em termos técnicos e semânticos.
-   **Cross-Encoder Reranking:** Reordenação de candidatos por um modelo de atenção profunda, eliminando alucinações de recuperação.
-   **Cognitive Pruning:** Sistema de Garbage Collection que deleta ruído e arquiva fatos estagnados automaticamente.
-   **Agent Bridge (Ollama):** Integração nativa com **Llama 3.2 (3B)** para extração automática de fatos e geração de respostas contextualizadas.
-   **Real-time Dashboard:** Interface Next.js para visualizar e gerenciar o substrato de memória do agente.

---

## 🛠️ Tech Stack

-   **Backend:** Python 3.12, FastAPI, PostgreSQL + pgvector, Ollama.
-   **Modelos:** 
    -   Embedding: `all-MiniLM-L6-v2`
    -   Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
    -   LLM: `Llama-3.2:3b` (via Ollama)
-   **Frontend:** Next.js 15, TypeScript, Tailwind CSS, Shadcn/UI, Lucide Icons.

---

## 🏗️ Estrutura do Projeto

```text
hcms/
├── core.py             # Engine: Hybrid Recall, RRF e 1-Hop Expansion
├── storage.py          # Camada de Persistência PostgreSQL
├── reranker.py         # Refinamento semântico profundo
├── agent_bridge.py     # Interface de Inteligência (Ollama)
├── pruner.py           # Metabolismo Cognitivo (Limpeza de ruído)
frontend/               # Next.js App
├── src/app/page.tsx    # Interface de Chat
└── src/components/     # Memory Dashboard & UI Components
```

---

## 💾 Configuração do Banco de Dados

O HCMS v2 exige suporte a busca textual exata. Execute no PostgreSQL:

```sql
-- 1. Suporte Vetorial e Textual
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS fts_tokens tsvector;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS importance FLOAT DEFAULT 1.0;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS access_count INTEGER DEFAULT 0;

-- 2. Índice GIN para Busca Híbrida
CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING GIN (fts_tokens);

-- 3. Trigger de Sincronização Automática
CREATE OR REPLACE FUNCTION memories_fts_trigger() RETURNS trigger AS $$
BEGIN
  new.fts_tokens := to_tsvector('simple', coalesce(new.content, ''));
  return new;
END
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_memories_fts_update
BEFORE INSERT OR UPDATE ON memories
FOR EACH ROW EXECUTE FUNCTION memories_fts_trigger();
```

---

## 🚦 Como Iniciar

### 1. Backend
```bash
# Instale as dependências
pip install fastapi uvicorn psycopg2-binary sentence-transformers requests zstandard

# Inicie o servidor
python server.py
```

### 2. IA Local
Certifique-se de que o Ollama está rodando:
```bash
ollama run llama3.2:3b
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🧠 Conceitos de Memória na V2

### 1. Recuperação em 4 Estágios
1.  **Hybrid Scan:** Busca vetorial (similaridade) + FTS (termos exatos).
2.  **RRF Fusion:** Combina os resultados priorizando documentos que aparecem em ambos os rankings.
3.  **Cross-Encoder:** Re-calcula a relevância real entre a query do usuário e o conteúdo dos top 20 candidatos.
4.  **1-Hop Injection:** Adiciona memórias relacionadas no grafo (edges) para dar contexto periférico ao agente.

### 2. Upsert Cognitivo
O sistema evita redundância. Se o usuário disser o mesmo fato várias vezes, o HCMS detecta a similaridade extrema e apenas atualiza o `last_access` da memória existente em vez de criar duplicatas.

### 3. Poda (Pruning)
O agente "esquece" informações inúteis. Memórias com baixa importância (< 0.3) e sem acessos frequentes são deletadas em ciclos de manutenção para garantir que o contexto não seja poluído por ruído conversacional.

---

## 🖥️ Interface de Controle
O Frontend inclui um **Memory Dashboard** lateral com um **Slider de Importância**. Isso permite:
-   Filtrar visualmente memórias irrelevantes.
-   Deletar manualmente alucinações ou erros de extração do LLM.
-   Visualizar a "Importância Cognitiva" atribuída pelo agente a cada fato.

