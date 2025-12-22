# HCMS — Hierarchical Compressed Memory System (V2)

**HCMS** é um substrato de memória de alta performance para agentes de IA. Ao contrário de RAGs tradicionais, o HCMS combina busca vetorial (semântica) com busca textual (BM25) e refinamento via Cross-Encoders para garantir que fatos críticos nunca sejam perdidos, independentemente do volume de dados.

## Arquitetura de Recuperação (SOTA 2025)

O sistema utiliza uma pipeline de 4 estágios para superar limitações de relevância:
1.  **Hybrid Search:** Scan simultâneo em `pgvector` e `tsvector` (PostgreSQL).
2.  **RRF Fusion:** Fusão de rankings (Reciprocal Rank Fusion) para unificar resultados semânticos e exatos.
3.  **Cross-Encoder Reranking:** Reordenação dos top candidatos através de atenção profunda (Acurácia > 90%).
4.  **1-Hop Context Injection:** Injeção automática de vizinhos relacionais para expandir a linha de raciocínio do agente.

---

## Estrutura do Projeto

```text
hcms/
├── core.py             # Engine principal (Hybrid Recall & RRF)
├── storage.py          # Interface PostgreSQL + FTS Support
├── reranker.py         # Refinamento via Cross-Encoder (Sentence Transformers)
├── compression.py      # Backend de compressão Zstd para documentos frios
├── tier_management.py  # Gestão de ciclo de vida (Hot/Cold/Archive)
└── scripts/
    └── test_hcms.py    # Suite de testes de integração V2
```

---

## Configuração Obrigatória do PostgreSQL

O HCMS exige extensões e triggers específicos para operar a busca híbrida. Execute os comandos abaixo no seu banco de dados antes de iniciar o sistema:

### 1. Extensões e Schema
```sql
-- Habilita suporte a vetores
CREATE EXTENSION IF NOT EXISTS vector;

-- Adiciona suporte a Busca Híbrida e Importância
ALTER TABLE memories ADD COLUMN IF NOT EXISTS fts_tokens tsvector;
ALTER TABLE memories ADD COLUMN IF NOT EXISTS importance FLOAT DEFAULT 1.0;

-- Índice GIN para performance em buscas textuais exatas
CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories USING GIN (fts_tokens);
```

### 2. Sincronização Automática (Trigger)
O HCMS automatiza a tokenização de texto diretamente no banco para garantir consistência entre o que é lido e o que é indexado:

```sql
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

## Instalação

```bash
pip install psycopg2-binary sentence-transformers numpy zstandard
```

---

## Como Usar

### Ingestão de Conhecimento
O sistema aceita importância manual e relações de grafo:
```python
from hcms.core import HCMS

hcms = HCMS("dbname=hcms user=seu_usuario")

# Fato isolado
hcms.remember("O servidor de produção utiliza a porta 9999.", importance=1.0)

# Fato relacionado (Grafo)
hcms.remember("A porta 9999 deve ser aberta no firewall.", relations=[("mem_id_anterior", "config")])
```

### Recuperação (Recall)
A busca híbrida lida automaticamente com termos exatos e semânticos:
```python
# O Recall executará Hybrid Search + RRF + Rerank + Context Injection
results = hcms.recall("qual a porta do servidor?", limit=3)

for r in results:
    print(f"Content: {r['content']}")
    print(f"Contexto Relacionado: {r['context_edges']}")
```

---

## Diferenciais Técnicos

| Recurso | HCMS V2 | RAG Comum |
| :--- | :--- | :--- |
| **Busca por IDs/Códigos** | Nativa (via BM25/FTS) | Falha (Alucinação Vetorial) |
| **Relação entre Fatos** | Automática (1-Hop Injection) | Inexistente (Fragmentada) |
| **Refinamento** | Cross-Encoder (Deep Match) | Distância de Cosseno Simples |
| **Storage** | Otimizado via Tiers & Zstd | Redundante e Pesado |

