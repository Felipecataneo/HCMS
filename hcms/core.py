import time
import math
from typing import List, Dict, Set, Tuple
from collections import deque
from psycopg2.extras import Json
from sentence_transformers import SentenceTransformer, CrossEncoder

from .compression import ZstdBackend
from .storage import PostgresStorageProvider
from .pruner import MemoryPruner

# ============================================================
# RE-RANKER (PRECISÃO CIRÚRGICA)
# ============================================================
class MemoryReranker:
    """Refina os candidatos usando um modelo de atenção total (Cross-Encoder)"""
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Modelo otimizado para latência e precisão em reranking
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates: return []
        
        # Prepara pares (Query, Documento)
        pairs = [[query, c['content'] if c['content'] else ""] for c in candidates]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = float(score)

        # Ordena pelo score do Cross-Encoder (Soberano)
        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)


class HCMS:
    """
    Hierarchical Compressed Memory System v2 (Agent-Centric)
    Supera MemGPT/Mem0 via Recuperação Híbrida e Reranking de Borda.
    """

    def __init__(self, dsn: str):
        self.storage = PostgresStorageProvider(dsn)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.reranker = MemoryReranker()
        self.zstd = ZstdBackend()
        self.pruner = MemoryPruner(self)

        # Parâmetros de decaimento (Cognitive Decay)
        self.decay_lambda = 0.0000001
        self.weight_decay_lambda = 0.00001157

    def maintenance(self):
        """Job periódico de limpeza e otimização"""
        self.sync_access_stats() # Primeiro consolida acessos
        stats = self.pruner.run_garbage_collection()
        dupes = self.pruner.consolidate_duplicates()
        stats["duplicates_removed"] = dupes
        return stats

    # ============================================================
    # INGESTÃO (ESCRITA)
    # ============================================================

    def remember(self, content: str, importance: float = 1.0, metadata: dict = None, relations: list = None) -> str:
        """Armazena um fato e dispara o trigger de FTS no banco"""
        mem_id = f"mem_{int(time.time() * 1000)}"
        embedding = self.encoder.encode(content).tolist()
        now = time.time()

        self.storage.execute(
            """
            INSERT INTO memories
                (id, content, embedding, metadata, importance, last_access, creation_time)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (mem_id, content, embedding, Json(metadata or {}), importance, now, now),
        )

        if relations:
            for target_id, rel_type in relations:
                self.storage.execute(
                    "INSERT INTO edges (src, dst, type, weight, last_update) VALUES (%s, %s, %s, 1.0, %s)",
                    (mem_id, target_id, rel_type, now),
                )
        return mem_id

    # ============================================================
    # RECALL (O MOTOR SOTA)
    # ============================================================

    def recall(self, query_text: str, limit: int = 5, expand_graph: bool = True):
        """
        Recall de Alta Fidelidade:
        1. Hybrid Search (Vector + BM25)
        2. RRF Fusion (Blended Ranking)
        3. Cross-Encoder Rerank (Refinamento)
        4. Context Injection (1-Hop neighbors)
        """
        query_emb = self.encoder.encode(query_text).tolist()
        now = time.time()
        k_rrf = 60 # Constante padrão para fusão RRF

        # --- STAGE 1: HYBRID SEARCH (CANDIDATOS) ---
        # Busca Vetorial (Semântica)
        vector_candidates = self.storage.fetch_all("""
            SELECT id, content, compressed_data, compression_type, metadata, last_access, importance,
                   (1 - (embedding <=> %s::vector)) AS semantic_sim
            FROM memories 
            WHERE tier < 3
            ORDER BY embedding <=> %s::vector LIMIT 40
        """, (query_emb, query_emb))

        # Busca Full-Text (Exatidão de termos)
        fts_candidates = self.storage.fetch_all("""
            SELECT id, content, compressed_data, compression_type, metadata, last_access, importance,
                   ts_rank_cd(fts_tokens, websearch_to_tsquery('simple', %s)) AS fts_rank
            FROM memories 
            WHERE fts_tokens @@ websearch_to_tsquery('simple', %s) AND tier < 3
            ORDER BY fts_rank DESC LIMIT 40
        """, (query_text, query_text))

        # --- STAGE 2: RRF FUSION ---
        # Combina os rankings sem precisar normalizar scores diferentes
        rrf_scores = {}
        candidate_map = {}

        for rank, res in enumerate(vector_candidates, 1):
            rrf_scores[res['id']] = rrf_scores.get(res['id'], 0) + (1.0 / (k_rrf + rank))
            candidate_map[res['id']] = res

        for rank, res in enumerate(fts_candidates, 1):
            rrf_scores[res['id']] = rrf_scores.get(res['id'], 0) + (1.0 / (k_rrf + rank))
            if res['id'] not in candidate_map:
                candidate_map[res['id']] = res

        # Seleciona os melhores após fusão para o reranking
        fused_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:20]
        candidates = []
        for cid in fused_ids:
            c = candidate_map[cid]
            # Descomprime se necessário antes do Reranker
            if c['content'] is None and c['compression_type'] == 'zstd':
                c['content'] = self.zstd.decompress(c['compressed_data']).decode('utf-8')
            candidates.append(c)

        # --- STAGE 3: CROSS-ENCODER RERANK (REFINAMENTO) ---
        refined_results = self.reranker.rerank(query_text, candidates)
        top_results = refined_results[:limit]

        # --- STAGE 4: CONTEXT INJECTION (1-HOP) ---
        if expand_graph:
            top_results = self._inject_context(top_results)

        # Log de acesso para manutenção de importância/recência
        self._log_access_batch([(r['id'], now) for r in top_results])

        return top_results

    def _inject_context(self, results: List[Dict]) -> List[Dict]:
        """Injeta vizinhos imediatos para expandir a 'linha de pensamento' do Agente"""
        if not results: return []
        ids = [r['id'] for r in results]
        
        neighbors = self.storage.fetch_all("""
            SELECT e.src, m.id, m.content, m.metadata 
            FROM edges e 
            JOIN memories m ON e.dst = m.id
            WHERE e.src = ANY(%s) AND m.tier < 3
            LIMIT 20
        """, (ids,))

        for r in results:
            r['context_edges'] = [
                {'id': n['id'], 'content': n['content'][:200]} 
                for n in neighbors if n['src'] == r['id']
            ]
        return results

    # ============================================================
    # GRAPH & MAINTENANCE (LEAN VERSION)
    # ============================================================

    def _log_access_batch(self, logs: List[tuple]):
        if not logs: return
        with self.storage._get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany("INSERT INTO access_logs (mem_id, access_time) VALUES (%s, %s)", logs)
            conn.commit()

    def sync_access_stats(self):
        """Atualiza estatísticas de acesso de forma atômica"""
        with self.storage._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    WITH agg AS (SELECT mem_id, MAX(access_time) AS latest, COUNT(*) AS cnt FROM access_logs GROUP BY mem_id)
                    UPDATE memories m SET last_access = GREATEST(m.last_access, agg.latest) FROM agg WHERE m.id = agg.mem_id
                """)
                cur.execute("TRUNCATE TABLE access_logs")
            conn.commit()

    def _update_sequential_weights(self, seed_id: str, results: List[dict], now: float):
        """Fortalece arestas entre memórias acessadas em conjunto (Co-ocorrência)"""
        if not results or not seed_id: return
        result_ids = [r["id"] for r in results]
        pairs = [(seed_id, result_ids[0], 0.1, now)]
        for i in range(len(result_ids) - 1):
            pairs.append((result_ids[i], result_ids[i+1], 0.1, now))

        with self.storage._get_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany("""
                    INSERT INTO edges (src, dst, type, weight, last_update) VALUES (%s, %s, 'co-occurrence', %s, %s)
                    ON CONFLICT (src, dst, type) DO UPDATE SET 
                        weight = edges.weight + EXCLUDED.weight, last_update = EXCLUDED.last_update
                """, pairs)
            conn.commit()