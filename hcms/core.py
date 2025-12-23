import time
import math
import re
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from .storage import PostgresStorageProvider

class RAGCore:
    def __init__(self, dsn: str):
        self.storage = PostgresStorageProvider(dsn)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.activation_levels = {} 
        self.last_activation_update = {} 
        self.short_term_context = [] 

    def _is_technical_query(self, query: str) -> bool:
        """Detecta códigos, IDs, UUIDs ou termos com caracteres especiais/números"""
        # Padrão para códigos como XK-9847, IDs alfanuméricos longos, etc.
        return bool(re.search(r'[A-Z0-9]{3,}-\d+|[a-f0-9]{8}-|\b[A-Z]{2,}\d{2,}\b', query))

    def remember(self, content: str, importance: float = 0.5, metadata: dict = None):
        emb = self.encoder.encode(content).tolist()
        mem_id = f"mem_{int(time.time() * 1000)}"
        self.storage.upsert_memory(mem_id, content, emb, metadata, importance)
        return mem_id

    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        now = time.time()
        query_emb = self.encoder.encode(query).tolist()
        is_tech = self._is_technical_query(query)

        # 1. BUSCA HÍBRIDA DE PRECISÃO (SQL)
        # Unificamos tudo em uma query para reduzir latência de rede (I/O)
        candidates = self._get_hybrid_candidates(query, query_emb, is_tech, limit=30)
        if not candidates: return []

        candidate_ids = [c['id'] for c in candidates]
        coactive_map = self._get_coactivation_map(candidate_ids)
        
        # 2. RANKING CONTEXTUAL
        for c in candidates:
            # Scores base vindos do DB
            s_hybrid = float(c['hybrid_score'])
            # Se for match literal em query técnica, boost absoluto
            s_literal = 1.0 if (is_tech and c['is_literal_match']) else 0.0
            
            # Contexto Temporal/Ativação
            s_act = self._get_current_activation(c['id'], now)
            s_coact = float(coactive_map.get(c['id'], 0.0))
            
            # Score Final: Prioridade para Literal Match > Hybrid > Contexto
            c['final_score'] = (s_literal * 0.6) + (s_hybrid * 0.25) + (s_act * 0.1) + (s_coact * 0.05)

        results = sorted(candidates, key=lambda x: x['final_score'], reverse=True)[:limit]
        
        # 3. ATUALIZAÇÃO DE ESTADO (Para a próxima query)
        self._update_system_state(results, now)
        
        return results

    def _get_hybrid_candidates(self, query, query_emb, is_tech, limit):
        # Para termos técnicos, usamos 'simple' para evitar o stemming do português
        ts_cfg = 'simple' if is_tech else 'portuguese'
        
        sql = f"""
        WITH v_results AS (
            SELECT id, content, metadata, importance, last_accessed,
                   (1 - (embedding <=> %s::vector)) as sim,
                   ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) as r_v
            FROM memories
            LIMIT 100
        ),
        f_results AS (
            SELECT id, 
                   ts_rank_cd(fts_tokens, websearch_to_tsquery('{ts_cfg}', %s)) as rnk_f,
                   ROW_NUMBER() OVER (ORDER BY ts_rank_cd(fts_tokens, websearch_to_tsquery('{ts_cfg}', %s)) DESC) as r_f
            FROM memories 
            WHERE fts_tokens @@ websearch_to_tsquery('{ts_cfg}', %s)
               OR content ILIKE '%%' || %s || '%%'
            LIMIT 100
        )
        SELECT v.*, 
               (f.id IS NOT NULL AND v.content ILIKE '%%' || %s || '%%') as is_literal_match,
               (1.0 / (60 + v.r_v)) + (1.5 / (60 + COALESCE(f.r_f, 100))) as hybrid_score
        FROM v_results v
        LEFT JOIN f_results f ON v.id = f.id
        ORDER BY is_literal_match DESC, hybrid_score DESC
        LIMIT %s
        """
        # Passamos a query para os parâmetros do SQL
        return self.storage.fetch_all(sql, (query_emb, query_emb, query, query, query, query, query, limit))

    def _get_current_activation(self, mem_id, now):
        lvl = self.activation_levels.get(mem_id, 0.0)
        last = self.last_activation_update.get(mem_id, now)
        # Decay: meia-vida de 5 minutos
        return lvl * math.exp(-0.693 * (now - last) / 300)

    def _get_coactivation_map(self, candidate_ids):
        if not self.short_term_context: return {}
        rows = self.storage.get_coactivation_scores(candidate_ids, self.short_term_context)
        scores = {}
        for r in rows:
            target = r['id_a'] if r['id_a'] in candidate_ids else r['id_b']
            scores[target] = scores.get(target, 0.0) + float(r['strength'])
        if not scores: return {}
        m = max(scores.values())
        return {k: v/m for k, v in scores.items()}

    def _update_system_state(self, results, now):
        ids = [r['id'] for r in results]
        if self.short_term_context:
            pairs = [(old, new) for old in self.short_term_context for new in ids if old != new]
            self.storage.record_coactivation(pairs)
        self.storage.update_access(ids)
        for i in ids:
            self.activation_levels[i] = 1.0
            self.last_activation_update[i] = now
        self.short_term_context = ids