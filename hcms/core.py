# hcms/core.py
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
        self._warm_short_term_cache()
        
        # Lock para prevenir race conditions em atualizações concorrentes
        from threading import Lock
        self._context_lock = Lock()

    def _warm_short_term_cache(self):
        """Inicializa o contexto com memórias frequentes/recentes como baseline mental"""
        rows = self.storage.fetch_all("""
            SELECT id FROM memories 
            ORDER BY access_count DESC, last_accessed DESC 
            LIMIT 5
        """)
        self.short_term_context = [r['id'] for r in rows] if rows else []

    def _is_technical_query(self, query: str) -> bool:
        """Detecta códigos, IDs, UUIDs ou termos com caracteres especiais/números"""
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

        # 1. BUSCA HÍBRIDA DE PRECISÃO
        candidates = self._get_hybrid_candidates(query, query_emb, is_tech, limit=30)
        if not candidates:
            return []

        candidate_ids = [c['id'] for c in candidates]
        coactive_map = self._get_coactivation_map(candidate_ids)
        
        # CORREÇÃO CRÍTICA: Normalização de RRF
        # O valor máximo teórico do nosso RRF é ~0.041 (Rank 1 em ambos)
        # Normalizamos para que um hit perfeito seja 1.0
        MAX_RRF = (1.0 / 60.0) + (1.5 / 60.0)
        
        # 2. RANKING CONTEXTUAL COM ESCALA CORRIGIDA
        for c in candidates:
            # 1. Normaliza o Hybrid Score para escala 0-1
            s_hybrid_raw = float(c['hybrid_score'])
            s_hybrid_norm = min(1.0, s_hybrid_raw / MAX_RRF)
            
            # 2. Aplica o Multiplicador Literal (apenas se for técnico)
            is_literal = (is_tech and c['is_literal_match'])
            literal_boost = 2.0 if is_literal else 1.0
            
            # 3. Contexto e Ativação
            s_act = self._get_current_activation(c['id'], now)
            s_coact = float(coactive_map.get(c['id'], 0.0))
            
            # 4. Cálculo Final com pesos equilibrados
            # Boost literal agora afeta o componente híbrido antes da ponderação
            # Clampa em 1.0 para evitar overflow do boost
            c['final_score'] = (min(1.0, s_hybrid_norm * literal_boost) * 0.6) + \
                               (s_act * 0.25) + \
                               (s_coact * 0.15)

        # Ordena pelo score final
        results = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
        
        # 3. FILTRO DE ALUCINAÇÃO (Agora com escala corrigida)
        # Se mesmo normalizado o score é menor que 0.15, é ruído
        if results and results[0]['final_score'] < 0.15:
            return []
        
        results = results[:limit]
        
        # 4. ATUALIZAÇÃO DE ESTADO
        self._update_system_state(results, now)
        
        return results

    def _get_hybrid_candidates(self, query, query_emb, is_tech, limit):
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
        return self.storage.fetch_all(sql, (query_emb, query_emb, query, query, query, query, query, limit))

    def _get_current_activation(self, mem_id, now):
        lvl = self.activation_levels.get(mem_id, 0.0)
        last = self.last_activation_update.get(mem_id, now)
        # Decay ajustado: meia-vida de 15 minutos (mais realista para working memory)
        return lvl * math.exp(-0.693 * (now - last) / 900)

    def _get_coactivation_map(self, candidate_ids):
        if not self.short_term_context:
            return {}
        rows = self.storage.get_coactivation_scores(candidate_ids, self.short_term_context)
        scores = {}
        for r in rows:
            target = r['id_a'] if r['id_a'] in candidate_ids else r['id_b']
            scores[target] = scores.get(target, 0.0) + float(r['strength'])
        if not scores:
            return {}
        m = max(scores.values())
        return {k: v/m for k, v in scores.items()}

    def _update_system_state(self, results, now):
        # Lock crítico: previne race conditions quando múltiplas requisições
        # tentam atualizar o contexto simultaneamente
        with self._context_lock:
            ids = [r['id'] for r in results]
            if self.short_term_context:
                pairs = [(old, new) for old in self.short_term_context for new in ids if old != new]
                self.storage.record_coactivation(pairs)
            self.storage.update_access(ids, now)
            for i in ids:
                self.activation_levels[i] = 1.0
                self.last_activation_update[i] = now
            self.short_term_context = ids