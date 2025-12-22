"""
Sistema automático de promoção/demoção de tiers
Baseado em access patterns e recency
"""

import time
from typing import Dict, List
import numpy as np

class TierManager:
    """
    Gerencia transições automáticas entre tiers:
    
    HOT (0):    Acesso frequente, sem compressão, latência <10ms
    WARM (1):   Acesso moderado, scalar quant (4x), latência ~20ms  
    COLD (2):   Acesso raro, PQ (192x), latência ~100ms
    ARCHIVE (3): Consolidado/deletado, não retornado em recall
    
    Transições:
    - Acesso → WARM/COLD promovem para HOT
    - 7 dias sem acesso → HOT demove para WARM
    - 30 dias sem acesso → WARM demove para COLD
    - Similaridade >0.95 → candidato para ARCHIVE (consolidação)
    """
    
    # Thresholds de transição (em segundos)
    HOT_TO_WARM_THRESHOLD = 7 * 86400      # 7 dias
    WARM_TO_COLD_THRESHOLD = 30 * 86400    # 30 dias
    COLD_TO_ARCHIVE_THRESHOLD = 90 * 86400 # 90 dias
    
    # Thresholds de promoção (acessos em janela)
    PROMOTE_TO_HOT_ACCESSES = 3  # 3+ acessos em 24h
    PROMOTE_WINDOW = 86400       # 24 horas
    
    def __init__(self, hcms_instance):
        self.hcms = hcms_instance
        self.storage = hcms_instance.storage
    
    # ============================================================
    # PROMOÇÃO (ACCESS-DRIVEN)
    # ============================================================
    
    def promote_hot_memories(self) -> int:
        """
        Promove memórias com acesso frequente recente para HOT tier
        Roda após sync_access_stats()
        """
        now = time.time()
        window_start = now - self.PROMOTE_WINDOW
        
        # Busca memórias não-HOT com 3+ acessos em 24h
        candidates = self.storage.fetch_all(
            """
            WITH recent_access AS (
                SELECT mem_id, COUNT(*) as recent_count
                FROM access_logs
                WHERE access_time > %s
                GROUP BY mem_id
                HAVING COUNT(*) >= %s
            )
            SELECT m.id, m.tier, m.compression_type, ra.recent_count
            FROM memories m
            JOIN recent_access ra ON m.id = ra.mem_id
            WHERE m.tier > 0
              AND m.tier < 3
            ORDER BY ra.recent_count DESC
            LIMIT 100
            """,
            (window_start, self.PROMOTE_TO_HOT_ACCESSES)
        )
        
        if not candidates:
            return 0
        
        promoted = 0
        for mem in candidates:
            # Descomprime se necessário
            if mem['compression_type'] != 'none':
                self._decompress_memory(mem['id'])
            
            # Promove para HOT
            self.storage.execute(
                """
                UPDATE memories
                SET tier = 0,
                    compression_type = 'none'
                WHERE id = %s
                """,
                (mem['id'],)
            )
            promoted += 1
        
        return promoted
    
    # ============================================================
    # DEMOÇÃO (TIME-DRIVEN)
    # ============================================================
    
    def demote_inactive_memories(self, batch_size: int = 1000) -> Dict[str, int]:
        """
        Demove memórias inativas para tiers inferiores
        """
        now = time.time()
        stats = {'hot_to_warm': 0, 'warm_to_cold': 0}
        
        # HOT → WARM
        hot_cutoff = now - self.HOT_TO_WARM_THRESHOLD
        hot_demotions = self.storage.fetch_all(
            """
            SELECT id FROM memories
            WHERE tier = 0
              AND last_access < %s
            LIMIT %s
            """,
            (hot_cutoff, batch_size)
        )
        
        for mem in hot_demotions:
            self._compress_and_demote(mem['id'], target_tier=1)
            stats['hot_to_warm'] += 1
        
        # WARM → COLD
        warm_cutoff = now - self.WARM_TO_COLD_THRESHOLD
        warm_demotions = self.storage.fetch_all(
            """
            SELECT id FROM memories
            WHERE tier = 1
              AND last_access < %s
            LIMIT %s
            """,
            (warm_cutoff, batch_size)
        )
        
        for mem in warm_demotions:
            self._compress_and_demote(mem['id'], target_tier=2)
            stats['warm_to_cold'] += 1
        
        return stats
    
    # ============================================================
    # CONSOLIDAÇÃO PARA ARCHIVE
    # ============================================================
    
    def archive_redundant_clusters(self, similarity_threshold: float = 0.95) -> int:
        """
        Move clusters redundantes para ARCHIVE tier
        Mais agressivo que consolidate_redundant_memories()
        """
        cold_cutoff = time.time() - self.COLD_TO_ARCHIVE_THRESHOLD
        
        # Busca apenas memórias COLD antigas
        candidates = self.storage.fetch_all(
            """
            SELECT m1.id AS master_id, 
                   ARRAY_AGG(m2.id) AS cluster_ids,
                   MIN(m1.last_access) AS oldest_access
            FROM memories m1
            JOIN memories m2
              ON (1 - (m1.embedding <=> m2.embedding)) > %s
            WHERE m1.id < m2.id
              AND m1.tier = 2
              AND m2.tier = 2
              AND m1.last_access < %s
              AND m2.last_access < %s
              AND (m1.metadata->>'consolidated') IS NULL
            GROUP BY m1.id
            HAVING COUNT(*) >= 2
            ORDER BY MIN(m1.last_access) ASC
            LIMIT 20
            """,
            (similarity_threshold, cold_cutoff, cold_cutoff)
        )
        
        archived = 0
        for cluster in candidates:
            ids = [cluster['master_id']] + cluster['cluster_ids']
            
            # Move diretamente para ARCHIVE (não cria summary)
            self.storage.execute(
                """
                UPDATE memories
                SET tier = 3,
                    metadata = metadata || '{"archived": true}'::jsonb
                WHERE id = ANY(%s)
                """,
                (ids,)
            )
            archived += len(ids)
        
        return archived
    
    # ============================================================
    # HELPERS
    # ============================================================
    
    def _decompress_memory(self, mem_id: str):
        """Descomprime embedding e texto"""
        row = self.storage.fetch_all(
            """
            SELECT compressed_data, compression_type, 
                   compressed_embedding, embedding_compression
            FROM memories WHERE id = %s
            """,
            (mem_id,)
        )[0]
        
        # Descomprime texto
        if row['compressed_data']:
            content = self.hcms.zstd.decompress(row['compressed_data']).decode('utf-8')
            self.storage.execute(
                "UPDATE memories SET content = %s WHERE id = %s",
                (content, mem_id)
            )
        
        # Descomprime embedding (se tiver compressor)
        if hasattr(self.hcms, 'compressor') and row['compressed_embedding']:
            embedding = self.hcms.compressor.decompress(
                row['compressed_embedding'], 
                row['embedding_compression']
            )
            self.storage.execute(
                "UPDATE memories SET embedding = %s WHERE id = %s",
                (embedding.tolist(), mem_id)
            )
    
    def _compress_and_demote(self, mem_id: str, target_tier: int):
        """Comprime e demove para tier alvo, lidando com dados já comprimidos"""
        # Busca o estado atual da memória
        row = self.storage.fetch_all(
            """
            SELECT content, compressed_data, compression_type, 
                   embedding, compressed_embedding, embedding_compression 
            FROM memories WHERE id = %s
            """,
            (mem_id,)
        )[0]
        
        # 1. Gerenciamento do Texto (Zstd)
        compressed_text = row['compressed_data']
        
        # Se o conteúdo ainda estiver em plain text, comprime agora
        if row['content'] is not None:
            compressed_text = self.hcms.zstd.compress(row['content'].encode('utf-8'))
        elif compressed_text is None:
            # Se não tem content nem compressed_data, a memória está vazia ou corrompida
            print(f"⚠️ Memória {mem_id} sem conteúdo para comprimir. Pulando.")
            return

        # 2. Gerenciamento do Embedding (Quantization)
        # Tenta obter o embedding original para re-comprimir
        embedding_to_process = None
        
        if row['embedding'] is not None:
            # Parsing de string se necessário (correção do erro anterior)
            emb_data = row['embedding']
            if isinstance(emb_data, str):
                emb_data = [float(x) for x in emb_data.strip('[]').split(',')]
            embedding_to_process = np.array(emb_data, dtype=np.float32)
        elif row['compressed_embedding'] is not None and hasattr(self.hcms, 'compressor'):
            # Se já está comprimido, precisamos descomprimir para re-comprimir no novo tier
            # (Ex: de Scalar para PQ)
            embedding_to_process = self.hcms.compressor.decompress(
                row['compressed_embedding'], 
                row['embedding_compression']
            )

        # 3. Execução da Compressão de Embedding
        final_comp_emb = row['compressed_embedding']
        final_emb_type = row['embedding_compression']
        
        if embedding_to_process is not None and hasattr(self.hcms, 'compressor'):
            
            final_comp_emb, final_emb_type = self.hcms.compressor.compress(
                embedding_to_process, target_tier
            )

        # 4. Atualização Atômica
        self.storage.execute(
            """
            UPDATE memories
            SET content = NULL,
                compressed_data = %s,
                compression_type = 'zstd',
                compressed_embedding = %s,
                embedding_compression = %s,
                tier = %s
            WHERE id = %s
            """,
            (compressed_text, final_comp_emb, final_emb_type, target_tier, mem_id)
        )
    
    # ============================================================
    # MAINTENANCE JOB
    # ============================================================
    
    def run_maintenance(self) -> Dict:
        """
        Job completo de manutenção de tiers
        Roda periodicamente (ex: a cada 1 hora)
        """
        print("🔧 Iniciando tier maintenance...")
        start = time.time()
        
        # 1. Sync access stats
        self.hcms.sync_access_stats()
        
        # 2. Promove memórias hot
        promoted = self.promote_hot_memories()
        
        # 3. Demove memórias inativas
        demotions = self.demote_inactive_memories()
        
        # 4. Archive clusters redundantes
        archived = self.archive_redundant_clusters()
        
        elapsed = time.time() - start
        
        stats = {
            'promoted_to_hot': promoted,
            'hot_to_warm': demotions['hot_to_warm'],
            'warm_to_cold': demotions['warm_to_cold'],
            'archived': archived,
            'elapsed_seconds': elapsed
        }
        
        print(f"✅ Maintenance completo em {elapsed:.1f}s")
        print(f"   Promoted: {promoted}, Demoted: {sum(demotions.values())}, Archived: {archived}")
        
        return stats
    
    def get_tier_distribution(self) -> Dict:
        """Estatísticas da distribuição de tiers"""
        rows = self.storage.fetch_all(
            """
            SELECT tier, 
                   COUNT(*) as count,
                   AVG(access_count) as avg_accesses,
                   AVG(EXTRACT(EPOCH FROM NOW()) - last_access) as avg_age_seconds
            FROM memories
            GROUP BY tier
            ORDER BY tier
            """
        )
        
        return {
            f"tier_{r['tier']}": {
                'count': r['count'],
                'avg_accesses': float(r['avg_accesses']),
                'avg_age_days': r['avg_age_seconds'] / 86400 if r['avg_age_seconds'] else 0
            }
            for r in rows
        }