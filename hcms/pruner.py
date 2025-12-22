import time
from typing import Dict

class MemoryPruner:
    def __init__(self, hcms_instance):
        self.hcms = hcms_instance
        self.storage = hcms_instance.storage

    def run_garbage_collection(self) -> Dict[str, int]:
        """Executa o ciclo de limpeza cognitiva"""
        print("🧹 Iniciando Garbage Collection Cognitiva...")
        now = time.time()
        stats = {"deleted_noise": 0, "deleted_stale": 0, "archived": 0}

        # 1. Deletar Ruído (Low Importance + Old)
        # Ex: "Bom dia", "Tudo bem" - Se ninguém acessou em 7 dias, tchau.
        noise_ids = self.storage.fetch_all("""
            SELECT id FROM memories 
            WHERE importance < 0.3 
              AND last_access < %s 
              AND access_count < 2
        """, (now - (7 * 86400),))
        
        if noise_ids:
            ids = [n['id'] for n in noise_ids]
            self.storage.execute("DELETE FROM memories WHERE id = ANY(%s)", (ids,))
            stats["deleted_noise"] = len(ids)

        # 2. Deletar Fatos Estagnados (Medium Importance + Very Old)
        # Se não foi útil em 30 dias e não é crítico, deletamos.
        stale_ids = self.storage.fetch_all("""
            SELECT id FROM memories 
            WHERE importance < 0.7 
              AND last_access < %s 
              AND access_count = 0
        """, (now - (30 * 86400),))

        if stale_ids:
            ids = [s['id'] for s in stale_ids]
            self.storage.execute("DELETE FROM memories WHERE id = ANY(%s)", (ids,))
            stats["deleted_stale"] = len(ids)

        # 3. Arquivamento (High Importance + Never Accessed)
        # Move para Tier 3 (Archive) em vez de deletar, para economizar RAM/Índices
        archive_ids = self.storage.fetch_all("""
            SELECT id FROM memories 
            WHERE importance >= 0.7 
              AND last_access < %s 
              AND tier < 3
        """, (now - (90 * 86400),))

        if archive_ids:
            ids = [a['id'] for a in archive_ids]
            self.storage.execute("UPDATE memories SET tier = 3 WHERE id = ANY(%s)", (ids,))
            stats["archived"] = len(ids)

        return stats

    def consolidate_duplicates(self, similarity_threshold: float = 0.98):
        """Remove memórias quase idênticas (Duplicatas semânticas)"""
        # Busca pares com similaridade altíssima
        duplicates = self.storage.fetch_all("""
            SELECT m1.id as master_id, m2.id as duplicate_id
            FROM memories m1
            JOIN memories m2 ON (1 - (m1.embedding <=> m2.embedding)) > %s
            WHERE m1.id < m2.id 
              AND m1.content = m2.content
        """, (similarity_threshold,))

        if not duplicates: return 0

        to_delete = [d['duplicate_id'] for d in duplicates]
        self.storage.execute("DELETE FROM memories WHERE id = ANY(%s)", (to_delete,))
        return len(to_delete)