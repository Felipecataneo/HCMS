# hcms/background_gc.py

import time
import threading
from datetime import datetime
from typing import Optional, Dict

class SimpleBackgroundGC:
    """
    Garbage Collector Cognitivo em Background (Thread Daemon).
    Executa ciclos de limpeza, poda e consolidação de memória sem bloquear
    a thread principal do Agente ou da API.
    """

    def __init__(self, hcms_instance, interval_seconds: int = 3600):
        """
        Args:
            hcms_instance: Instância do HCMSOptimized.
            interval_seconds: Intervalo entre os ciclos de limpeza (padrão 1 hora).
        """
        self.hcms = hcms_instance
        self.interval = interval_seconds
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.last_run_time: Optional[float] = None
        self.cycle_count = 0
        self.last_stats: Dict = {
            "deleted_noise": 0,
            "deleted_stale": 0,
            "archived": 0,
            "duplicates_removed": 0
        }

    def start(self):
        """Inicia o worker de manutenção em uma thread separada."""
        if self.running:
            print("⚠️  [Background GC] O worker já está em execução.")
            return

        self.running = True
        # daemon=True garante que a thread morra quando o processo principal (FastAPI) fechar
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        print(f"✅ [Background GC] Worker iniciado. Intervalo: {self.interval}s")

    def stop(self):
        """Para o worker de forma graciosa."""
        if not self.running:
            return
            
        print("🛑 [Background GC] Solicitando parada do worker...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("✅ [Background GC] Worker encerrado com sucesso.")

    def _worker_loop(self):
        """Loop principal de execução do worker."""
        # Aguarda um pouco antes do primeiro ciclo para não sobrecarregar o startup do servidor
        time.sleep(10)

        while self.running:
            try:
                start_time = time.time()
                current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"🧹 [{current_time_str}] Iniciando ciclo de manutenção cognitiva...")

                # 1. Sincroniza estatísticas de acesso (Contadores de uso)
                self.hcms.sync_access_stats()

                # 2. Executa Garbage Collection (Deleta ruído e memórias estagnadas)
                # Esta chamada utiliza o pruner.py que já implementamos
                gc_stats = self.hcms.pruner.run_garbage_collection()

                # 3. Consolida Duplicatas (Limpeza de redundância semântica)
                dupes_count = self.hcms.pruner.consolidate_duplicates()

                # 4. Atualiza importância via Grafo (PageRank simplificado)
                # Opcional: apenas se o método existir no seu core otimizado
                importance_updated = 0
                if hasattr(self.hcms, '_update_importance_scores'):
                    importance_updated = self.hcms._update_importance_scores()

                # Atualiza métricas internas do worker
                self.cycle_count += 1
                self.last_run_time = time.time()
                self.last_stats = {
                    "deleted_noise": gc_stats.get('deleted_noise', 0),
                    "deleted_stale": gc_stats.get('deleted_stale', 0),
                    "archived": gc_stats.get('archived', 0),
                    "duplicates_removed": dupes_count,
                    "importance_updated": importance_updated,
                    "duration_seconds": round(time.time() - start_time, 2)
                }

                print(f"✨ [{current_time_str}] Ciclo finalizado em {self.last_stats['duration_seconds']}s")
                print(f"   - Removidos: {self.last_stats['deleted_noise']} ruídos, {self.last_stats['duplicates_removed']} duplicatas")
                print(f"   - Arquivados: {self.last_stats['archived']} memórias")

            except Exception as e:
                print(f"❌ [Background GC] Erro crítico no ciclo de manutenção: {str(e)}")

            # Aguarda o intervalo definido antes da próxima execução
            # Fazemos o sleep em pequenos pedaços para permitir parada rápida do sistema
            elapsed = 0
            while elapsed < self.interval and self.running:
                time.sleep(1)
                elapsed += 1

    def get_status(self) -> Dict:
        """Retorna o estado atual e as últimas estatísticas do worker."""
        now = time.time()
        time_since_last = (now - self.last_run_time) if self.last_run_time else None
        
        return {
            "active": self.running,
            "interval_seconds": self.interval,
            "cycles_completed": self.cycle_count,
            "last_run_timestamp": self.last_run_time,
            "seconds_since_last_run": round(time_since_last, 1) if time_since_last else None,
            "last_cycle_metrics": self.last_stats
        }

def create_background_gc(hcms_instance, mode: str = "simple", interval: int = 3600):
    """
    Factory para criar o worker. 
    Mantido compatível com futuras implementações (APScheduler/Celery).
    """
    if mode == "simple":
        return SimpleBackgroundGC(hcms_instance, interval_seconds=interval)
    else:
        raise ValueError(f"Modo '{mode}' não suportado nesta versão simplificada. Use 'simple'.")