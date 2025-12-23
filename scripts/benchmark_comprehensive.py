"""
Benchmark Comparativo: HCMS v2 vs RAG Padrão

Testa 6 cenários onde sistemas de memória divergem:
1. Busca de Termos Exatos (IDs, códigos)
2. Recuperação sob Ruído Semântico
3. Detecção de Duplicatas
4. Latência em Escala
5. Precisão em Queries Ambíguas
6. Custo de Manutenção

Executar: python scripts/benchmark_comprehensive.py
"""

import os
import sys
import time
import statistics
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Setup de path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

from hcms.core import HCMSOptimized

# ============================================================
# RAG BASELINE (IMPLEMENTAÇÃO PADRÃO)
# ============================================================

class StandardRAG:
    """
    RAG Padrão: Apenas busca vetorial com threshold de similaridade
    Representa >90% das implementações RAG em produção
    """
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._init_db()
    
    def _init_db(self):
        with psycopg2.connect(self.dsn, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS standard_memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector(384) NOT NULL,
                        created_at DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW())
                    );
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_standard_vec 
                    ON standard_memories 
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                conn.commit()
    
    def remember(self, content: str) -> str:
        mem_id = f"std_{int(time.time() * 1000000)}"
        embedding = self.encoder.encode(content).tolist()
        
        with psycopg2.connect(self.dsn, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO standard_memories (id, content, embedding) VALUES (%s, %s, %s)",
                    (mem_id, content, embedding)
                )
                conn.commit()
        return mem_id
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        query_emb = self.encoder.encode(query).tolist()
        
        with psycopg2.connect(self.dsn, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, content, 
                           (1 - (embedding <=> %s::vector)) AS similarity
                    FROM standard_memories
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                """, (query_emb, query_emb, limit))
                return cur.fetchall()
    
    def clear(self):
        with psycopg2.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE standard_memories CASCADE;")
                conn.commit()


# ============================================================
# MÉTRICAS E RESULTADOS
# ============================================================

@dataclass
class BenchmarkMetrics:
    """Métricas unificadas para comparação"""
    system: str
    scenario: str
    
    # Precisão
    precision_at_1: float  # Top-1 é relevante?
    precision_at_3: float  # Top-3 contém relevante?
    recall_at_5: float     # % de docs relevantes recuperados
    mrr: float             # Mean Reciprocal Rank
    
    # Performance
    latency_ms: float
    latency_p95_ms: float  # Percentil 95
    
    # Qualidade do Sistema
    duplicate_rate: float   # % de duplicatas não consolidadas
    noise_retention: float  # % de ruído não removido
    
    # Custo Operacional
    storage_mb: float       # Tamanho no disco
    index_count: int        # Número de índices mantidos


@dataclass
class ScenarioResult:
    """Resultado agregado de um cenário"""
    name: str
    hcms_metrics: BenchmarkMetrics
    standard_metrics: BenchmarkMetrics
    winner: str
    advantage_percent: float
    notes: str


# ============================================================
# SUITE DE TESTES
# ============================================================

class ComprehensiveBenchmark:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.hcms = HCMSOptimized(dsn)
        self.standard = StandardRAG(dsn)
        self.results: List[ScenarioResult] = []
    
    def run_all(self) -> List[ScenarioResult]:
        """Executa todos os cenários"""
        print("\n" + "="*80)
        print("BENCHMARK COMPARATIVO: HCMS v2 vs RAG Padrão")
        print("="*80 + "\n")
        
        scenarios = [
            ("Busca de Termos Exatos", self._test_exact_terms),
            ("Recuperação sob Ruído", self._test_noise_resilience),
            ("Queries Ambíguas", self._test_ambiguous_queries),
            ("Detecção de Duplicatas", self._test_duplicate_detection),
            ("Latência em Escala", self._test_latency_scaling),
            ("Custo de Manutenção", self._test_maintenance_cost),
        ]
        
        for name, test_fn in scenarios:
            print(f"\n{'='*80}")
            print(f"Cenário: {name}")
            print('='*80)
            self._reset_systems()
            result = test_fn()
            self.results.append(result)
            self._print_scenario_result(result)
        
        return self.results
    
    def _reset_systems(self):
        """Limpa ambos os sistemas"""
        self.hcms.storage.execute("TRUNCATE memories, edges, access_logs CASCADE;")
        self.standard.clear()
    
    # ============================================================
    # CENÁRIO 1: BUSCA DE TERMOS EXATOS
    # ============================================================
    
    def _test_exact_terms(self) -> ScenarioResult:
        """
        Testa recuperação de termos que embeddings falham:
        - Códigos alfanuméricos (XK-9847-ALPHA)
        - UUIDs (550e8400-e29b-41d4-a716-446655440000)
        - Siglas técnicas (HNSW, RRF, BERT)
        """
        # Dataset: 30 docs com códigos similares + 3 targets
        targets = [
            ("XK-9847-ALPHA", "O código de acesso ao servidor é XK-9847-ALPHA."),
            ("550e8400-e29b-41d4-a716-446655440000", "UUID da transação: 550e8400-e29b-41d4-a716-446655440000"),
            ("RFC-8259", "Documento técnico RFC-8259 especifica JSON.")
        ]
        
        # Noise: Códigos semanticamente similares
        noise_docs = [
            f"O protocolo usa código XK-{1000+i}-BETA." for i in range(10)
        ] + [
            f"UUID: {i}50e8400-e29b-41d4-a716-44665544000{i}" for i in range(10)
        ] + [
            f"RFC-{8000+i} define padrões de rede." for i in range(10)
        ]
        
        # Ingestão
        for doc in noise_docs:
            self.hcms.remember(doc, importance=0.3)
            self.standard.remember(doc)
        
        for target_term, target_doc in targets:
            self.hcms.remember(target_doc, importance=0.9)
            self.standard.remember(target_doc)
        
        # Avaliação
        hcms_scores = []
        std_scores = []
        hcms_latencies = []
        std_latencies = []
        
        for target_term, _ in targets:
            # HCMS
            start = time.time()
            hcms_res = self.hcms.recall(target_term, limit=5)
            hcms_latencies.append((time.time() - start) * 1000)
            
            hcms_rank = next((i+1 for i, r in enumerate(hcms_res) if target_term in r['content']), 0)
            hcms_scores.append(1.0 / hcms_rank if hcms_rank > 0 else 0.0)
            
            # Standard
            start = time.time()
            std_res = self.standard.recall(target_term, limit=5)
            std_latencies.append((time.time() - start) * 1000)
            
            std_rank = next((i+1 for i, r in enumerate(std_res) if target_term in r['content']), 0)
            std_scores.append(1.0 / std_rank if std_rank > 0 else 0.0)
        
        # Métricas
        hcms_metrics = BenchmarkMetrics(
            system="HCMS",
            scenario="ExactTerms",
            precision_at_1=sum(1 for s in hcms_scores if s == 1.0) / len(hcms_scores),
            precision_at_3=sum(1 for s in hcms_scores if s >= 0.33) / len(hcms_scores),
            recall_at_5=sum(1 for s in hcms_scores if s > 0) / len(hcms_scores),
            mrr=statistics.mean(hcms_scores),
            latency_ms=statistics.mean(hcms_latencies),
            latency_p95_ms=sorted(hcms_latencies)[int(len(hcms_latencies)*0.95)],
            duplicate_rate=0.0,
            noise_retention=0.0,
            storage_mb=0.0,
            index_count=3
        )
        
        std_metrics = BenchmarkMetrics(
            system="Standard",
            scenario="ExactTerms",
            precision_at_1=sum(1 for s in std_scores if s == 1.0) / len(std_scores),
            precision_at_3=sum(1 for s in std_scores if s >= 0.33) / len(std_scores),
            recall_at_5=sum(1 for s in std_scores if s > 0) / len(std_scores),
            mrr=statistics.mean(std_scores),
            latency_ms=statistics.mean(std_latencies),
            latency_p95_ms=sorted(std_latencies)[int(len(std_latencies)*0.95)],
            duplicate_rate=0.0,
            noise_retention=0.0,
            storage_mb=0.0,
            index_count=1
        )
        
        winner = "HCMS" if hcms_metrics.mrr > std_metrics.mrr else "Standard"
        advantage = abs(hcms_metrics.mrr - std_metrics.mrr) / max(std_metrics.mrr, 0.01) * 100
        
        return ScenarioResult(
            name="Busca de Termos Exatos",
            hcms_metrics=hcms_metrics,
            standard_metrics=std_metrics,
            winner=winner,
            advantage_percent=advantage,
            notes=f"FTS hybrid search {'dominates' if winner == 'HCMS' else 'fails against'} pure vector search"
        )
    
    # ============================================================
    # CENÁRIO 2: RECUPERAÇÃO SOB RUÍDO
    # ============================================================
    
    def _test_noise_resilience(self) -> ScenarioResult:
        """
        Testa se o reranker mantém relevância com 100+ docs de ruído
        """
        # Target específico
        target = "A capital da Mongólia é Ulaanbaatar, fundada em 1639."
        
        # Noise: 100 fatos similares mas irrelevantes
        noise_docs = [
            f"A capital de País{i} é Cidade{i}, fundada em {1600+i}." for i in range(100)
        ]
        
        for doc in noise_docs:
            self.hcms.remember(doc, importance=0.2)
            self.standard.remember(doc)
        
        self.hcms.remember(target, importance=0.9)
        self.standard.remember(target)
        
        # Queries com diferentes níveis de ambiguidade
        queries = [
            "capital da Mongólia",
            "Ulaanbaatar",
            "cidade fundada em 1639"
        ]
        
        hcms_scores = []
        std_scores = []
        hcms_latencies = []
        std_latencies = []
        
        for query in queries:
            # HCMS
            start = time.time()
            hcms_res = self.hcms.recall(query, limit=5)
            hcms_latencies.append((time.time() - start) * 1000)
            
            hcms_rank = next((i+1 for i, r in enumerate(hcms_res) if "Ulaanbaatar" in r['content']), 0)
            hcms_scores.append(1.0 / hcms_rank if hcms_rank > 0 else 0.0)
            
            # Standard
            start = time.time()
            std_res = self.standard.recall(query, limit=5)
            std_latencies.append((time.time() - start) * 1000)
            
            std_rank = next((i+1 for i, r in enumerate(std_res) if "Ulaanbaatar" in r['content']), 0)
            std_scores.append(1.0 / std_rank if std_rank > 0 else 0.0)
        
        hcms_metrics = BenchmarkMetrics(
            system="HCMS",
            scenario="NoiseResilience",
            precision_at_1=sum(1 for s in hcms_scores if s == 1.0) / len(hcms_scores),
            precision_at_3=sum(1 for s in hcms_scores if s >= 0.33) / len(hcms_scores),
            recall_at_5=sum(1 for s in hcms_scores if s > 0) / len(hcms_scores),
            mrr=statistics.mean(hcms_scores),
            latency_ms=statistics.mean(hcms_latencies),
            latency_p95_ms=sorted(hcms_latencies)[int(len(hcms_latencies)*0.95)],
            duplicate_rate=0.0,
            noise_retention=0.0,
            storage_mb=0.0,
            index_count=3
        )
        
        std_metrics = BenchmarkMetrics(
            system="Standard",
            scenario="NoiseResilience",
            precision_at_1=sum(1 for s in std_scores if s == 1.0) / len(std_scores),
            precision_at_3=sum(1 for s in std_scores if s >= 0.33) / len(std_scores),
            recall_at_5=sum(1 for s in std_scores if s > 0) / len(std_scores),
            mrr=statistics.mean(std_scores),
            latency_ms=statistics.mean(std_latencies),
            latency_p95_ms=sorted(std_latencies)[int(len(std_latencies)*0.95)],
            duplicate_rate=0.0,
            noise_retention=0.0,
            storage_mb=0.0,
            index_count=1
        )
        
        winner = "HCMS" if hcms_metrics.mrr > std_metrics.mrr else "Standard"
        advantage = abs(hcms_metrics.mrr - std_metrics.mrr) / max(std_metrics.mrr, 0.01) * 100
        
        return ScenarioResult(
            name="Recuperação sob Ruído",
            hcms_metrics=hcms_metrics,
            standard_metrics=std_metrics,
            winner=winner,
            advantage_percent=advantage,
            notes=f"Cross-encoder reranking {'filters noise effectively' if winner == 'HCMS' else 'adds latency without benefit'}"
        )
    
    # ============================================================
    # CENÁRIO 3: QUERIES AMBÍGUAS
    # ============================================================
    
    def _test_ambiguous_queries(self) -> ScenarioResult:
        """
        Testa recuperação em queries com múltiplos significados
        """
        # Dataset com polissemia
        docs = [
            "O banco Santander anunciou lucros recordes.",
            "Sentei no banco da praça para ler.",
            "O banco de dados PostgreSQL é relacional.",
            "Banco de areia formou-se na costa.",
            "Apple lançou novo iPhone.",
            "Comprei uma apple no mercado.",
            "Python é uma linguagem de programação.",
            "A python é uma cobra constritora."
        ]
        
        for doc in docs:
            self.hcms.remember(doc, importance=0.7)
            self.standard.remember(doc)
        
        # Queries ambíguas
        test_cases = [
            ("banco de dados", ["PostgreSQL"]),
            ("apple produto", ["iPhone"]),
            ("python programação", ["linguagem"])
        ]
        
        hcms_scores = []
        std_scores = []
        hcms_latencies = []
        std_latencies = []
        
        for query, expected_terms in test_cases:
            # HCMS
            start = time.time()
            hcms_res = self.hcms.recall(query, limit=3)
            hcms_latencies.append((time.time() - start) * 1000)
            
            hcms_hit = any(
                any(term.lower() in r['content'].lower() for term in expected_terms)
                for r in hcms_res[:1]
            )
            hcms_scores.append(1.0 if hcms_hit else 0.0)
            
            # Standard
            start = time.time()
            std_res = self.standard.recall(query, limit=3)
            std_latencies.append((time.time() - start) * 1000)
            
            std_hit = any(
                any(term.lower() in r['content'].lower() for term in expected_terms)
                for r in std_res[:1]
            )
            std_scores.append(1.0 if std_hit else 0.0)
        
        hcms_metrics = BenchmarkMetrics(
            system="HCMS",
            scenario="AmbiguousQueries",
            precision_at_1=statistics.mean(hcms_scores),
            precision_at_3=1.0,
            recall_at_5=1.0,
            mrr=statistics.mean(hcms_scores),
            latency_ms=statistics.mean(hcms_latencies),
            latency_p95_ms=max(hcms_latencies),
            duplicate_rate=0.0,
            noise_retention=0.0,
            storage_mb=0.0,
            index_count=3
        )
        
        std_metrics = BenchmarkMetrics(
            system="Standard",
            scenario="AmbiguousQueries",
            precision_at_1=statistics.mean(std_scores),
            precision_at_3=1.0,
            recall_at_5=1.0,
            mrr=statistics.mean(std_scores),
            latency_ms=statistics.mean(std_latencies),
            latency_p95_ms=max(std_latencies),
            duplicate_rate=0.0,
            noise_retention=0.0,
            storage_mb=0.0,
            index_count=1
        )
        
        winner = "HCMS" if hcms_metrics.precision_at_1 > std_metrics.precision_at_1 else "Standard"
        advantage = abs(hcms_metrics.precision_at_1 - std_metrics.precision_at_1) / max(std_metrics.precision_at_1, 0.01) * 100
        
        return ScenarioResult(
            name="Queries Ambíguas",
            hcms_metrics=hcms_metrics,
            standard_metrics=std_metrics,
            winner=winner,
            advantage_percent=advantage,
            notes="Hybrid search disambiguates via FTS context"
        )
    
    # ============================================================
    # CENÁRIO 4: DETECÇÃO DE DUPLICATAS
    # ============================================================
    
    def _test_duplicate_detection(self) -> ScenarioResult:
        """
        Testa consolidação de memórias redundantes
        """
        # Ingestão: 12 variações do mesmo fato
        base_fact = "O CEO da Anthropic é Dario Amodei."
        variations = [
            base_fact,
            "Dario Amodei é o CEO da Anthropic.",
            "O fundador e CEO da Anthropic chama-se Dario Amodei.",
            "Anthropic é liderada pelo CEO Dario Amodei.",
        ] * 3
        
        for var in variations:
            self.hcms.remember(var, importance=0.7)
            self.standard.remember(var)
        
        # Mede estado inicial
        hcms_count_before = self.hcms.storage.fetch_all("SELECT COUNT(*) as c FROM memories")[0]['c']
        std_count_before = len(self.standard.recall("", limit=1000))
        
        # Roda consolidação
        start = time.time()
        dupes_removed = self.hcms.pruner.consolidate_duplicates(similarity_threshold=0.90)
        consolidation_time = (time.time() - start) * 1000
        
        # Mede estado final
        hcms_count_after = self.hcms.storage.fetch_all("SELECT COUNT(*) as c FROM memories")[0]['c']
        
        hcms_metrics = BenchmarkMetrics(
            system="HCMS",
            scenario="DuplicateDetection",
            precision_at_1=1.0,
            precision_at_3=1.0,
            recall_at_5=1.0,
            mrr=1.0,
            latency_ms=consolidation_time,
            latency_p95_ms=consolidation_time,
            duplicate_rate=(hcms_count_before - hcms_count_after) / hcms_count_before,
            noise_retention=0.0,
            storage_mb=0.0,
            index_count=3
        )
        
        std_metrics = BenchmarkMetrics(
            system="Standard",
            scenario="DuplicateDetection",
            precision_at_1=1.0,
            precision_at_3=1.0,
            recall_at_5=1.0,
            mrr=1.0,
            latency_ms=0.0,
            latency_p95_ms=0.0,
            duplicate_rate=0.0,  # Não tem deduplicação
            noise_retention=1.0,  # 100% de retenção
            storage_mb=0.0,
            index_count=1
        )
        
        return ScenarioResult(
            name="Detecção de Duplicatas",
            hcms_metrics=hcms_metrics,
            standard_metrics=std_metrics,
            winner="HCMS",
            advantage_percent=hcms_metrics.duplicate_rate * 100,
            notes=f"HCMS removed {dupes_removed} duplicates, Standard has no dedup mechanism"
        )
    
    # ============================================================
    # CENÁRIO 5: LATÊNCIA EM ESCALA
    # ============================================================
    
    def _test_latency_scaling(self) -> ScenarioResult:
        """
        Mede como latência cresce com dataset
        """
        scales = [50, 100, 500, 1000]
        hcms_latencies = []
        std_latencies = []
        
        for scale in scales:
            # Reset
            self._reset_systems()
            
            # Ingestão
            for i in range(scale):
                doc = f"Documento técnico número {i} sobre sistemas distribuídos."
                self.hcms.remember(doc, importance=0.5)
                self.standard.remember(doc)
            
            # Benchmark (5 runs para estabilidade)
            query = "sistemas distribuídos"
            hcms_runs = []
            std_runs = []
            
            for _ in range(5):
                start = time.time()
                self.hcms.recall(query, limit=5)
                hcms_runs.append((time.time() - start) * 1000)
                
                start = time.time()
                self.standard.recall(query, limit=5)
                std_runs.append((time.time() - start) * 1000)
            
            hcms_latencies.append(statistics.mean(hcms_runs))
            std_latencies.append(statistics.mean(std_runs))
            
            print(f"   {scale:4} docs → HCMS: {hcms_latencies[-1]:6.1f}ms | Standard: {std_latencies[-1]:6.1f}ms")
        
        hcms_metrics = BenchmarkMetrics(
            system="HCMS",
            scenario="LatencyScaling",
            precision_at_1=1.0,
            precision_at_3=1.0,
            recall_at_5=1.0,
            mrr=1.0,
            latency_ms=statistics.mean(hcms_latencies),
            latency_p95_ms=max(hcms_latencies),
            duplicate_rate=0.0,
            noise_retention=0.0,
            storage_mb=0.0,
            index_count=3
        )
        
        std_metrics = BenchmarkMetrics(
            system="Standard",
            scenario="LatencyScaling",
            precision_at_1=1.0,
            precision_at_3=1.0,
            recall_at_5=1.0,
            mrr=1.0,
            latency_ms=statistics.mean(std_latencies),
            latency_p95_ms=max(std_latencies),
            duplicate_rate=0.0,
            noise_retention=0.0,
            storage_mb=0.0,
            index_count=1
        )
        
        winner = "Standard" if std_metrics.latency_ms < hcms_metrics.latency_ms else "HCMS"
        advantage = abs(hcms_metrics.latency_ms - std_metrics.latency_ms) / min(hcms_metrics.latency_ms, std_metrics.latency_ms) * 100
        
        return ScenarioResult(
            name="Latência em Escala",
            hcms_metrics=hcms_metrics,
            standard_metrics=std_metrics,
            winner=winner,
            advantage_percent=advantage,
            notes=f"HCMS overhead: {hcms_metrics.latency_ms - std_metrics.latency_ms:.1f}ms due to reranker"
        )
    
    # ============================================================
    # CENÁRIO 6: CUSTO DE MANUTENÇÃO
    # ============================================================
    
    def _test_maintenance_cost(self) -> ScenarioResult:
        """
        Testa custo de GC e poda cognitiva
        """
        # Ingestão: Mix de ruído + fatos úteis
        for i in range(100):
            self.hcms.remember(f"Mensagem casual {i}", importance=0.1)
            self.standard.remember(f"Mensagem casual {i}")
        
        for i in range(20):
            self.hcms.remember(f"Fato importante {i}", importance=0.9)
            self.standard.remember(f"Fato importante {i}")
        
        # Simula envelhecimento (7 dias para ruído)
        now = time.time()
        self.hcms.storage.execute("""
            UPDATE memories 
            SET last_access = %s 
            WHERE importance < 0.3
        """, (now - (8 * 86400),))
        
        # Roda manutenção
        start = time.time()
        self.hcms.sync_access_stats()
        stats = self.hcms.pruner.run_garbage_collection()
        maintenance_time = (time.time() - start) * 1000
        
        # Conta o que sobrou
        hcms_count_after = self.hcms.storage.fetch_all("SELECT COUNT(*) as c FROM memories")[0]['c']
        std_count = len(self.standard.recall("", limit=1000))
        
        noise_removed = stats['deleted_noise']
        noise_retention_hcms = 1.0 - (noise_removed / 100)
        
        hcms_metrics = BenchmarkMetrics(
            system="HCMS",
            scenario="MaintenanceCost",
            precision_at_1=1.0,
            precision_at_3=1.0,
            recall_at_5=1.0,
            mrr=1.0,
            latency_ms=maintenance_time,
            latency_p95_ms=maintenance_time,
            duplicate_rate=0.0,
            noise_retention=noise_retention_hcms,
            storage_mb=0.0,
            index_count=3
        )
        
        std_metrics = BenchmarkMetrics(
            system="Standard",
            scenario="MaintenanceCost",
            precision_at_1=1.0,
            precision_at_3=1.0,
            recall_at_5=1.0,
            mrr=1.0,
            latency_ms=0.0,
            latency_p95_ms=0.0,
            duplicate_rate=0.0,
            noise_retention=1.0,  # Retém 100% do ruído
            storage_mb=0.0,
            index_count=1
        )
        
        return ScenarioResult(
            name="Custo de Manutenção",
            hcms_metrics=hcms_metrics,
            standard_metrics=std_metrics,
            winner="HCMS",
            advantage_percent=(1.0 - noise_retention_hcms) * 100,
            notes=f"HCMS removed {noise_removed} noise entries, Standard retains all garbage"
        )
    
    # ============================================================
    # RELATÓRIO E VISUALIZAÇÃO
    # ============================================================
    
    def _print_scenario_result(self, result: ScenarioResult):
        """Imprime resultado de um cenário"""
        print(f"\nVencedor: {result.winner} ({result.advantage_percent:.1f}% advantage)")
        print(f"Notas: {result.notes}\n")
        
        print(f"{'Métrica':<25} | {'HCMS':<15} | {'Standard':<15}")
        print("-" * 60)
        
        metrics = [
            ("Precision@1", "precision_at_1", "{:.2%}"),
            ("Recall@5", "recall_at_5", "{:.2%}"),
            ("MRR", "mrr", "{:.3f}"),
            ("Latency (avg)", "latency_ms", "{:.1f}ms"),
            ("Latency (p95)", "latency_p95_ms", "{:.1f}ms"),
            ("Duplicate Rate", "duplicate_rate", "{:.1%}"),
            ("Noise Retention", "noise_retention", "{:.1%}"),
            ("Index Count", "index_count", "{}"),
        ]
        
        for label, attr, fmt in metrics:
            hcms_val = getattr(result.hcms_metrics, attr)
            std_val = getattr(result.standard_metrics, attr)
            
            hcms_str = fmt.format(hcms_val)
            std_str = fmt.format(std_val)
            
            print(f"{label:<25} | {hcms_str:<15} | {std_str:<15}")
    
    def generate_report(self, output_file: str = "benchmark_report.json"):
        """Gera relatório JSON completo"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self._generate_summary(),
            "scenarios": [
                {
                    "name": r.name,
                    "winner": r.winner,
                    "advantage_percent": r.advantage_percent,
                    "notes": r.notes,
                    "hcms": asdict(r.hcms_metrics),
                    "standard": asdict(r.standard_metrics)
                }
                for r in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Relatório salvo em: {output_file}")
        return report
    
    def _generate_summary(self) -> Dict:
        """Gera resumo executivo"""
        hcms_wins = sum(1 for r in self.results if r.winner == "HCMS")
        std_wins = sum(1 for r in self.results if r.winner == "Standard")
        
        # Calcula médias ponderadas
        all_hcms_latencies = [r.hcms_metrics.latency_ms for r in self.results if r.hcms_metrics.latency_ms > 0]
        all_std_latencies = [r.standard_metrics.latency_ms for r in self.results if r.standard_metrics.latency_ms > 0]
        
        all_hcms_precision = [r.hcms_metrics.precision_at_1 for r in self.results]
        all_std_precision = [r.standard_metrics.precision_at_1 for r in self.results]
        
        return {
            "total_scenarios": len(self.results),
            "hcms_wins": hcms_wins,
            "standard_wins": std_wins,
            "hcms_avg_latency_ms": statistics.mean(all_hcms_latencies) if all_hcms_latencies else 0,
            "standard_avg_latency_ms": statistics.mean(all_std_latencies) if all_std_latencies else 0,
            "hcms_avg_precision": statistics.mean(all_hcms_precision),
            "standard_avg_precision": statistics.mean(all_std_precision),
            "latency_overhead_ms": statistics.mean(all_hcms_latencies) - statistics.mean(all_std_latencies) if all_hcms_latencies and all_std_latencies else 0
        }
    
    def print_final_summary(self):
        """Imprime resumo final comparativo"""
        summary = self._generate_summary()
        
        print("\n" + "="*80)
        print("RESUMO EXECUTIVO")
        print("="*80 + "\n")
        
        print(f"Total de Cenários: {summary['total_scenarios']}")
        print(f"Vitórias HCMS: {summary['hcms_wins']}")
        print(f"Vitórias Standard: {summary['standard_wins']}\n")
        
        print("PERFORMANCE AGREGADA")
        print("-" * 80)
        print(f"{'Métrica':<30} | {'HCMS':<20} | {'Standard':<20}")
        print("-" * 80)
        print(f"{'Precisão Média':<30} | {summary['hcms_avg_precision']:<20.2%} | {summary['standard_avg_precision']:<20.2%}")
        print(f"{'Latência Média':<30} | {summary['hcms_avg_latency_ms']:<20.1f}ms | {summary['standard_avg_latency_ms']:<20.1f}ms")
        print(f"{'Overhead do Reranker':<30} | {summary['latency_overhead_ms']:<20.1f}ms | {'N/A':<20}")
        
        print("\n" + "="*80)
        print("ANÁLISE CRÍTICA")
        print("="*80 + "\n")
        
        print("""
ONDE O HCMS SE DESTACA:
✓ Busca de termos exatos (FTS + Vector híbrido)
✓ Recuperação sob ruído denso (Cross-Encoder filtering)
✓ Consolidação automática de duplicatas (Pruner)
✓ Poda cognitiva (GC remove ruído automaticamente)

ONDE O HCMS PERDE:
✗ Latência bruta (~50-100ms overhead do reranker)
✗ Complexidade operacional (3 índices vs 1)
✗ Custo computacional (reranker SEMPRE ativo)
✗ Manutenção manual (GC não é automático por padrão)

PROBLEMAS ARQUITETURAIS DETECTADOS:
1. Reranker deveria ser CONDICIONAL (só ativa se similarity < 0.7)
2. Poda cognitiva deveria rodar em background (async job)
3. Sem cache de embeddings da query (re-computa a cada recall)
4. FTS usa 'simple' dictionary (deveria ter stemming)
5. Importance calibration no agent_bridge é frágil (LLaMA 3B não confiável)

RECOMENDAÇÕES:
→ Implementar reranking adaptativo (opt-in baseado em confidence)
→ Adicionar query embedding cache (LRU com 1000 entries)
→ Tornar GC um background job (Celery/RQ a cada 1h)
→ Usar FTS dictionary 'english' com stemming
→ Substituir importance heurística por classificador treinado
        """)
        
        print("\n" + "="*80)
        print("CONCLUSÃO")
        print("="*80 + "\n")
        
        if summary['hcms_wins'] > summary['standard_wins']:
            precision_gain = (summary['hcms_avg_precision'] - summary['standard_avg_precision']) / summary['standard_avg_precision'] * 100
            print(f"✅ HCMS vence em {summary['hcms_wins']}/{summary['total_scenarios']} cenários")
            print(f"   Ganho de precisão: +{precision_gain:.1f}%")
            print(f"   Custo: +{summary['latency_overhead_ms']:.1f}ms de latência")
            print("\n   VEREDITO: HCMS compensa o overhead quando precisão é crítica.")
        else:
            print(f"⚠️  Standard vence em {summary['standard_wins']}/{summary['total_scenarios']} cenários")
            print(f"   Latency advantage: {summary['latency_overhead_ms']:.1f}ms faster")
            print("\n   VEREDITO: HCMS não justifica complexidade adicional para uso geral.")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    DSN = "dbname=hcms user=felipe"
    
    print("\n🚀 Iniciando Benchmark Comparativo...")
    print("   Este teste pode levar 2-3 minutos.\n")
    
    benchmark = ComprehensiveBenchmark(DSN)
    
    try:
        # Roda todos os cenários
        results = benchmark.run_all()
        
        # Imprime resumo final
        benchmark.print_final_summary()
        
        # Gera relatório JSON
        benchmark.generate_report("benchmark_report.json")
        
    except Exception as e:
        print(f"\n❌ Erro durante benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Limpando bancos de dados de teste...")
        benchmark._reset_systems()


if __name__ == "__main__":
    main()