"""
BENCHMARK CRÍTICO: Cenários onde HCMS deve vencer para justificar existência

4 Cenários Decisivos:
1. Multi-Document Synthesis (Graph context injection)
2. Temporal Consistency (Tier decay + importance)
3. Noisy Ingest Resilience (GC automático)
4. Adversarial Queries (Cross-encoder + FTS)

REGRA: HCMS deve vencer 3/4 para ser viável.

Executar: python scripts/benchmark_critical.py
"""

import os
import sys
import time
import statistics
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

from hcms.core import HCMSOptimized


# ============================================================
# RAG BASELINE
# ============================================================

class StandardRAG:
    """RAG padrão: Vector search puro, sem features avançadas"""
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self._init_db()
    
    def _init_db(self):
        with psycopg2.connect(self.dsn, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Drop e recria tabela para garantir schema correto
                cur.execute("DROP TABLE IF EXISTS standard_memories CASCADE;")
                
                cur.execute("""
                    CREATE TABLE standard_memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector(384) NOT NULL,
                        created_at DOUBLE PRECISION DEFAULT EXTRACT(EPOCH FROM NOW()),
                        metadata JSONB DEFAULT '{}'::jsonb
                    );
                """)
                cur.execute("""
                    CREATE INDEX idx_standard_vec 
                    ON standard_memories 
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64);
                """)
                conn.commit()
    
    def remember(self, content: str, metadata: dict = None) -> str:
        mem_id = f"std_{int(time.time() * 1000000)}"
        embedding = self.encoder.encode(content).tolist()
        
        with psycopg2.connect(self.dsn, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO standard_memories (id, content, embedding, metadata) VALUES (%s, %s, %s, %s)",
                    (mem_id, content, embedding, json.dumps(metadata or {}))
                )
                conn.commit()
        return mem_id
    
    def recall(self, query: str, limit: int = 5) -> List[Dict]:
        query_emb = self.encoder.encode(query).tolist()
        
        with psycopg2.connect(self.dsn, cursor_factory=RealDictCursor) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, content, metadata,
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
# MÉTRICAS
# ============================================================

@dataclass
class CriticalMetrics:
    """Métricas específicas para cada cenário crítico"""
    system: str
    scenario: str
    
    # Qualidade de resposta
    answer_accuracy: float      # % de respostas corretas
    context_relevance: float    # % de contexto realmente útil
    
    # Robustez
    false_positive_rate: float  # % de ruído retornado
    temporal_precision: float   # % de fatos desatualizados filtrados
    
    # Performance
    latency_ms: float
    throughput_qps: float       # Queries per second
    
    # Custo
    maintenance_overhead: float  # Tempo gasto em GC/pruning


@dataclass  
class CriticalResult:
    """Resultado de um cenário crítico"""
    name: str
    hcms_metrics: CriticalMetrics
    standard_metrics: CriticalMetrics
    winner: str
    margin: float  # Diferença percentual
    justification: str  # Por que esse cenário importa


# ============================================================
# SUITE DE TESTES CRÍTICOS
# ============================================================

class CriticalBenchmark:
    def __init__(self, dsn: str):
        self.dsn = dsn
        self.hcms = HCMSOptimized(dsn, enable_auto_gc=True)
        self.standard = StandardRAG(dsn)
        self.results: List[CriticalResult] = []
    
    def run_all(self) -> List[CriticalResult]:
        """Executa os 4 cenários críticos"""
        print("\n" + "="*80)
        print("BENCHMARK CRÍTICO: HCMS deve vencer 3/4 para ser viável")
        print("="*80 + "\n")
        
        scenarios = [
            ("Multi-Document Synthesis", self._test_multi_doc_synthesis),
            ("Temporal Consistency", self._test_temporal_consistency),
            ("Noisy Ingest Resilience", self._test_noisy_ingest),
            ("Adversarial Queries", self._test_adversarial_queries),
        ]
        
        for name, test_fn in scenarios:
            print(f"\n{'='*80}")
            print(f"CENÁRIO CRÍTICO: {name}")
            print('='*80)
            self._reset_systems()
            result = test_fn()
            self.results.append(result)
            self._print_result(result)
        
        return self.results
    
    def _reset_systems(self):
        """Limpa ambos os sistemas"""
        self.hcms.storage.execute("TRUNCATE memories, edges, access_logs CASCADE;")
        self.standard.clear()
    
    # ============================================================
    # CENÁRIO 1: MULTI-DOCUMENT SYNTHESIS
    # ============================================================
    
    def _test_multi_doc_synthesis(self) -> CriticalResult:
        """
        Testa se graph context injection realmente ajuda em queries
        que precisam cruzar informações de múltiplos documentos.
        
        Dataset: 50 docs sobre 5 empresas tech
        Query: "Compare políticas de privacidade entre Google e Apple"
        
        Expectativa: HCMS recupera AMBOS os docs relevantes via grafo
        """
        
        # Dataset: Fatos distribuídos sobre empresas
        companies = {
            "Google": [
                "Google coleta dados de localização mesmo com GPS desligado.",
                "Google usa dados para treinar modelos de IA sem consentimento explícito.",
                "Google foi multada em €50M por violação GDPR em 2019.",
                "Política de privacidade do Google muda sem notificação clara.",
            ],
            "Apple": [
                "Apple implementou App Tracking Transparency em iOS 14.5.",
                "Apple criptografa mensagens end-to-end no iMessage por padrão.",
                "Apple não vende dados de usuários para terceiros.",
                "Apple permite opt-out de coleta de dados analíticos.",
            ],
            "Meta": [
                "Meta rastreia usuários mesmo fora do Facebook via pixels.",
                "Meta foi multada em €1.2B por transferência ilegal de dados EU-EUA.",
                "Meta usa reconhecimento facial sem consentimento em fotos.",
            ],
            "Microsoft": [
                "Microsoft coleta telemetria do Windows 10 por padrão.",
                "Microsoft oferece certificações ISO 27001 para Azure.",
            ],
            "Amazon": [
                "Amazon Alexa grava conversas e as compartilha com funcionários.",
                "Amazon foi acusada de vender dados de compras para anunciantes.",
            ]
        }
        
        # Ingest com relações explícitas para HCMS
        company_ids = {}
        for company, facts in companies.items():
            for fact in facts:
                hcms_id = self.hcms.remember(
                    fact, 
                    importance=0.8,
                    metadata={"company": company, "type": "privacy_policy"}
                )
                self.standard.remember(fact, metadata={"company": company})
                
                # Cria arestas entre fatos da mesma empresa
                if company not in company_ids:
                    company_ids[company] = []
                company_ids[company].append(hcms_id)
        
        # Conecta fatos relacionados via grafo (HCMS only)
        for company, ids in company_ids.items():
            for i, src_id in enumerate(ids):
                for dst_id in ids[i+1:]:
                    self.hcms.storage.execute(
                        "INSERT INTO edges (src, dst, type, weight) VALUES (%s, %s, 'same_company', 1.0)",
                        (src_id, dst_id)
                    )
        
        # Queries que exigem múltiplos documentos
        test_queries = [
            {
                "query": "Compare políticas de privacidade entre Google e Apple",
                "required_companies": ["Google", "Apple"],
                "min_facts_per_company": 2
            },
            {
                "query": "Quais empresas foram multadas por violação de privacidade?",
                "required_companies": ["Google", "Meta"],
                "min_facts_per_company": 1
            },
            {
                "query": "Diferenças entre coleta de dados da Meta e Microsoft",
                "required_companies": ["Meta", "Microsoft"],
                "min_facts_per_company": 1
            }
        ]
        
        hcms_scores = []
        std_scores = []
        hcms_latencies = []
        std_latencies = []
        
        for test_case in test_queries:
            query = test_case["query"]
            required = test_case["required_companies"]
            min_facts = test_case["min_facts_per_company"]
            
            # HCMS com graph expansion
            start = time.time()
            hcms_res = self.hcms.recall(query, limit=10, expand_graph=True)
            hcms_latencies.append((time.time() - start) * 1000)
            
            # Verifica se recuperou fatos de AMBAS as empresas
            hcms_companies = set()
            for r in hcms_res:
                company = r.get('metadata', {}).get('company')
                if company in required:
                    hcms_companies.add(company)
            
            # Score: 1.0 se recuperou ambas, 0.5 se apenas uma, 0.0 se nenhuma
            if len(hcms_companies) == len(required):
                hcms_scores.append(1.0)
            elif len(hcms_companies) > 0:
                hcms_scores.append(0.5)
            else:
                hcms_scores.append(0.0)
            
            # Standard (sem grafo)
            start = time.time()
            std_res = self.standard.recall(query, limit=10)
            std_latencies.append((time.time() - start) * 1000)
            
            std_companies = set()
            for r in std_res:
                company = r.get('metadata', {}).get('company')
                if company in required:
                    std_companies.add(company)
            
            if len(std_companies) == len(required):
                std_scores.append(1.0)
            elif len(std_companies) > 0:
                std_scores.append(0.5)
            else:
                std_scores.append(0.0)
        
        # Métricas
        hcms_metrics = CriticalMetrics(
            system="HCMS",
            scenario="MultiDocSynthesis",
            answer_accuracy=statistics.mean(hcms_scores),
            context_relevance=statistics.mean(hcms_scores),  # Proxy
            false_positive_rate=0.0,
            temporal_precision=1.0,
            latency_ms=statistics.mean(hcms_latencies),
            throughput_qps=1000 / statistics.mean(hcms_latencies),
            maintenance_overhead=0.0
        )
        
        std_metrics = CriticalMetrics(
            system="Standard",
            scenario="MultiDocSynthesis",
            answer_accuracy=statistics.mean(std_scores),
            context_relevance=statistics.mean(std_scores),
            false_positive_rate=0.0,
            temporal_precision=1.0,
            latency_ms=statistics.mean(std_latencies),
            throughput_qps=1000 / statistics.mean(std_latencies),
            maintenance_overhead=0.0
        )
        
        winner = "HCMS" if hcms_metrics.answer_accuracy > std_metrics.answer_accuracy else "Standard"
        margin = abs(hcms_metrics.answer_accuracy - std_metrics.answer_accuracy) / max(std_metrics.answer_accuracy, 0.01) * 100
        
        return CriticalResult(
            name="Multi-Document Synthesis",
            hcms_metrics=hcms_metrics,
            standard_metrics=std_metrics,
            winner=winner,
            margin=margin,
            justification="Graph context injection deve permitir cruzamento de informações distribuídas"
        )
    
    # ============================================================
    # CENÁRIO 2: TEMPORAL CONSISTENCY
    # ============================================================
    
    def _test_temporal_consistency(self) -> CriticalResult:
        """
        Testa se tier management + importance decay filtram fatos desatualizados.
        
        Dataset: Fatos sobre CEOs que mudaram ao longo do tempo
        Query: "Quem é o CEO atual da empresa X?"
        
        Expectativa: HCMS filtra fatos antigos via decay temporal
        """
        
        # Dataset temporal: CEOs que mudaram
        temporal_facts = [
            # Twitter/X
            ("Jack Dorsey foi CEO do Twitter de 2006 a 2008.", "2008-01-01", 0.3),
            ("Evan Williams foi CEO do Twitter de 2008 a 2010.", "2010-01-01", 0.3),
            ("Dick Costolo foi CEO do Twitter de 2010 a 2015.", "2015-01-01", 0.3),
            ("Jack Dorsey retornou como CEO do Twitter em 2015.", "2021-01-01", 0.5),
            ("Parag Agrawal foi CEO do Twitter de 2021 a 2022.", "2022-10-01", 0.7),
            ("Elon Musk é CEO do X (Twitter) desde outubro de 2022.", datetime.now().strftime("%Y-%m-%d"), 1.0),
            
            # OpenAI
            ("Sam Altman foi CEO da OpenAI desde sua fundação.", "2023-11-01", 0.6),
            ("Mira Murati foi CEO interina da OpenAI em novembro de 2023.", "2023-11-20", 0.8),
            ("Sam Altman retornou como CEO da OpenAI em novembro de 2023.", datetime.now().strftime("%Y-%m-%d"), 1.0),
        ]
        
        # Ingest com timestamps
        now = time.time()
        for fact, date_str, recency_weight in temporal_facts:
            # Calcula age em dias
            fact_date = datetime.strptime(date_str, "%Y-%m-%d")
            age_days = (datetime.now() - fact_date).days
            
            # Timestamp simulado (passado)
            simulated_timestamp = now - (age_days * 86400)
            
            # HCMS: Importance decai com idade
            importance = recency_weight * max(0.1, 1.0 - (age_days / 365.0))
            mem_id = self.hcms.remember(fact, importance=importance)
            
            # Simula envelhecimento via last_access
            self.hcms.storage.execute(
                "UPDATE memories SET last_access = %s, creation_time = %s WHERE id = %s",
                (simulated_timestamp, simulated_timestamp, mem_id)
            )
            
            # Standard: Não tem decay temporal
            self.standard.remember(fact, metadata={"date": date_str})
        
        # Queries sobre estado ATUAL
        test_queries = [
            ("Quem é o CEO atual do Twitter?", "Elon Musk"),
            ("Quem lidera a OpenAI atualmente?", "Sam Altman"),
            ("CEO do X em 2024", "Elon Musk"),
        ]
        
        hcms_scores = []
        std_scores = []
        hcms_latencies = []
        std_latencies = []
        
        for query, expected_ceo in test_queries:
            # HCMS
            start = time.time()
            hcms_res = self.hcms.recall(query, limit=5)
            hcms_latencies.append((time.time() - start) * 1000)
            
            # Verifica se top-1 é o CEO atual
            top_content = hcms_res[0]['content'] if hcms_res else ""
            hcms_correct = 1.0 if expected_ceo.lower() in top_content.lower() else 0.0
            hcms_scores.append(hcms_correct)
            
            # Conta quantos resultados são desatualizados (não contêm CEO atual)
            hcms_outdated = sum(1 for r in hcms_res if expected_ceo.lower() not in r['content'].lower())
            
            # Standard
            start = time.time()
            std_res = self.standard.recall(query, limit=5)
            std_latencies.append((time.time() - start) * 1000)
            
            top_content = std_res[0]['content'] if std_res else ""
            std_correct = 1.0 if expected_ceo.lower() in top_content.lower() else 0.0
            std_scores.append(std_correct)
            
            std_outdated = sum(1 for r in std_res if expected_ceo.lower() not in r['content'].lower())
        
        # Métricas
        hcms_metrics = CriticalMetrics(
            system="HCMS",
            scenario="TemporalConsistency",
            answer_accuracy=statistics.mean(hcms_scores),
            context_relevance=1.0,
            false_positive_rate=0.0,
            temporal_precision=statistics.mean(hcms_scores),  # % de fatos atualizados no top-1
            latency_ms=statistics.mean(hcms_latencies),
            throughput_qps=1000 / statistics.mean(hcms_latencies),
            maintenance_overhead=0.0
        )
        
        std_metrics = CriticalMetrics(
            system="Standard",
            scenario="TemporalConsistency",
            answer_accuracy=statistics.mean(std_scores),
            context_relevance=1.0,
            false_positive_rate=0.0,
            temporal_precision=statistics.mean(std_scores),
            latency_ms=statistics.mean(std_latencies),
            throughput_qps=1000 / statistics.mean(std_latencies),
            maintenance_overhead=0.0
        )
        
        winner = "HCMS" if hcms_metrics.temporal_precision > std_metrics.temporal_precision else "Standard"
        margin = abs(hcms_metrics.temporal_precision - std_metrics.temporal_precision) / max(std_metrics.temporal_precision, 0.01) * 100
        
        return CriticalResult(
            name="Temporal Consistency",
            hcms_metrics=hcms_metrics,
            standard_metrics=std_metrics,
            winner=winner,
            margin=margin,
            justification="Importance decay deve priorizar fatos recentes sobre desatualizados"
        )
    
    # ============================================================
    # CENÁRIO 3: NOISY INGEST RESILIENCE
    # ============================================================
    
    def _test_noisy_ingest(self) -> CriticalResult:
        """
        Simula chat real: 80% mensagens casuais, 20% fatos importantes.
        Roda por 100 ciclos de ingest + recall.
        
        Expectativa: HCMS mantém SNR (signal-to-noise) alto via GC automático
        """
        
        # Dataset: Mix realista de chat
        important_facts = [
            "Senha do servidor de produção: XK-9847-ALPHA",
            "Cliente VIP: João Silva, telefone 11-98765-4321",
            "Bug crítico: Memory leak no módulo de pagamento",
            "Reunião de sprint: Toda segunda às 9h",
            "Endpoint de API: https://api.example.com/v2/users",
        ]
        
        noise_templates = [
            "Bom dia!", "Como vai?", "Tudo bem?", "Ótimo!",
            "Obrigado", "De nada", "Até logo", "Tchau",
            "Legal", "Entendi", "Ok", "Perfeito",
            "Hmm", "Interessante", "Vejo", "Certo",
        ]
        
        # Fase 1: Ingest ruidoso (100 mensagens)
        print("   [Fase 1] Simulando ingest ruidoso...")
        
        for i in range(100):
            if i % 20 == 0:  # 20% fatos importantes
                fact = important_facts[i // 20]
                self.hcms.remember(fact, importance=0.9)
                self.standard.remember(fact)
            else:  # 80% ruído
                noise = f"{noise_templates[i % len(noise_templates)]} #{i}"
                self.hcms.remember(noise, importance=0.1)
                self.standard.remember(noise)
        
        # Simula envelhecimento (7 dias no ruído)
        now = time.time()
        self.hcms.storage.execute("""
            UPDATE memories 
            SET last_access = %s 
            WHERE importance < 0.3
        """, (now - (8 * 86400),))
        
        # GC automático (HCMS only)
        print("   [Fase 2] Executando GC automático...")
        gc_start = time.time()
        self.hcms.sync_access_stats()
        gc_stats = self.hcms.pruner.run_garbage_collection()
        gc_time = (time.time() - gc_start) * 1000
        
        noise_removed = gc_stats['deleted_noise']
        
        # Fase 2: Recall sobre fatos importantes
        print("   [Fase 3] Testando recall pós-GC...")
        
        test_queries = [
            ("senha do servidor", "XK-9847-ALPHA"),
            ("telefone do João", "11-98765-4321"),
            ("bug no pagamento", "memory leak"),
            ("horário da reunião", "9h"),
            ("endpoint da API", "api.example.com"),
        ]
        
        hcms_scores = []
        std_scores = []
        hcms_latencies = []
        std_latencies = []
        hcms_noise_counts = []
        std_noise_counts = []
        
        for query, expected in test_queries:
            # HCMS
            start = time.time()
            hcms_res = self.hcms.recall(query, limit=5)
            hcms_latencies.append((time.time() - start) * 1000)
            
            # Verifica se encontrou o fato
            found = any(expected.lower() in r['content'].lower() for r in hcms_res)
            hcms_scores.append(1.0 if found else 0.0)
            
            # Conta ruído nos resultados
            hcms_noise = sum(1 for r in hcms_res if r.get('importance', 1.0) < 0.3)
            hcms_noise_counts.append(hcms_noise)
            
            # Standard
            start = time.time()
            std_res = self.standard.recall(query, limit=5)
            std_latencies.append((time.time() - start) * 1000)
            
            found = any(expected.lower() in r['content'].lower() for r in std_res)
            std_scores.append(1.0 if found else 0.0)
            
            # Heurística de ruído: mensagens muito curtas
            std_noise = sum(1 for r in std_res if len(r['content']) < 20)
            std_noise_counts.append(std_noise)
        
        # Métricas
        hcms_metrics = CriticalMetrics(
            system="HCMS",
            scenario="NoisyIngest",
            answer_accuracy=statistics.mean(hcms_scores),
            context_relevance=statistics.mean(hcms_scores),
            false_positive_rate=statistics.mean(hcms_noise_counts) / 5.0,  # Normalizado
            temporal_precision=1.0,
            latency_ms=statistics.mean(hcms_latencies),
            throughput_qps=1000 / statistics.mean(hcms_latencies),
            maintenance_overhead=gc_time
        )
        
        std_metrics = CriticalMetrics(
            system="Standard",
            scenario="NoisyIngest",
            answer_accuracy=statistics.mean(std_scores),
            context_relevance=statistics.mean(std_scores),
            false_positive_rate=statistics.mean(std_noise_counts) / 5.0,
            temporal_precision=1.0,
            latency_ms=statistics.mean(std_latencies),
            throughput_qps=1000 / statistics.mean(std_latencies),
            maintenance_overhead=0.0
        )
        
        # Winner baseado em SNR (signal-to-noise ratio)
        hcms_snr = hcms_metrics.answer_accuracy / max(hcms_metrics.false_positive_rate, 0.01)
        std_snr = std_metrics.answer_accuracy / max(std_metrics.false_positive_rate, 0.01)
        
        winner = "HCMS" if hcms_snr > std_snr else "Standard"
        margin = abs(hcms_snr - std_snr) / max(std_snr, 0.01) * 100
        
        return CriticalResult(
            name="Noisy Ingest Resilience",
            hcms_metrics=hcms_metrics,
            standard_metrics=std_metrics,
            winner=winner,
            margin=margin,
            justification=f"GC removeu {noise_removed} entradas de ruído. SNR: HCMS={hcms_snr:.2f}, Std={std_snr:.2f}"
        )
    
    # ============================================================
    # CENÁRIO 4: ADVERSARIAL QUERIES
    # ============================================================
    
    def _test_adversarial_queries(self) -> CriticalResult:
        """
        Queries projetadas para confundir embeddings:
        - Homônimos (banco = instituição vs móvel)
        - Paráfrases extremas
        - Negações
        - Queries ambíguas
        
        Expectativa: Cross-encoder + FTS são mais robustos
        """
        
        # Dataset adversarial
        adversarial_docs = [
            # Homônimos
            ("O banco Santander anunciou lucros recordes.", "finance"),
            ("Sentei no banco da praça para ler.", "furniture"),
            ("O banco de dados PostgreSQL é relacional.", "tech"),
            
            # Paráfrases extremas
            ("Python é uma linguagem de programação criada por Guido van Rossum.", "tech"),
            ("O código foi escrito na linguagem criada pelo holandês em 1991.", "tech"),
            
            # Negações
            ("A reunião NÃO foi cancelada.", "meeting_confirmed"),
            ("A reunião foi cancelada.", "meeting_cancelled"),
            
            # Ambiguidade temporal
            ("O projeto foi concluído em 2023.", "past"),
            ("O projeto será concluído em 2025.", "future"),
            ("O projeto está em andamento.", "present"),
        ]
        
        for doc, category in adversarial_docs:
            self.hcms.remember(doc, importance=0.8, metadata={"category": category})
            self.standard.remember(doc, metadata={"category": category})
        
        # Queries adversariais
        test_cases = [
            # Teste 1: Homônimo com contexto
            {
                "query": "banco de dados relacional",
                "expected_category": "tech",
                "adversarial_type": "homonym"
            },
            # Teste 2: Paráfrase extrema
            {
                "query": "criador do Python",
                "expected_category": "tech",
                "adversarial_type": "paraphrase"
            },
            # Teste 3: Negação
            {
                "query": "a reunião vai acontecer?",
                "expected_category": "meeting_confirmed",
                "adversarial_type": "negation"
            },
            # Teste 4: Temporal
            {
                "query": "status atual do projeto",
                "expected_category": "present",
                "adversarial_type": "temporal"
            },
        ]
        
        hcms_scores = []
        std_scores = []
        hcms_latencies = []
        std_latencies = []
        
        for test_case in test_cases:
            query = test_case["query"]
            expected = test_case["expected_category"]
            
            # HCMS (com cross-encoder + FTS)
            start = time.time()
            hcms_res = self.hcms.recall(query, limit=3, force_rerank=True)
            hcms_latencies.append((time.time() - start) * 1000)
            
            # Verifica se top-1 é a categoria correta
            top_category = hcms_res[0]['metadata'].get('category') if hcms_res else None
            hcms_correct = 1.0 if top_category == expected else 0.0
            hcms_scores.append(hcms_correct)
            
            # Standard (vector puro)
            start = time.time()
            std_res = self.standard.recall(query, limit=3)
            std_latencies.append((time.time() - start) * 1000)
            
            top_category = std_res[0]['metadata'].get('category') if std_res else None
            std_correct = 1.0 if top_category == expected else 0.0
            std_scores.append(std_correct)
        
        # Métricas
        hcms_metrics = CriticalMetrics(
            system="HCMS",
            scenario="AdversarialQueries",
            answer_accuracy=statistics.mean(hcms_scores),
            context_relevance=statistics.mean(hcms_scores),
            false_positive_rate=1.0 - statistics.mean(hcms_scores),
            temporal_precision=1.0,
            latency_ms=statistics.mean(hcms_latencies),
            throughput_qps=1000 / statistics.mean(hcms_latencies),
            maintenance_overhead=0.0
        )
        
        std_metrics = CriticalMetrics(
            system="Standard",
            scenario="AdversarialQueries",
            answer_accuracy=statistics.mean(std_scores),
            context_relevance=statistics.mean(std_scores),
            false_positive_rate=1.0 - statistics.mean(std_scores),
            temporal_precision=1.0,
            latency_ms=statistics.mean(std_latencies),
            throughput_qps=1000 / statistics.mean(std_latencies),
            maintenance_overhead=0.0
        )
        
        winner = "HCMS" if hcms_metrics.answer_accuracy > std_metrics.answer_accuracy else "Standard"
        margin = abs(hcms_metrics.answer_accuracy - std_metrics.answer_accuracy) / max(std_metrics.answer_accuracy, 0.01) * 100
        
        return CriticalResult(
            name="Adversarial Queries",
            hcms_metrics=hcms_metrics,
            standard_metrics=std_metrics,
            winner=winner,
            margin=margin,
            justification="Cross-encoder + FTS devem superar embeddings puros em queries ambíguas"
        )
    
    # ============================================================
    # RELATÓRIO
    # ============================================================
    
    def _print_result(self, result: CriticalResult):
        """Imprime resultado do cenário"""
        print(f"\n{'='*80}")
        print(f"RESULTADO: {result.name}")
        print('='*80)
        print(f"\nVencedor: {result.winner} (margem: {result.margin:.1f}%)")
        print(f"Justificativa: {result.justification}\n")
        
        print(f"{'Métrica':<30} | {'HCMS':<15} | {'Standard':<15}")
        print("-" * 62)
        
        metrics = [
            ("Answer Accuracy", "answer_accuracy", "{:.1%}"),
            ("Context Relevance", "context_relevance", "{:.1%}"),
            ("False Positive Rate", "false_positive_rate", "{:.1%}"),
            ("Temporal Precision", "temporal_precision", "{:.1%}"),
            ("Latency (avg)", "latency_ms", "{:.1f}ms"),
            ("Throughput", "throughput_qps", "{:.1f} qps"),
            ("Maintenance Overhead", "maintenance_overhead", "{:.1f}ms"),
        ]
        
        for label, attr, fmt in metrics:
            hcms_val = getattr(result.hcms_metrics, attr)
            std_val = getattr(result.standard_metrics, attr)
            
            hcms_str = fmt.format(hcms_val)
            std_str = fmt.format(std_val)
            
            print(f"{label:<30} | {hcms_str:<15} | {std_str:<15}")
    
    def print_final_verdict(self):
        """Veredicto final: HCMS é viável?"""
        print("\n" + "="*80)
        print("VEREDICTO FINAL: HCMS É VIÁVEL?")
        print("="*80 + "\n")
        
        hcms_wins = sum(1 for r in self.results if r.winner == "HCMS")
        std_wins = sum(1 for r in self.results if r.winner == "Standard")
        
        print(f"Resultados: HCMS venceu {hcms_wins}/4 cenários críticos\n")
        
        for r in self.results:
            symbol = "✅" if r.winner == "HCMS" else "❌"
            print(f"{symbol} {r.name}: {r.winner} (margem: {r.margin:.1f}%)")
        
        print("\n" + "="*80)
        
        if hcms_wins >= 3:
            print("✅ PROJETO VIÁVEL")
            print("="*80)
            print("""
HCMS demonstrou superioridade em 3+ cenários críticos.
A complexidade adicional se justifica pelos ganhos:

PRÓXIMOS PASSOS:
1. Otimize os cenários onde perdeu
2. Implemente reranking condicional agressivo (threshold=0.8)
3. Profile latência para reduzir overhead
4. Considere deployment em produção (staging primeiro)

RECOMENDAÇÃO: Continue o desenvolvimento.
            """)
        else:
            print("❌ PROJETO INVIÁVEL")
            print("="*80)
            print(f"""
HCMS venceu apenas {hcms_wins}/4 cenários críticos.
Standard RAG é mais eficiente na maioria dos casos.

ANÁLISE CRÍTICA:
- Complexidade operacional (3 índices, GC manual) não compensa
- Latência 2x maior sem ganho proporcional de qualidade
- Features avançadas (grafo, tiers) não demonstraram valor

RECOMENDAÇÕES:
1. ABANDONE o projeto atual
2. Extraia apenas: reranking condicional + FTS híbrido
3. Simplifique para RAG++ (1 índice vetorial + 1 FTS)
4. Foque em otimizar o que já funciona (embeddings, chunking)

VEREDICTO: Não justifica produção. Use para aprendizado.
            """)
        
        # Salva relatório
        self._save_report()
    
    def _save_report(self):
        """Salva relatório JSON"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "verdict": "VIABLE" if sum(1 for r in self.results if r.winner == "HCMS") >= 3 else "NOT_VIABLE",
            "hcms_wins": sum(1 for r in self.results if r.winner == "HCMS"),
            "standard_wins": sum(1 for r in self.results if r.winner == "Standard"),
            "scenarios": [
                {
                    "name": r.name,
                    "winner": r.winner,
                    "margin": r.margin,
                    "justification": r.justification,
                    "hcms": {
                        "answer_accuracy": r.hcms_metrics.answer_accuracy,
                        "latency_ms": r.hcms_metrics.latency_ms,
                        "false_positive_rate": r.hcms_metrics.false_positive_rate,
                    },
                    "standard": {
                        "answer_accuracy": r.standard_metrics.answer_accuracy,
                        "latency_ms": r.standard_metrics.latency_ms,
                        "false_positive_rate": r.standard_metrics.false_positive_rate,
                    }
                }
                for r in self.results
            ]
        }
        
        with open("benchmark_critical_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Relatório salvo: benchmark_critical_report.json")


# ============================================================
# EXECUÇÃO
# ============================================================

def main():
    DSN = "dbname=hcms user=felipe"
    
    print("\n🔥 BENCHMARK CRÍTICO: HCMS deve vencer 3/4 para ser viável")
    print("   Este teste levará ~5 minutos.\n")
    
    benchmark = CriticalBenchmark(DSN)
    
    try:
        results = benchmark.run_all()
        benchmark.print_final_verdict()
        
    except Exception as e:
        print(f"\n❌ Erro crítico: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🧹 Limpando dados de teste...")
        benchmark._reset_systems()


if __name__ == "__main__":
    main()