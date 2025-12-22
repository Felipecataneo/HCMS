import os
import sys
import time

# Ajuste de Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from hcms.core import HCMS

DSN = "dbname=hcms user=felipe"

def run_test_v2():
    print("--- INICIANDO TESTE HCMS V2 (HYBRID + RERANK) ---")
    
    # 1. Inicialização
    hcms = HCMS(DSN)
    
    # Limpeza para teste limpo (Opcional - Cuidado em produção)
    hcms.storage.execute("TRUNCATE memories, edges, access_logs CASCADE;")

    # 2. Ingestão com Importância
    print("\n[1/4] Ingestão de conhecimento...")
    # Um fato genérico
    hcms.remember("A Torre Eiffel é um monumento de ferro em Paris.", importance=0.5)
    
    # Um fato específico com termo difícil para vetores (ID numérico)
    secret_id = hcms.remember("O protocolo de segurança alfa-beta usa o código 9999.", importance=1.0)
    
    # Um fato relacionado que deve ser injetado como contexto
    hcms.remember("O código 9999 deve ser trocado a cada 24 horas.", relations=[(secret_id, "rules")])

    # 3. Teste de Busca Híbrida (Termo exato)
    print("\n[2/4] Testando Busca Híbrida (FTS + Vector)...")
    # Buscamos por um termo exato que o vetor costuma falhar, mas o FTS brilha.
    results = hcms.recall("código 9999", limit=1)
    
    if results and "9999" in results[0]['content']:
        print(f"✅ SUCESSO: Busca Híbrida encontrou o termo exato.")
    else:
        print(f"❌ FALHA: Busca Híbrida não localizou o código 9999.")

    # 4. Teste de Injeção de Contexto (1-Hop)
    print("\n[3/4] Testando Injeção de Contexto (Edges)...")
    if results and 'context_edges' in results[0] and len(results[0]['context_edges']) > 0:
        print(f"✅ SUCESSO: Contexto injetado. Vizinho: {results[0]['context_edges'][0]['content'][:40]}...")
    else:
        print(f"❌ FALHA: Nenhum contexto relacionado foi injetado.")

    # 5. Teste de Reranking
    print("\n[4/4] Validando Reranking...")
    # Ingestão de ruído (vários documentos sobre códigos que não são o 9999)
    for i in range(10):
        hcms.remember(f"Código de erro aleatório {i*100} detectado no sistema.")
    
    # O Reranker deve manter o código 9999 no topo apesar do ruído
    final_results = hcms.recall("qual o código de segurança?", limit=3)
    
    if "9999" in final_results[0]['content']:
        print(f"✅ SUCESSO: Reranker priorizou a informação correta.")
        print(f"Score do Reranker: {final_results[0].get('rerank_score'):.4f}")
    else:
        print(f"❌ FALHA: Reranker perdeu a relevância para o ruído.")

if __name__ == "__main__":
    run_test_v2()