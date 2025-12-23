import os
import sys


# Setup de path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import time
import statistics
from hcms.core import RAGCore

def run_benchmark():
    DSN = "dbname=hcms user=felipe"
    rag_plus = RAGCore(DSN)

    print("📥 Resetando e Ingerindo dados de stress...")
    rag_plus.storage.execute("TRUNCATE memories CASCADE;")
    
    # Dataset desenhado para confundir vetores
    test_data = [
        ("O código de segurança é 99-X-1234", {"id": "target_1"}),
        ("Norma técnica sobre códigos de segurança v1", {"id": "noise"}),
        ("O banco central elevou a taxa de juros para conter a inflação", {"id": "target_2"}),
        ("Relatório financeiro sobre taxas e juros bancários", {"id": "noise"}),
        ("Sentei no banco de madeira da praça", {"id": "target_3"}),
        ("O banco de investimentos faliu", {"id": "noise"}),
        ("Python é uma linguagem de programação", {"id": "target_4"}),
        ("A cobra python é um réptil da família Pythonidae", {"id": "noise"}),
    ]
    for content, meta in test_data:
        rag_plus.remember(content, metadata=meta)

    # Queries e IDs esperados no Rank 1
    queries = [
        ("Qual o código 99-X-1234?", "target_1"), # FTS Test
        ("inflação e taxa de juros banco central", "target_2"), # Semantic Precision
        ("sentar no banco da praça", "target_3"), # Disambiguation 1
        ("programação em python", "target_4"), # Disambiguation 2
    ]

    print(f"\n{'Query':<35} | {'RAG++ Rank 1?':<15} | {'Latência':<10}")
    print("-" * 70)

    for q, target_id in queries:
        t0 = time.time()
        results = rag_plus.recall(q, limit=1)
        lat = (time.time() - t0) * 1000
        
        # Validação correta via metadata ID
        success = False
        if results and results[0]['metadata'].get('id') == target_id:
            success = True
        
        status = "✅ SIM" if success else "❌ NÃO"
        print(f"{q:<35} | {status:<15} | {lat:.2f}ms")

if __name__ == "__main__":
    run_benchmark()