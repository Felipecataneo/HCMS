import time
import os
import sys

# Ajuste de Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from hcms.core import HCMS


hcms = HCMS("dbname=hcms user=felipe")

def test_pruning():
    print("--- TESTE DE PODA COGNITIVA ---")
    
    # 1. Ingestão de Ruído
    noise_id = hcms.remember("Olá, como vai você?", importance=0.1)
    
    # 2. Ingestão de Fato Crítico
    fact_id = hcms.remember("A senha do cofre é 12345.", importance=1.0)
    
    # 3. Simular que o tempo passou (7 dias para o ruído)
    hcms.storage.execute(
        "UPDATE memories SET last_access = %s WHERE id = %s", 
        (time.time() - (8 * 86400), noise_id)
    )
    
    # 4. Executar Manutenção
    stats = hcms.maintenance()
    print(f"Stats da Limpeza: {stats}")
    
    # 5. Verificar se o ruído sumiu e o fato ficou
    results = hcms.storage.fetch_all("SELECT id FROM memories WHERE id = ANY(%s)", ([noise_id, fact_id],))
    remaining_ids = [r['id'] for r in results]
    
    if noise_id not in remaining_ids and fact_id in remaining_ids:
        print("✅ SUCESSO: O ruído foi esquecido, o fato crítico foi preservado.")
    else:
        print("❌ FALHA: A lógica de esquecimento falhou.")

if __name__ == "__main__":
    test_pruning()