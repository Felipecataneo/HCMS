# scripts/clean_invalid_json.py
"""
Remove linhas com JSON inválido do dataset
"""

import json
from pathlib import Path
from typing import Dict, List

INPUT_FILE = Path("training_data/synthetic_llm_dataset.jsonl")
OUTPUT_FILE = Path("training_data/synthetic_llm_dataset_clean.jsonl")
INVALID_LOG = Path("training_data/invalid_entries.jsonl")

def validate_entry(entry: Dict) -> tuple[bool, str]:
    """
    Valida estrutura do exemplo de treino
    Retorna: (is_valid, reason)
    """
    try:
        # Verifica campo messages
        if "messages" not in entry:
            return False, "missing_messages_field"
        
        messages = entry["messages"]
        if not isinstance(messages, list):
            return False, "messages_not_list"
        
        if len(messages) < 3:
            return False, f"too_few_messages_{len(messages)}"
        
        # Verifica roles necessárias
        roles = [msg.get("role", "").lower() for msg in messages]
        required_roles = ["system", "user", "assistant"]
        
        for role in required_roles:
            if role not in roles:
                return False, f"missing_role_{role}"
        
        # Verifica que cada mensagem tem content
        for idx, msg in enumerate(messages):
            if "role" not in msg:
                return False, f"message_{idx}_missing_role"
            
            if "content" not in msg:
                return False, f"message_{idx}_missing_content"
            
            if not isinstance(msg["content"], str):
                return False, f"message_{idx}_content_not_string"
            
            if len(msg["content"].strip()) == 0:
                return False, f"message_{idx}_empty_content"
        
        # Verifica metadata (opcional mas recomendado)
        if "metadata" in entry:
            metadata = entry["metadata"]
            if not isinstance(metadata, dict):
                return False, "metadata_not_dict"
        
        return True, "valid"
        
    except Exception as e:
        return False, f"exception_{str(e)[:50]}"


def clean_dataset():
    """Remove entradas inválidas e gera relatório"""
    
    if not INPUT_FILE.exists():
        print(f"❌ Arquivo não encontrado: {INPUT_FILE}")
        return
    
    print(f"📂 Processando: {INPUT_FILE}")
    print(f"📊 Tamanho original: {INPUT_FILE.stat().st_size / 1024:.1f} KB\n")
    
    valid_entries: List[Dict] = []
    invalid_entries: List[Dict] = []
    
    total_lines = 0
    parse_errors = 0
    validation_errors = {}
    
    # Processa linha por linha
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()
            
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                
                # Valida estrutura
                is_valid, reason = validate_entry(entry)
                
                if is_valid:
                    valid_entries.append(entry)
                else:
                    validation_errors[reason] = validation_errors.get(reason, 0) + 1
                    invalid_entries.append({
                        "line": line_num,
                        "reason": reason,
                        "data": entry
                    })
                    
            except json.JSONDecodeError as e:
                parse_errors += 1
                invalid_entries.append({
                    "line": line_num,
                    "reason": "json_parse_error",
                    "error": str(e),
                    "data": line[:200]  # Primeiros 200 chars
                })
    
    # Salva dataset limpo
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in valid_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Salva log de inválidos
    with open(INVALID_LOG, 'w', encoding='utf-8') as f:
        for entry in invalid_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Relatório
    print("="*80)
    print("RELATÓRIO DE LIMPEZA")
    print("="*80)
    
    print(f"\n📊 Estatísticas:")
    print(f"   Total de linhas: {total_lines}")
    print(f"   ✅ Válidas: {len(valid_entries)} ({len(valid_entries)/total_lines*100:.1f}%)")
    print(f"   ❌ Inválidas: {len(invalid_entries)} ({len(invalid_entries)/total_lines*100:.1f}%)")
    print(f"   🔧 Erros de parse JSON: {parse_errors}")
    
    if validation_errors:
        print(f"\n📋 Erros de validação:")
        for reason, count in sorted(validation_errors.items(), key=lambda x: -x[1]):
            print(f"   {reason:40} {count:4} exemplos")
    
    print(f"\n💾 Arquivos gerados:")
    print(f"   Dataset limpo: {OUTPUT_FILE}")
    print(f"   Tamanho: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print(f"   Log de inválidos: {INVALID_LOG}")
    print(f"   Total removido: {len(invalid_entries)} entradas")
    
    # Taxa de sucesso
    success_rate = len(valid_entries) / total_lines * 100 if total_lines > 0 else 0
    
    if success_rate < 80:
        print(f"\n⚠️ ATENÇÃO: Taxa de sucesso baixa ({success_rate:.1f}%)")
        print(f"   Verifique os erros em {INVALID_LOG}")
    elif success_rate < 95:
        print(f"\n⚡ Taxa de sucesso aceitável: {success_rate:.1f}%")
    else:
        print(f"\n✅ Excelente taxa de sucesso: {success_rate:.1f}%")
    
    # Distribuição por tipo (se houver metadata)
    from collections import Counter
    types = Counter([
        entry.get("metadata", {}).get("type", "unknown") 
        for entry in valid_entries
    ])
    
    if types and "unknown" not in types:
        print(f"\n📈 Distribuição por tipo:")
        for scenario_type, count in sorted(types.items()):
            percentage = count / len(valid_entries) * 100
            bar = "█" * (count // 5) + "░" * (20 - count // 5)
            print(f"   {scenario_type:25} [{bar}] {count:3} ({percentage:5.1f}%)")


if __name__ == "__main__":
    print("="*80)
    print("LIMPEZA DE DATASET - REMOÇÃO DE JSON INVÁLIDO")
    print("="*80)
    print()
    
    clean_dataset()
    
    print("\n" + "="*80)
    print("✅ Processo concluído")
    print("="*80)