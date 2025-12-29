# test_dataset_format.py
"""
Valida se o dataset está compatível com PyArrow ANTES de gastar horas gerando
"""

import json
from pathlib import Path
from datasets import load_dataset

def create_test_examples():
    """Cria 5 exemplos de teste no formato correto"""
    
    examples = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um assistente pessoal com memória persistente.\n\nMemórias disponíveis:\n[Memória 1 | 2024-01-15] A senha do Wi-Fi é Secure@2024\n[Memória 2 | 2024-01-10] O escritório fica na Rua das Flores"
                },
                {"role": "user", "content": "Qual a senha do Wi-Fi?"},
                {"role": "assistant", "content": "A senha do Wi-Fi é Secure@2024."}
            ],
            "metadata": {"type": "factual_lookup"}
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um assistente pessoal com memória persistente.\n\nMemórias disponíveis:\n[Memória 1 | 2024-01-20] Meu email agora é novo@empresa.com\n[Memória 2 | 2023-12-10] Meu email é antigo@empresa.com"
                },
                {"role": "user", "content": "Qual meu email?"},
                {"role": "assistant", "content": "Seu email atual é novo@empresa.com (atualizado em janeiro de 2024)."}
            ],
            "metadata": {"type": "conflicting_memory"}
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um assistente pessoal com memória persistente.\n\nMemórias disponíveis:\n[Memória 1 | 2024-01-15] Reunião cancelada\n[Memória 2 | 2024-01-14] Reunião com fornecedor às 14h sala 3"
                },
                {"role": "user", "content": "Onde é a reunião?"},
                {"role": "assistant", "content": "A reunião com o fornecedor será às 14h na sala 3."}
            ],
            "metadata": {"type": "noisy_context"}
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um assistente pessoal com memória persistente.\n\nMemórias disponíveis:\n[Memória 1 | 2024-01-15] Reunião com cliente ABC às 14h"
                },
                {"role": "user", "content": "Qual o telefone do cliente ABC?"},
                {"role": "assistant", "content": "Não tenho o telefone do cliente ABC registrado nas minhas memórias."}
            ],
            "metadata": {"type": "missing_information"}
        },
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um assistente pessoal com memória persistente.\n\nMemórias disponíveis:\n[Memória 1 | 2024-01-10] João trabalha no TI\n[Memória 2 | 2024-01-12] Ramal do TI é 5042"
                },
                {"role": "user", "content": "Qual o ramal do João?"},
                {"role": "assistant", "content": "O João trabalha no TI, cujo ramal é 5042."}
            ],
            "metadata": {"type": "synthesis"}
        }
    ]
    
    return examples

def test_pyarrow_compatibility():
    """Testa se o formato é compatível com PyArrow"""
    
    test_file = Path("training_data/test_format.jsonl")
    test_file.parent.mkdir(exist_ok=True)
    
    print("="*80)
    print("TESTE DE COMPATIBILIDADE PYARROW")
    print("="*80)
    
    # Cria exemplos de teste
    examples = create_test_examples()
    
    print(f"\n✓ Criando {len(examples)} exemplos de teste...")
    
    # Salva JSONL
    with open(test_file, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    print(f"✓ Arquivo salvo: {test_file}")
    print(f"✓ Tamanho: {test_file.stat().st_size} bytes")
    
    # Testa carregamento com datasets
    print("\n" + "="*80)
    print("CARREGANDO COM DATASETS LIBRARY")
    print("="*80)
    
    try:
        dataset = load_dataset("json", data_files=str(test_file), split="train")
        
        print(f"\n✅ SUCESSO!")
        print(f"   Exemplos carregados: {len(dataset)}")
        print(f"   Colunas: {dataset.column_names}")
        print(f"   Features: {dataset.features}")
        
        # Mostra primeiro exemplo
        print("\n📋 Primeiro exemplo:")
        first = dataset[0]
        print(f"   Messages: {len(first['messages'])}")
        print(f"   Metadata: {first.get('metadata', {})}")
        
        # Valida estrutura de cada exemplo
        print("\n🔍 Validando estrutura de todos exemplos...")
        for i, ex in enumerate(dataset):
            assert isinstance(ex['messages'], list), f"Exemplo {i}: messages não é lista"
            assert len(ex['messages']) >= 3, f"Exemplo {i}: < 3 mensagens"
            
            for msg in ex['messages']:
                assert 'role' in msg, f"Exemplo {i}: falta role"
                assert 'content' in msg, f"Exemplo {i}: falta content"
                assert isinstance(msg['content'], str), f"Exemplo {i}: content não é string"
        
        print("✅ Todos exemplos válidos!")
        
        # Testa formatação para chat template
        print("\n🔧 Testando formatação para treinamento...")
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
        
        # Formata primeiro exemplo
        formatted = tokenizer.apply_chat_template(
            first['messages'],
            tokenize=False,
            add_generation_prompt=False
        )
        
        print("✅ Chat template aplicado com sucesso!")
        print(f"   Tamanho formatado: {len(formatted)} chars")
        
        print("\n" + "="*80)
        print("✅ FORMATO VÁLIDO - PODE PROSSEGUIR COM GERAÇÃO")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERRO AO CARREGAR DATASET:")
        print(f"   {type(e).__name__}: {e}")
        print("\n⚠️ FORMATO INVÁLIDO - CORRIJA ANTES DE GERAR")
        return False

if __name__ == "__main__":
    success = test_pyarrow_compatibility()
    exit(0 if success else 1)