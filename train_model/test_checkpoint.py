# test_checkpoint.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Carregando checkpoint-300...")

model = AutoModelForCausalLM.from_pretrained(
    "hcms_personal_llm/checkpoint-300",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    "hcms_personal_llm/checkpoint-300"
)

# Teste 1: Factual lookup
print("\n" + "="*80)
print("TESTE 1: Busca factual")
print("="*80)

messages = [
    {
        "role": "system",
        "content": "Você é um assistente pessoal com memória persistente.\n\nMemórias disponíveis:\n[Memória 1 | 2024-01-15] A senha do Wi-Fi é Secure@2024\n[Memória 2 | 2024-01-10] O escritório fica na Rua das Flores"
    },
    {"role": "user", "content": "Qual a senha do Wi-Fi?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print(f"\nPergunta: Qual a senha do Wi-Fi?")
print(f"Resposta: {response}")
print(f"✓ Esperado: Mencionar 'Secure@2024'")

# Teste 2: Informação ausente
print("\n" + "="*80)
print("TESTE 2: Informação ausente")
print("="*80)

messages = [
    {
        "role": "system",
        "content": "Você é um assistente pessoal com memória persistente.\n\nMemórias disponíveis:\n[Memória 1 | 2024-01-15] Reunião com cliente ABC às 14h"
    },
    {"role": "user", "content": "Qual o telefone do cliente ABC?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print(f"\nPergunta: Qual o telefone do cliente ABC?")
print(f"Resposta: {response}")
print(f"✓ Esperado: Dizer que não sabe/não tem a informação")

# Teste 3: Memórias conflitantes
print("\n" + "="*80)
print("TESTE 3: Memórias conflitantes")
print("="*80)

messages = [
    {
        "role": "system",
        "content": "Você é um assistente pessoal com memória persistente.\n\nMemórias disponíveis:\n[Memória 1 | 2024-01-20] Meu email agora é novo@empresa.com\n[Memória 2 | 2023-12-10] Meu email é antigo@empresa.com"
    },
    {"role": "user", "content": "Qual meu email?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print(f"\nPergunta: Qual meu email?")
print(f"Resposta: {response}")
print(f"✓ Esperado: Usar 'novo@empresa.com' (memória mais recente)")