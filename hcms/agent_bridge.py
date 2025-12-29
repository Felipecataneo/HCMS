# agent_bridge.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import re

class LlamaNotebookBridge:
    def __init__(self, core_instance, model_path="train_model/hcms_personal_llm/checkpoint-500"):
        self.core = core_instance
        
        print(f"🔄 Carregando modelo RAG-aware de: {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        print("✅ Modelo carregado com sucesso")
        
    def chat(self, user_input: str) -> str:
        """Chat usando memórias do RAG"""
        # Busca memórias relevantes
        context_docs = self.core.recall(user_input, limit=5)
        
        # Formata memórias no estilo do treinamento
        if context_docs:
            memories_text = "\n".join([
                f"[Memória {i+1} | {self._format_timestamp(c.get('creation_time', 0))}] {c['content']}"
                for i, c in enumerate(context_docs)
            ])
        else:
            memories_text = "[Nenhuma memória relevante encontrada]"
        
        # Monta mensagens no formato treinado
        messages = [
            {
                "role": "system",
                "content": f"""Você é um assistente pessoal com memória persistente.

REGRAS IMPORTANTES:
- Use apenas informações presentes nas memórias
- Priorize memórias mais recentes se houver conflito
- Ignore memórias irrelevantes
- Nunca invente dados
- Se a informação não existir, diga explicitamente que não sabe

Memórias disponíveis:
{memories_text}"""
            },
            {"role": "user", "content": user_input}
        ]
        
        # Gera resposta
        return self._generate(messages, max_new_tokens=200, temperature=0.3)

    def analyze_and_remember(self, user_input: str):
        """Extrai fatos e salva no banco"""
        # Sistema especializado em extração
        messages = [
            {
                "role": "system",
                "content": """Você é um extrator de informações. Analise a mensagem do usuário e:

1. Se contiver informação factual (senha, código, data, endereço, contato, preferência), extraia
2. Responda APENAS com JSON: {"fact": "informação extraída", "importance": 0.0-1.0, "permanent": true/false}
3. Se não houver nada relevante, responda: {}

Exemplos:
User: A senha do Wi-Fi é Secure@2024
Response: {"fact": "A senha do Wi-Fi é Secure@2024", "importance": 0.9, "permanent": true}

User: Oi, tudo bem?
Response: {}"""
            },
            {"role": "user", "content": user_input}
        ]
        
        raw_output = self._generate(messages, max_new_tokens=100, temperature=0.0)
        
        try:
            # Extrai JSON da resposta
            json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                fact = data.get("fact")
                
                if fact:
                    importance = data.get("importance", 0.8)
                    is_perm = data.get("permanent", True)
                    self.core.remember(
                        content=fact, 
                        importance=importance, 
                        is_permanent=is_perm
                    )
                    print(f"✅ Memória salva: {fact}")
            else:
                print("🍃 Nada relevante para memorizar")
                
        except Exception as e:
            print(f"⚠️ Erro ao processar extração: {e} | Output: {raw_output}")

    def _generate(self, messages, max_new_tokens, temperature):
        """Gera resposta usando o modelo fine-tunado"""
        # Aplica chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokeniza
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Gera
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodifica apenas a nova parte
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _format_timestamp(self, timestamp):
        """Converte timestamp Unix para data legível"""
        from datetime import datetime
        try:
            dt = datetime.fromtimestamp(float(timestamp))
            return dt.strftime("%Y-%m-%d")
        except:
            return "data desconhecida"