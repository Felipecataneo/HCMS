import requests
import json
import re

class HCMSAgentBridge:
    def __init__(self, core_instance, model="llama3.2:3b"):
        self.core = core_instance
        self.model = model
        self.ollama_url = "http://localhost:11434/api/chat"

    def chat(self, user_input: str) -> str:
        # 1. Recall de contexto (CD-RAG v3)
        context_docs = self.core.recall(user_input, limit=5)
        context_str = "\n".join([f"- {c['content']}" for c in context_docs])

        # 2. Gerar Resposta do Agente
        system_prompt = (
            "Você é um assistente técnico com memória de longo prazo. "
            "Use o contexto fornecido para responder. Se não souber, diga que não tem essa memória."
        )
        
        prompt = f"Contexto:\n{context_str}\n\nPergunta: {user_input}\nResponda de forma direta."
        response = self._call_ollama([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])

        # 3. EXTRAÇÃO DE MEMÓRIA (O segredo para a barra lateral aparecer)
        # Pedimos ao LLM para avaliar se o input do usuário contém um fato novo
        self._analyze_and_remember(user_input)

        return response

    def _analyze_and_remember(self, user_input: str):
        """Usa o LLM para extrair fatos e definir importância automaticamente"""
        
        # Se a frase for muito curta, nem gasta processamento
        if len(user_input.split()) < 3:
            return

        mem_prompt = (
            "Analise a mensagem do usuário abaixo. Se ela contiver uma informação, fato, "
            "configuração ou instrução que valha a pena lembrar, extraia apenas o fato "
            "de forma concisa e atribua uma importância de 0.0 a 1.0.\n"
            "Responda EXATAMENTE no formato: Fato: [texto] | Score: [valor]\n"
            "Se não houver nada útil, responda: IGNORAR\n\n"
            f"Mensagem: {user_input}"
        )

        analysis = self._call_ollama([{"role": "user", "content": mem_prompt}])

        if "Fato:" in analysis and "|" in analysis:
            try:
                # Extrai o fato e o score usando Regex ou Split
                fact_match = re.search(r"Fato: (.*?) \|", analysis)
                score_match = re.search(r"Score: ([\d.]+)", analysis)
                
                if fact_match and score_match:
                    fact = fact_match.group(1).strip()
                    importance = float(score_match.group(1).strip())
                    
                    # Salva no Core (v3)
                    self.core.remember(
                        content=fact, 
                        importance=importance, 
                        metadata={"source": "auto_extraction"}
                    )
                    print(f"🧠 Memória Salva: {fact} (Imp: {importance})")
            except Exception as e:
                print(f"Erro ao processar extração de memória: {e}")

    def _call_ollama(self, messages):
        try:
            payload = {"model": self.model, "messages": messages, "stream": False}
            res = requests.post(self.ollama_url, json=payload, timeout=10)
            return res.json()["message"]["content"]
        except Exception as e:
            return f"Erro de conexão com Ollama: {e}"