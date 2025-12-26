# hcms/agent_bridge.py
import requests
import json
import re

class HCMSAgentBridge:
    def __init__(self, core_instance, model="llama3.2:3b"):
        """
        Ponte entre o Core (RAG) e o LLM (Ollama)
        
        Args:
            core_instance: Instância do RAGCore
            model: Nome do modelo Ollama a usar
        """
        self.core = core_instance
        self.model = model
        self.ollama_url = "http://localhost:11434/api/chat"

    def chat(self, user_input: str) -> str:
        """
        Processa uma mensagem do usuário e retorna resposta do agente
        
        Args:
            user_input: Mensagem do usuário
            
        Returns:
            Resposta gerada pelo LLM com base no contexto recuperado
        """
        # 1. RECALL: Recupera contexto relevante do sistema de memória
        context_docs = self.core.recall(user_input, limit=5)
        context_str = "\n".join([f"- {c['content']}" for c in context_docs])

        # 2. PROMPT AUTORITATIVO: Força o LLM a confiar nas próprias memórias
        # Isso resolve o problema de o agente dizer "não sei" quando tem a info
        system_prompt = (
            "Você é o sistema de inteligência de um Agente Pessoal com memória de longo prazo. "
            "As informações no 'Contexto' são suas PRÓPRIAS memórias reais e verificadas. "
            "Se a resposta estiver listada no contexto, você DEVE usá-la com confiança. "
            "Nunca diga 'Não tenho essa memória' ou 'Não sei' se a informação está no contexto abaixo. "
            "Se a informação NÃO estiver no contexto, aí sim você pode dizer que não sabe. "
            "Responda de forma natural e confiante com base no que você lembra."
        )
        
        prompt = f"Contexto:\n{context_str}\n\nPergunta: {user_input}"
        
        # 3. GERA RESPOSTA usando Ollama
        response = self._call_ollama([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ])



        return response


    def analyze_and_remember(self, user_input: str):
        if len(user_input.split()) < 3: return

        mem_prompt = (
            "Analise a mensagem do usuário.\n"
            "Extraia o fato essencial. Atribua importância 0.0-1.0.\n"
            "Se o usuário quer guardar para sempre, Permanent: True.\n"
            "Formato: Fato: [texto] | Score: [valor] | Permanent: [True/False]\n\n"
            f"Mensagem: {user_input}"
        )

        analysis = self._call_ollama([{"role": "user", "content": mem_prompt}])
        print(f"DEBUG LLM Extração: {analysis}") # <--- Verifique isso no terminal

        if "Fato:" in analysis and "|" in analysis:
            try:
                # Regex mais flexível para espaços
                fact = re.search(r"Fato:\s*(.*?)\s*\|", analysis).group(1).strip()
                score = float(re.search(r"Score:\s*([\d.]+)", analysis).group(1).strip())
                is_permanent = "Permanent: True" in analysis or "permanente" in user_input.lower()

                # CHAMADA PARA O CORE
                self.core.remember(content=fact, importance=score, is_permanent=is_permanent)
                print(f"🧠 Memória Salva: {fact} (Score: {score})")
            except Exception as e:
                print(f"⚠️ Erro ao salvar: {e}")

    def _call_ollama(self, messages):
        """
        Faz chamada HTTP ao Ollama
        
        Args:
            messages: Lista de mensagens no formato OpenAI
            
        Returns:
            Resposta do LLM como string
        """
        try:
            payload = {
                "model": self.model, 
                "messages": messages, 
                "stream": False
            }
            
            res = requests.post(
                self.ollama_url, 
                json=payload, 
                timeout=30  # Aumentado de 10s para 30s (modelos maiores)
            )
            
            if res.status_code != 200:
                return f"Erro HTTP {res.status_code}: {res.text}"
            
            return res.json()["message"]["content"]
            
        except requests.exceptions.Timeout:
            return "Erro: Timeout ao conectar com Ollama (>30s). O modelo está rodando?"
        except requests.exceptions.ConnectionError:
            return "Erro: Não foi possível conectar ao Ollama. Verifique se está rodando em localhost:11434"
        except Exception as e:
            return f"Erro inesperado ao chamar Ollama: {e}"