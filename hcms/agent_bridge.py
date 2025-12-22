# hcms/agent_bridge.py

import json
import requests
import time
from typing import List, Dict

class HCMSAgentBridge:
    def __init__(self, hcms_instance, model_name="llama3.2:3b", ollama_url="http://localhost:11434"):
        self.hcms = hcms_instance
        self.model = model_name
        self.url = f"{ollama_url}/api/chat"

    def chat(self, user_input: str) -> str:
        """
        Ciclo de vida de uma interação com filtro de relevância e memória.
        """
        # 1. Recuperação com Filtro de Relevância
        memories = self.hcms.recall(user_input, limit=10)

        relevant_memories = [
            m for m in memories
            if m.get('rerank_score', 0) > 0.5 or m.get('importance', 0) > 0.7
        ]

        context = self._format_context(relevant_memories)

        system_prompt = f"""
        Você é um assistente pessoal. Responda de forma direta e amigável.
        Use o CONTEXTO abaixo para personalizar a resposta.
        IMPORTANTE: Se o contexto for irrelevante ao assunto atual, IGNORE-O totalmente.

        CONTEXTO DE MEMÓRIA:
        {context}
        """

        response = self._call_ollama([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ])

        # 2. Memorização Inteligente
        self._extract_and_smart_remember(user_input, response)

        return response

    def _format_context(self, memories: List[Dict]) -> str:
        if not memories:
            return "Nenhuma memória anterior relevante para este assunto."

        lines = []
        for m in memories:
            line = f"- {m['content']}"
            if m.get('context_edges'):
                related = ", ".join([e['content'] for e in m['context_edges']])
                line += f" (Relacionado: {related})"
            lines.append(line)
        return "\n".join(lines)

    def _call_ollama(self, messages: List[Dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        try:
            res = requests.post(self.url, json=payload)
            return res.json()['message']['content']
        except Exception as e:
            return f"Erro ao conectar com Ollama: {str(e)}"

    def _extract_and_smart_remember(self, user_input: str, agent_response: str):
        """
        Extrai fatos da conversa com correção de calibragem
        e evita duplicatas (Upsert Cognitivo).
        """
        extract_prompt = f"""
        Extraia UM fato atômico.

        Regras de Importância:
        - 0.9–1.0: APENAS nomes, locais de nascimento ou dados de segurança reais.
        - 0.5–0.8: Preferências e informações úteis.
        - 0.1–0.4: Contexto geral ou conversa casual.

        Conversa: {user_input} | {agent_response}

        Responda APENAS o JSON:
        {{"fact": "string", "importance": float}}
        """

        raw_json = self._call_ollama([{"role": "user", "content": extract_prompt}])

        try:
            # --- Parsing robusto ---
            start = raw_json.find("{")
            end = raw_json.rfind("}") + 1
            data = json.loads(raw_json[start:end])

            fact = data["fact"]
            importance = float(data["importance"])

            # ==========================================================
            # 🔧 CORREÇÃO DE CALIBRAGEM (anti-ruído do LLaMA 3B)
            # ==========================================================

            # Penaliza termos técnicos genéricos superestimados
            noise_keywords = [
                "erro", "aleatório", "sistema", "teste",
                "documento", "código", "modelo", "pipeline"
            ]
            if any(word in fact.lower() for word in noise_keywords):
                importance *= 0.3

            # Valoriza dados pessoais identificáveis
            personal_keywords = [
                "nome", "nascido", "nasceu", "mora",
                "vive", "prefere", "gosta", "trabalha"
            ]
            if any(word in fact.lower() for word in personal_keywords):
                importance = min(1.0, importance * 1.2)

            # Clamp final defensivo
            importance = max(0.05, min(1.0, importance))

            # ==========================================================
            # 🧠 UPSERT COGNITIVO
            # ==========================================================

            existing = self.hcms.recall(fact, limit=1)

            if existing and existing[0].get("rerank_score", 0) > 7.0:
                print(f"♻️ Memória redundante evitada: {fact}")
                self.hcms.storage.execute(
                    "UPDATE memories SET last_access = %s WHERE id = %s",
                    (time.time(), existing[0]["id"])
                )
            else:
                self.hcms.remember(fact, importance=importance)
                print(f"💾 Novo fato memorizado ({importance:.2f}): {fact}")

        except Exception as e:
            # Falha silenciosa esperada em modelos pequenos
            print(f"⚠️ Falha na extração de fatos: {str(e)}")
