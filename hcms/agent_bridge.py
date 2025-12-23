# hcms/agent_bridge.py

import json
import requests
import time
from typing import List, Dict
from hcms.core import HCMSOptimized


class HCMSAgentBridge:
    def __init__(
        self,
        hcms_instance: HCMSOptimized,
        model_name: str = "llama3.2:3b",
        ollama_url: str = "http://localhost:11434",
    ):
        self.hcms = hcms_instance
        self.model = model_name
        self.url = f"{ollama_url}/api/chat"

    # =========================================================================
    # CICLO PRINCIPAL DE CHAT
    # =========================================================================
    def chat(self, user_input: str) -> str:
        """
        Interação com:
        - Recall rápido
        - Filtro de relevância robusto
        - Contexto seletivo
        - Memorização inteligente
        """

        # 1. Recall (score 0–10)
        memories = self.hcms.recall(user_input, limit=10)

        # 2. Filtro de relevância (decisão cognitiva)
        relevant_memories = [
            m for m in memories
            if m.get("rerank_score", 0) >= 5.0
            or m.get("importance", 0) >= 0.7
        ]

        context = self._format_context(relevant_memories)

        system_prompt = f"""
Você é um assistente pessoal.
Responda de forma direta e amigável.

Use o CONTEXTO apenas se ele for realmente relevante.
Se não ajudar na resposta atual, IGNORE-O completamente.

CONTEXTO DE MEMÓRIA:
{context}
""".strip()

        response = self._call_ollama([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ])

        # 3. Extração + Memorização
        self._extract_and_smart_remember(user_input, response)

        return response

    # =========================================================================
    # CONTEXTO
    # =========================================================================
    def _format_context(self, memories: List[Dict]) -> str:
        if not memories:
            return "Nenhuma memória relevante."

        lines = []
        for m in memories:
            line = f"- {m['content']}"
            if m.get("context_edges"):
                related = ", ".join(e["content"] for e in m["context_edges"])
                line += f" (Relacionado: {related})"
            lines.append(line)

        return "\n".join(lines)

    # =========================================================================
    # OLLAMA
    # =========================================================================
    def _call_ollama(self, messages: List[Dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        try:
            res = requests.post(self.url, json=payload, timeout=60)
            return res.json()["message"]["content"]
        except Exception as e:
            return f"Erro ao conectar com Ollama: {str(e)}"

    # =========================================================================
    # EXTRAÇÃO + UPSERT COGNITIVO
    # =========================================================================
    def _extract_and_smart_remember(self, user_input: str, agent_response: str):
        extract_prompt = f"""
Extraia UM fato atômico da conversa.

Regras de Importância:
- 0.9–1.0: Dados pessoais ou críticos reais
- 0.5–0.8: Preferências e informações úteis
- 0.1–0.4: Contexto geral ou casual

Conversa:
Usuário: {user_input}
Assistente: {agent_response}

Responda APENAS o JSON:
{{"fact": "string", "importance": float}}
"""

        raw_json = self._call_ollama(
            [{"role": "user", "content": extract_prompt}]
        )

        try:
            # Parsing defensivo
            start = raw_json.find("{")
            end = raw_json.rfind("}") + 1
            data = json.loads(raw_json[start:end])

            fact = data["fact"].strip()
            importance = float(data["importance"])

            # ----------------------------------------------------------
            # Correção de calibragem (anti-ruído LLaMA 3B)
            # ----------------------------------------------------------
            noise_keywords = {
                "erro", "aleatório", "sistema", "teste",
                "documento", "código", "modelo", "pipeline",
            }
            if any(w in fact.lower() for w in noise_keywords):
                importance *= 0.3

            personal_keywords = {
                "nome", "nascido", "nasceu", "mora",
                "vive", "prefere", "gosta", "trabalha",
            }
            if any(w in fact.lower() for w in personal_keywords):
                importance = min(1.0, importance * 1.2)

            importance = max(0.05, min(1.0, importance))

            # ----------------------------------------------------------
            # UPSERT COGNITIVO
            # ----------------------------------------------------------
            existing = self.hcms.recall(fact, limit=1)

            if existing and existing[0].get("rerank_score", 0) >= 7.0:
                # Memória já bem consolidada
                self.hcms.storage.execute(
                    "UPDATE memories SET last_access = %s WHERE id = %s",
                    (time.time(), existing[0]["id"]),
                )
                print(f"♻️ Memória redundante evitada: {fact}")
            else:
                self.hcms.remember(fact, importance=importance)
                print(f"💾 Novo fato memorizado ({importance:.2f}): {fact}")

        except Exception as e:
            # Falha silenciosa é aceitável
            print(f"⚠️ Falha na extração de fatos: {e}")
