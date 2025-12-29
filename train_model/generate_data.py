# scripts/generate_synthetic_llm_dataset.py
"""
Gera dataset realista usando LLM (OpenAI GPT-4o-mini)
Foco: uso correto de memórias, ruído, contradições e honestidade
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import sys

from openai import OpenAI

# ============================================================
# CONFIGURAÇÃO OPENAI
# ============================================================

MODEL = "gpt-4.1-mini"
client = OpenAI()

OUTPUT_FILE = Path("training_data/synthetic_llm_dataset.jsonl")
OUTPUT_FILE.parent.mkdir(exist_ok=True)

CHECKPOINT_FILE = Path("training_data/.checkpoint.json")
REJECTED_LOG = Path("training_data/.rejected_examples.jsonl")

SYSTEM_CORE = """Você é um assistente pessoal com memória persistente.
REGRAS IMPORTANTES:
- Use apenas informações presentes nas memórias
- Priorize memórias mais recentes se houver conflito
- Ignore memórias irrelevantes
- Nunca invente dados
- Se a informação não existir, diga explicitamente que não sabe"""

# ============================================================
# CENÁRIOS EXPANDIDOS
# ============================================================

SCENARIO_TEMPLATES = {
    "factual_lookup": {
        "description": "Busca direta de informação específica",
        "prompt_template": """Crie um exemplo onde o usuário busca uma informação específica (senha, código, telefone, data, etc).

Estrutura:
- 1 memória contém a resposta exata
- 0-2 memórias irrelevantes como ruído
- Resposta deve citar literalmente o valor da memória

Exemplo de boa resposta:
[Memória 1 | 2024-01-15] A senha do Wi-Fi é Secure@2024
[Memória 2 | 2024-01-10] O escritório fica na Rua das Flores

User: Qual a senha do Wi-Fi?
Assistant: A senha do Wi-Fi é Secure@2024."""
    },

    "conflicting_memory": {
        "description": "Memórias contraditórias - usar a mais recente",
        "prompt_template": """Crie um exemplo onde existem 2-3 memórias COM DATAS sobre o mesmo assunto, mas com valores diferentes.

IMPORTANTE:
- Memórias devem ter datas explícitas [Memória X | YYYY-MM-DD]
- O assistente deve usar a memória MAIS RECENTE
- Mencionar que a informação foi atualizada

Exemplo:
[Memória 1 | 2024-01-20] Meu email agora é novo@empresa.com
[Memória 2 | 2023-12-10] Meu email é antigo@empresa.com

User: Qual meu email?
Assistant: Seu email atual é novo@empresa.com (atualizado em janeiro de 2024)."""
    },

    "noisy_context": {
        "description": "Muitas memórias irrelevantes - filtrar ruído",
        "prompt_template": """Crie um exemplo com 3-4 memórias onde APENAS 1 é relevante para a pergunta.

As outras devem ser sobre temas similares mas não responderem a pergunta.

Exemplo:
[Memória 1 | 2024-01-15] Reunião com cliente ABC cancelada
[Memória 2 | 2024-01-14] Reunião com fornecedor XYZ às 14h na sala 3
[Memória 3 | 2024-01-13] Preciso enviar proposta para cliente ABC
[Memória 4 | 2024-01-12] Cliente ABC pediu desconto de 15%

User: Onde é a reunião com o fornecedor?
Assistant: A reunião com o fornecedor XYZ será às 14h na sala 3."""
    },

    "temporal_reasoning": {
        "description": "Raciocínio sobre linha do tempo de eventos",
        "prompt_template": """Crie uma interação de raciocínio temporal.
        
REQUISITOS DE CONTEÚDO:
- Inclua 2 ou 3 memórias com datas explícitas no formato [Memória X | YYYY-MM-DD].
- O usuário deve fazer uma pergunta que exija comparar essas datas (ex: "O que aconteceu primeiro?", "Qual a ordem dos eventos?", "Quanto tempo faz desde o evento X?").
- O assistente deve responder analisando a linha do tempo, citando as datas e a sucessão dos fatos de forma lógica."""
    },

    "synthesis": {
        "description": "Sintetizar informações de múltiplas memórias",
        "prompt_template": """Crie um exemplo onde a resposta exige COMBINAR informações de 2-3 memórias diferentes.

Exemplo:
[Memória 1 | 2024-01-10] João trabalha no departamento de TI
[Memória 2 | 2024-01-12] O ramal do departamento de TI é 5042
[Memória 3 | 2024-01-05] João atende de segunda a sexta

User: Qual o ramal do João e quando ele trabalha?
Assistant: O João trabalha no departamento de TI, cujo ramal é 5042. Ele atende de segunda a sexta."""
    },

    "missing_information": {
        "description": "Informação não existe nas memórias - admitir ignorância",
        "prompt_template": """Crie um exemplo onde as memórias NÃO contêm a informação solicitada.

Exemplo:
[Memória 1 | 2024-01-15] Reunião com cliente ABC às 14h
[Memória 2 | 2024-01-10] Proposta enviada para cliente ABC

User: Qual o telefone do cliente ABC?
Assistant: Não tenho o telefone do cliente ABC registrado nas minhas memórias. Você pode verificar no sistema de CRM ou na planilha de contatos."""
    },

    "personal_preferences": {
        "description": "Preferências e características pessoais do usuário",
        "prompt_template": """Crie um exemplo sobre preferências pessoais, gostos, restrições ou características do usuário.

Exemplo:
[Memória 1 | 2024-01-20] Sou alérgico a camarão e frutos do mar
[Memória 2 | 2024-01-18] Gosto de comida japonesa, especialmente sushi

User: Pode me sugerir um prato japonês?
Assistant: Como você gosta de comida japonesa mas é alérgico a frutos do mar, sugiro pratos como yakisoba, tempurá de legumes, gyoza ou teriyaki. Evite sushi e sashimi que normalmente contêm frutos do mar."""
    },

    "code_and_technical": {
        "description": "Códigos técnicos, senhas, APIs, comandos",
        "prompt_template": """Crie um exemplo sobre informação técnica: senhas, chaves API, comandos, configurações.

Exemplo:
[Memória 1 | 2024-01-15] API key do serviço AWS: AKIAIOSFODNN7EXAMPLE
[Memória 2 | 2024-01-10] Comando deploy: ./deploy.sh --env=production --region=us-east-1

User: Qual a chave da API AWS?
Assistant: A chave da API AWS é: AKIAIOSFODNN7EXAMPLE"""
    },

    "multi_person_context": {
        "description": "Informações sobre diferentes pessoas",
        "prompt_template": """Crie um exemplo com informações sobre 2-3 pessoas diferentes.

Exemplo:
[Memória 1 | 2024-01-15] Maria Silva - tel: (11) 98765-4321, cargo: Gerente de Vendas
[Memória 2 | 2024-01-14] João Santos - tel: (11) 91234-5678, cargo: Analista de TI
[Memória 3 | 2024-01-13] Ana Costa - tel: (11) 99876-5432, cargo: Diretora Financeira

User: Qual o telefone da Maria?
Assistant: O telefone da Maria Silva é (11) 98765-4321."""
    },

    "project_status": {
        "description": "Status e progresso de projetos/tarefas",
        "prompt_template": """Crie um exemplo sobre status de projeto, tarefa ou atividade.

Exemplo:
[Memória 1 | 2024-01-20] Projeto X: 65% concluído, fase de testes em andamento
[Memória 2 | 2024-01-18] Próximo milestone: entrega do módulo de relatórios até 30/01
[Memória 3 | 2024-01-15] Responsável técnico: Carlos Oliveira

User: Como está o projeto X?
Assistant: O Projeto X está 65% concluído, atualmente na fase de testes. O próximo marco é a entrega do módulo de relatórios até 30/01. O responsável técnico é Carlos Oliveira."""
    },

    "financial_info": {
        "description": "Dados financeiros",
        "prompt_template": """Crie um exemplo sobre informação financeira: valores, orçamentos, custos, faturamento.

Exemplo:
[Memória 1 | 2024-01-20] Orçamento marketing Q1: R$ 45.000,00
[Memória 2 | 2024-01-18] Gasto até agora: R$ 28.500,00
[Memória 3 | 2024-01-15] Saldo disponível: R$ 16.500,00

User: Quanto ainda temos de orçamento de marketing?
Assistant: Do orçamento de marketing do primeiro trimestre de R$ 45.000,00, já foram gastos R$ 28.500,00. O saldo disponível é de R$ 16.500,00."""
    },

    "location_address": {
        "description": "Endereços e localizações",
        "prompt_template": """Crie um exemplo sobre endereços, locais ou como chegar a algum lugar.

Exemplo:
[Memória 1 | 2024-01-15] Escritório central: Rua das Flores, 123 - Jardim Paulista, São Paulo
[Memória 2 | 2024-01-10] Para chegar: Metrô estação Consolação, saída sul, 5 minutos a pé

User: Onde fica o escritório?
Assistant: O escritório central fica na Rua das Flores, 123 - Jardim Paulista, São Paulo. Para chegar de metrô, desça na estação Consolação (saída sul) e caminhe 5 minutos."""
    },

    "health_medical": {
        "description": "Informações de saúde",
        "prompt_template": """Crie um exemplo sobre informação de saúde: alergias, medicamentos, exames, tipo sanguíneo.

Exemplo:
[Memória 1 | 2024-01-20] Tipo sanguíneo: O+
[Memória 2 | 2024-01-15] Medicamento contínuo: Losartana 50mg - 1x ao dia pela manhã
[Memória 3 | 2024-01-10] Alérgico a dipirona

User: Qual meu tipo sanguíneo e alergias?
Assistant: Seu tipo sanguíneo é O+ e você tem alergia a dipirona."""
    },

    "comparative_analysis": {
        "description": "Comparação entre períodos",
        "prompt_template": """Crie um exemplo onde o usuário quer comparar informações de diferentes períodos.

Exemplo:
[Memória 1 | 2024-01-15] Vendas janeiro 2024: R$ 180.000
[Memória 2 | 2023-12-20] Vendas dezembro 2023: R$ 150.000
[Memória 3 | 2023-11-25] Vendas novembro 2023: R$ 145.000

User: Como foram as vendas nos últimos 3 meses?
Assistant: As vendas apresentaram crescimento: novembro/2023 com R$ 150.000, dezembro/2023 com R$ 150.000 e janeiro/2024 com R$ 180.000 - um aumento de 20% em relação a dezembro."""
    },

    "ambiguous_query": {
        "description": "Pergunta ambígua resolvida por contexto",
        "prompt_template": """Crie um exemplo onde a pergunta do usuário é ambígua (ex: "ele", "isso", "lá") mas o contexto permite entender.

Exemplo:
[Memória 1 | 2024-01-20] Reunião com Dr. Roberto Silva agendada para 25/01 às 15h
[Memória 2 | 2024-01-18] Dr. Roberto atende na Clínica São Lucas, Av. Paulista 1000

User: Onde é lá mesmo?
Assistant: A reunião com o Dr. Roberto Silva será na Clínica São Lucas, localizada na Av. Paulista 1000, no dia 25/01 às 15h."""
    }
}

# ============================================================
# FUNÇÕES DE GERAÇÃO
# ============================================================

def openai_generate(prompt: str, temperature: float = 0.7, timeout: int = 30, attempt: int = 0) -> str:
    """Chama OpenAI com retry progressivo"""
    max_retries = 3
    actual_timeout = timeout + (attempt * 10)
    
    for retry in range(max_retries):
        try:
            print(f"   🔄 Chamando OpenAI...", end=" ", flush=True)
            
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Você é um gerador de dados sintéticos. Responda SEMPRE em JSON válido sem markdown."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=1000,
                timeout=actual_timeout
            )
            return response.choices[0].message.content
            
        except Exception as e:
            if retry < max_retries - 1:
                wait_time = 2 ** retry
                print(f"Erro, aguardando {wait_time}s...", end=" ", flush=True)
                time.sleep(wait_time)
            else:
                raise e


def build_scenario_prompt(scenario: str) -> str:
    """Constrói prompt específico para cada cenário"""
    
    template = SCENARIO_TEMPLATES.get(scenario)
    
    if not template:
        base_instruction = f"""Gere UMA interação em português no formato JSON.

IMPORTANTE: As memórias devem aparecer APENAS dentro do campo "content" da mensagem system, como texto formatado [Memória X | data].

NÃO crie um campo "memories" separado no JSON.

REGRAS:
- O assistente SEMPRE usa informações das memórias fornecidas
- Para {scenario}, forneça exemplos concretos e específicos
- A resposta deve ser direta e baseada nas memórias

FORMATO CORRETO:"""
    else:
        base_instruction = f"""{template['prompt_template']}

CRÍTICO PARA FORMATO JSON:
- As memórias devem estar DENTRO do campo "content" da mensagem system
- NÃO crie um campo "memories" separado no root do JSON
- Formato: "Você é um assistente...\n\nMemórias disponíveis:\n[Memória 1 | data] texto\n[Memória 2 | data] texto"

Exemplo de estrutura CORRETA:"""

    return base_instruction + f"""

{{
  "messages": [
    {{
      "role": "system",
      "content": "{SYSTEM_CORE}\n\nMemórias disponíveis:\n[Memória 1 | 2024-01-15] Exemplo de memória aqui\n[Memória 2 | 2024-01-10] Outra memória aqui"
    }},
    {{"role": "user", "content": "pergunta do usuário"}},
    {{"role": "assistant", "content": "resposta baseada nas memórias"}}
  ],
  "metadata": {{"type": "{scenario}"}}
}}

IMPORTANTE: 
- NÃO adicione campo "memories" no root
- Memórias vão DENTRO de messages[0].content
- Retorne APENAS o JSON, sem markdown ou explicações"""


def parse_json_safe(text: str) -> Dict:
    """Parse JSON com tratamento robusto de markdown e erros"""
    try:
        # Remove markdown fences se existir
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if len(lines) > 2:
                text = "\n".join(lines[1:-1])
            else:
                text = text.replace("```json", "").replace("```", "")
        
        parsed = json.loads(text)
        
        # Validação estrutural básica
        if "messages" not in parsed or not isinstance(parsed["messages"], list):
            return None
        
        # Garante que metadata existe
        if "metadata" not in parsed:
            parsed["metadata"] = {}
        
        # Valida roles necessárias
        roles = [msg.get("role", "").lower() for msg in parsed["messages"]]
        required_roles = ["system", "user", "assistant"]
        
        if not all(r in roles for r in required_roles):
            return None
        
        # Garante que todas as mensagens têm content
        for msg in parsed["messages"]:
            if "content" not in msg or not isinstance(msg["content"], str):
                return None
        
        return parsed
        
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None
def validate_example(example: Dict) -> bool:
    """Validação final - JSONL estável para PyArrow"""
    try:
        # 1. PROÍBE memories no root
        if "memories" in example:
            log_rejection(example, "Campo memories no root")
            return False
        
        # 2. Valida messages
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            log_rejection(example, "messages inválido")
            return False
        
        # 3. Valida cada mensagem
        for msg in messages:
            if not isinstance(msg, dict):
                log_rejection(example, "Mensagem não é dict")
                return False
            
            role = msg.get("role")
            content = msg.get("content")
            
            if role not in ("system", "user", "assistant"):
                log_rejection(example, f"Role inválido: {role}")
                return False
            
            if not isinstance(content, str):
                log_rejection(example, "content não é string")
                return False
            
            # REFINADO: detecta JSON estrutural, não texto casual
            content_stripped = content.strip()
            if content_stripped.startswith('{') and content_stripped.endswith('}'):
                try:
                    parsed = json.loads(content_stripped)
                    # Se tem campo "memories" ou "messages" = é schema fantasma
                    if isinstance(parsed, dict) and any(k in parsed for k in ['memories', 'messages', 'metadata']):
                        log_rejection(example, "JSON estrutural em content")
                        return False
                except:
                    pass  # Não é JSON válido = OK
            
            if content_stripped.startswith('[') and content_stripped.endswith(']'):
                try:
                    parsed = json.loads(content_stripped)
                    if isinstance(parsed, list):
                        log_rejection(example, "Array JSON em content")
                        return False
                except:
                    pass
        
        # 4. Valida ordem
        if messages[0]["role"] != "system":
            log_rejection(example, "Primeira msg não é system")
            return False
        
        if messages[-1]["role"] != "assistant":
            log_rejection(example, "Última msg não é assistant")
            return False
        
        # 5. Conteúdo mínimo
        if len(messages[-2]["content"]) < 5 or len(messages[-1]["content"]) < 5:
            log_rejection(example, "Conteúdo muito curto")
            return False
        
        # 6. Metadata opcional
        metadata = example.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            log_rejection(example, "metadata inválido")
            return False
        
        return True
    
    except Exception as e:
        log_rejection(example, f"Erro: {e}")
        return False



def log_rejection(example: Dict, reason: str):
    """Registra exemplos rejeitados para debug"""
    try:
        with open(REJECTED_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "reason": reason,
                "example": example,
                "timestamp": datetime.now().isoformat()
            }, ensure_ascii=False) + "\n")
    except:
        pass  # Não falhar por erro de logging


def save_checkpoint(scenario_progress: Dict):
    """Salva progresso para retomar depois"""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump(scenario_progress, f)


def load_checkpoint() -> Dict:
    """Carrega checkpoint se existir"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def print_progress_bar(current: int, total: int, prefix: str = "", length: int = 40):
    """Imprime barra de progresso"""
    percent = current / total if total > 0 else 0
    filled = int(length * percent)
    bar = "█" * filled + "░" * (length - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({percent*100:.1f}%)", end="", flush=True)


# ============================================================
# GERAÇÃO DO DATASET
# ============================================================

def generate_dataset(total_examples: int = 1000, examples_per_scenario: int = None, resume: bool = True):
    """
    Gera dataset com checkpoint e progress tracking
    
    Args:
        total_examples: Total de exemplos
        examples_per_scenario: Exemplos por cenário (None = distribui uniformemente)
        resume: Se True, retoma de checkpoint se existir
    """
    
    scenarios = list(SCENARIO_TEMPLATES.keys())
    
    if examples_per_scenario is None:
        examples_per_scenario = total_examples // len(scenarios)
    
    # Carrega checkpoint se existir
    scenario_progress = load_checkpoint() if resume else {}
    
    print(f"\n📊 Configuração:")
    print(f"   Cenários: {len(scenarios)}")
    print(f"   Exemplos por cenário: {examples_per_scenario}")
    print(f"   Total estimado: {examples_per_scenario * len(scenarios)}")
    
    if scenario_progress:
        completed = sum(scenario_progress.values())
        print(f"   ✓ Retomando checkpoint: {completed} exemplos já gerados")
    
    print()
    
    generated: List[Dict] = []
    failed_count = 0
    total_attempts = 0
    
    # Carrega exemplos já gerados se estiver retomando
    if resume and OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    generated.append(json.loads(line))
                except:
                    pass
        print(f"📥 Carregados {len(generated)} exemplos existentes\n")
    
    start_time = time.time()
    
    for scenario_idx, scenario in enumerate(scenarios, 1):
        print(f"\n{'='*80}")
        print(f"[{scenario_idx}/{len(scenarios)}] Cenário: {scenario}")
        print(f"Descrição: {SCENARIO_TEMPLATES[scenario]['description']}")
        print('='*80)
        
        # Verifica se já completou este cenário
        scenario_count = scenario_progress.get(scenario, 0)
        
        if scenario_count >= examples_per_scenario:
            print(f"✓ Cenário já completado ({scenario_count} exemplos)")
            continue
        
        attempts = 0
        max_attempts = (examples_per_scenario - scenario_count) * 2  # Reduzido de *3
        
        while scenario_count < examples_per_scenario and attempts < max_attempts:
            attempts += 1
            total_attempts += 1
            
            # Progress bar
            print_progress_bar(
                scenario_count, 
                examples_per_scenario,
                prefix=f"   Progresso"
            )
            
            try:
                # Gera exemplo
                prompt = build_scenario_prompt(scenario)
                raw_response = openai_generate(prompt, timeout=30, attempt=attempts)
                
                # Parse
                parsed = parse_json_safe(raw_response)
                
                if not parsed:
                    failed_count += 1
                    print(f"\n   ⚠️ Falha no parse (tentativa {attempts}/{max_attempts})")
                    continue
                
                # Valida qualidade
                if not validate_example(parsed):
                    failed_count += 1
                    print(f"\n   ⚠️ Exemplo rejeitado (tentativa {attempts}/{max_attempts})")
                    continue
                
                # Adiciona metadata
                if "metadata" not in parsed:
                    parsed["metadata"] = {}
                    
                parsed["metadata"]["generated_by"] = "openai"
                parsed["metadata"]["model"] = MODEL
                parsed["metadata"]["timestamp"] = datetime.now().isoformat()
                parsed["metadata"]["type"] = scenario
                
                generated.append(parsed)
                scenario_count += 1

                # Salva incrementalmente
                with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                
                # Atualiza checkpoint a cada 5 exemplos
                if scenario_count % 5 == 0:
                    scenario_progress[scenario] = scenario_count
                    save_checkpoint(scenario_progress)
                
            except KeyboardInterrupt:
                print("\n\n⚠️ Interrompido pelo usuário. Salvando progresso...")
                scenario_progress[scenario] = scenario_count
                save_checkpoint(scenario_progress)
                
                print(f"✓ Progresso salvo: {len(generated)} exemplos")
                sys.exit(0)
                
            except Exception as e:
                failed_count += 1
                print(f"\n   ❌ Erro: {str(e)[:100]}")
                
                # Se muitas falhas consecutivas, pausa mais tempo
                if failed_count % 5 == 0:
                    print(f"   ⏸️ Muitas falhas. Pausando 10s...")
                    time.sleep(10)
                
                continue
        
        # Finaliza barra de progresso
        print_progress_bar(scenario_count, examples_per_scenario, prefix=f"   Progresso")
        print()  # Nova linha
        
        # Atualiza checkpoint final do cenário
        scenario_progress[scenario] = scenario_count
        save_checkpoint(scenario_progress)
        
        elapsed = time.time() - start_time
        rate = len(generated) / elapsed if elapsed > 0 else 0
        
        print(f"\n   ✅ Completado: {scenario_count} exemplos")
        print(f"   ⏱️ Taxa: {rate:.2f} exemplos/segundo")
        print(f"   ❌ Falhas neste cenário: {attempts - scenario_count}")
    
    # Shuffle final
    random.shuffle(generated)
    
    # Salva final
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in generated:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    
    # Remove checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
    
    # Estatísticas finais
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("ESTATÍSTICAS FINAIS")
    print("="*80)
    
    print(f"\n✅ Exemplos gerados: {len(generated)}")
    print(f"❌ Falhas/Rejeições: {failed_count}")
    print(f"⏱️ Tempo total: {total_time/60:.1f} minutos")
    print(f"📊 Taxa média: {len(generated)/total_time:.2f} exemplos/segundo")
    print(f"📁 Arquivo: {OUTPUT_FILE}")
    print(f"💾 Tamanho: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    
    # Taxa de sucesso
    success_rate = (len(generated) / total_attempts * 100) if total_attempts > 0 else 0
    print(f"📈 Taxa de sucesso: {success_rate:.1f}%")
    
    # Distribuição por tipo
    from collections import Counter
    types = Counter([ex["metadata"]["type"] for ex in generated])
    
    print("\n📈 Distribuição por cenário:")
    for scenario_type in scenarios:
        count = types.get(scenario_type, 0)
        percentage = count / len(generated) * 100 if generated else 0
        bar = "█" * (count // 5) + "░" * ((examples_per_scenario - count) // 5)
        print(f"  {scenario_type:25} [{bar}] {count:3} ({percentage:5.1f}%)")


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("GERAÇÃO DE DATASET REALISTA COM OPENAI GPT-4o-mini")
    print("="*80)
    
    try:
        generate_dataset(
            total_examples=1500,
            examples_per_scenario=100,
            resume=True
        )
    except KeyboardInterrupt:
        print("\n\n⚠️ Processo interrompido. Progresso foi salvo.")
        print("   Execute novamente para retomar de onde parou.")
    except Exception as e:
        print(f"\n\n❌ Erro fatal: {e}")
        print("   Verifique os logs em .rejected_examples.jsonl")