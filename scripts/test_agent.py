import os
import sys

# Ajuste de Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from hcms.core import HCMS
from hcms.agent_bridge import HCMSAgentBridge

hcms = HCMS("dbname=hcms user=felipe")
# Verifique se o Ollama está rodando e com o modelo llama3.2:3b baixado
agent = HCMSAgentBridge(hcms)

print("--- AGENTE COM MEMÓRIA HCMS (OLLAMA) ---")
print("Digite 'sair' para encerrar.\n")

while True:
    u_input = input("Você: ")
    if u_input.lower() == 'sair': break
    
    response = agent.chat(u_input)
    print(f"\nAssistente: {response}\n")