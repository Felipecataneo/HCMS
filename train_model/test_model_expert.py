from unsloth import FastLanguageModel
import torch
import re

model_dir = "hcms_v4_pc" 
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_dir,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

# TOOL DECLARATION V4
tools = [{
    "type": "function",
    "function": {
        "name": "extract_technical_fact",
        "description": "Extract ONLY real technical facts shared as actual information.",
        "parameters": {
            "type": "object",
            "properties": {
                "extracted_credential": {"type": ["string", "null"]},
                "importance": {"type": "number"},
                "permanent": {"type": "boolean"}
            },
            "required": ["extracted_credential", "importance", "permanent"]
        }
    }
}]

# PROMPT SISTEMA V4 (O SEGREDO ESTÁ AQUI - DEVE SER IGUAL AO GERADOR)
system_prompt = (
    "You are a personal notebook extractor. "
    "If the user shares a real password, credential, IP, code, or personal note, extract it. "
    "If it is a question, hypothesis, or casual message, extract nothing."
)

print("\n=== TESTE V4 COM PROMPT SINCRONIZADO ===\n")

test_cases = [
    "a senha do cadeado é 0904",
    "IP do servidor: 192.168.1.10",
    "bom dia, como vai?"
]

for text in test_cases:
    messages = [{"role": "developer", "content": system_prompt}, {"role": "user", "content": text}]
    
    # Gerando o prompt
    prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Gerando a resposta
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100, 
        temperature=0.01, 
        do_sample=False,
        repetition_penalty=1.2 # Ajuda o modelo pequeno a não travar
    )
    
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=False)
    
    print(f"INPUT: {text}")
    print(f"RAW OUTPUT: {response.strip()}")
    
    # Parser simples para o debug
    if "call:extract_technical_fact" in response:
        print("✅ MODELO DISPAROU A FUNÇÃO!")
    else:
        print("❌ MODELO IGNOROU.")
    print("-" * 50)