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

system_prompt = (
    "You are a technical data extractor. "
    "If a message contains a real password, IP, API key, or credential, extract it. "
    "Otherwise, say there is nothing to extract."
)

print("\n=== DEBUG DE SAÍDA BRUTA (RAW) ===\n")

# Testaremos apenas os 3 principais
test_cases = [
    "a senha do cadeado é 0904",
    "IP do servidor: 192.168.1.10",
    "bom dia, como vai?"
]

for text in test_cases:
    messages = [{"role": "developer", "content": system_prompt}, {"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.01, do_sample=False)
    
    # AQUI ESTÁ O SEGREDO: skip_special_tokens=False para ver as tags
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=False)
    
    print(f"INPUT: {text}")
    print(f"RAW OUTPUT: {response.strip()}") # Isso vai mostrar a verdade
    print("-" * 50)