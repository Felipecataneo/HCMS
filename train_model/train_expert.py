import psutil
import builtins
builtins.psutil = psutil


"""
Fine-tuning para modelo híbrido RAG-aware
Modelo aprende a usar output do HCMS
"""

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torch

def finetune_rag_aware_model(
    dataset_path: str = "training_data/synthetic_llm_dataset.jsonl",
    base_model: str = "unsloth/Qwen2.5-1.5B-Instruct",  # Melhor para PT-BR
    output_dir: str = "./hcms_personal_llm",
    max_steps: int = 500,
    learning_rate: float = 2e-4
):
    
    print(f"🚀 Iniciando fine-tuning híbrido")
    print(f"   Base: {base_model}")
    print(f"   Dataset: {dataset_path}")
    
    # 1. Carrega modelo base (4-bit)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=4096,  # Contexto maior para memórias
        dtype=None,
        load_in_4bit=True,
    )
    
    # 2. LoRA config
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # Rank maior (aprende mais nuances)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    
    # 3. Carrega dataset
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    print(f"📊 Dataset: {len(dataset)} exemplos")
    
    # Análise de distribuição
    from collections import Counter
    types = Counter([ex['metadata']['type'] for ex in dataset])
    print(f"   Tipos: {dict(types)}")
    
    # 4. Tokenização
    def format_prompt(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(format_prompt, batched=True)
    
    # 5. Treina
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=4096,
        args=TrainingArguments(
            per_device_train_batch_size=1,  # Batch pequeno para contexto grande
            gradient_accumulation_steps=8,  # Compensa batch size
            warmup_steps=20,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",  # Melhor para fine-tuning
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=100,
        ),
    )
    
    print("\n🔥 Iniciando treinamento...")
    trainer.train()
    
    # 6. Salva
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✅ Modelo salvo em {output_dir}")
    print(f"   Para testar: python scripts/test_finetuned.py")


if __name__ == "__main__":
    finetune_rag_aware_model()