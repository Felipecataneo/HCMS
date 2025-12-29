
import json
with open('training_data/synthetic_llm_dataset.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            # Verifica estrutura
            assert 'messages' in data
            assert isinstance(data['messages'], list)
        except Exception as e:
            print(f'Linha {i}: {e}')
