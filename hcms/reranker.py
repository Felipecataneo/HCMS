# hcms/reranker.py
from sentence_transformers import CrossEncoder

class MemoryReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        # Modelo leve e extremamente rápido para reranking de Top 20
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        if not candidates: return []
        
        # Prepara os pares para o modelo
        pairs = [[query, c['content']] for c in candidates]
        scores = self.model.predict(pairs)

        # Atribui scores e reordena
        for i, score in enumerate(scores):
            candidates[i]['rerank_score'] = float(score)

        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)

