import os
import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from typing import Literal, List

# Descargar stopwords en inglés si no existen en el servidor
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Cargar el índice precalculado desde la misma carpeta del archivo main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "ir_index.pkl")

with open(INDEX_PATH, "rb") as f:
    data = pickle.load(f)

doc_ids = data["doc_ids"]
documents_meta = data["documents_meta"]
tfidf_vectorizer = data["tfidf_vectorizer"]
tfidf_matrix = data["tfidf_matrix"]
bm25_model = data["bm25_model"]

app = FastAPI(
    title="SciFact Information Retrieval API",
    description="Servicio web de recuperación de información usando TF-IDF y BM25 sobre BeIR/scifact",
    version="1.0.0"
)

# Habilitar CORS para evitar bloqueos desde clientes web o Postman
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Esquemas de entrada y salida exactos según especificación
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Texto de la consulta")
    model: Literal["bm25", "tfidf"] = Field(..., description="Modelo: 'bm25' o 'tfidf'")
    top_k: int = Field(default=10, ge=1, le=100, description="Cantidad de resultados a devolver")

class SearchResultItem(BaseModel):
    rank: int
    document_id: str
    score: float
    title: str
    snippet: str

class SearchResponse(BaseModel):
    model: str
    results: List[SearchResultItem]

def preprocess_text(text: str) -> list[str]:
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return [stemmer.stem(tok) for tok in tokens if tok not in stop_words and len(tok) > 1]

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "corpus": "BeIR/scifact",
        "indexed_documents": len(doc_ids)
    }

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    query_tokens = preprocess_text(req.query)
    if not query_tokens:
        return SearchResponse(model=req.model, results=[])
    
    if req.model == "bm25":
        scores = bm25_model.get_scores(query_tokens)
    elif req.model == "tfidf":
        query_str = " ".join(query_tokens)
        query_vec = tfidf_vectorizer.transform([query_str])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    else:
        raise HTTPException(status_code=400, detail="Modelo no soportado. Use 'bm25' o 'tfidf'.")
    
    # Ordenar índices descendentemente
    sorted_indices = np.argsort(scores)[::-1]
    
    results = []
    seen_ids = set()
    rank_counter = 1
    
    for idx in sorted_indices:
        if len(results) >= req.top_k:
            break
        score = float(scores[idx])
        if score <= 0:
            continue
        d_id = doc_ids[idx]
        if d_id in seen_ids:
            continue
        seen_ids.add(d_id)
        
        meta = documents_meta[d_id]
        raw_text = meta["text"]
        snippet = raw_text[:250].replace("\n", " ") + ("..." if len(raw_text) > 250 else "")
        
        results.append(
            SearchResultItem(
                rank=rank_counter,
                document_id=d_id,
                score=round(score, 4),
                title=meta["title"],
                snippet=snippet
            )
        )
        rank_counter += 1
        
    return SearchResponse(model=req.model, results=results)