from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from utils.search_engine import load_data, recommend_assessments

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

catalog_df, embeddings, model = load_data()

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/recommend")
def recommend(req: QueryRequest):
    recommendations = recommend_assessments(req.query, catalog_df, embeddings, model)
    return {"recommendations": recommendations}
