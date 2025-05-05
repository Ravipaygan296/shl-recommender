import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os

DATA_PATH = os.path.join("data", "shl_mock_catalog.csv")

# Load and preprocess data
def load_data():
    df = pd.read_csv(DATA_PATH)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = df['Description'].fillna('').tolist()
    embeddings = model.encode(corpus, convert_to_tensor=True)
    return df, embeddings, model

# Core recommendation logic
def recommend_assessments(query, df, embeddings, model, top_k=10):
    query_embedding = model.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, embeddings)[0]
    top_results = np.argpartition(-cos_scores, range(top_k))[:top_k]

    results = []
    for idx in top_results:
        row = df.iloc[idx]
        result = {
            "assessment_name": row["Assessment Name"],
            "url": row.get("URL", "https://www.shl.com/solutions/products/product-catalog/"),
            "remote_testing": row.get("Remote Testing Support", "Unknown"),
            "adaptive_support": row.get("Adaptive/IRT Support", "Unknown"),
            "duration": row.get("Duration", "Not Provided"),
            "test_type": row.get("Assessment Type", "Not Provided")
        }
        results.append(result)

    return results

