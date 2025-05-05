import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os

DATA_PATH = os.path.join("data", "shl_mock_catalog.csv")

# Load the catalog and precompute embeddings
def load_data(catalog_path=DATA_PATH):
    df = pd.read_csv(catalog_path)
    df['Description'] = df['Description'].fillna('')
    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus = df['Description'].tolist()
    embeddings = model.encode(corpus, convert_to_tensor=True)
    return df, embeddings, model

# Recommend assessments based on semantic similarity
def recommend_assessments(query, df, embeddings, model, top_k=10):
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_indices = np.argpartition(-scores, range(min(top_k, len(scores))))[:top_k]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "assessment_name": row["Assessment Name"],
            "url": row.get("URL", "https://www.shl.com/solutions/products/product-catalog/"),
            "remote_testing": row.get("Remote Testing Support", "Unknown"),
            "adaptive_support": row.get("Adaptive/IRT Support", "Unknown"),
            "duration": row.get("Duration", "Not Provided"),
            "test_type": row.get("Assessment Type", "Not Provided")
        })
    return results


