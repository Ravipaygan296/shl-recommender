import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from utils.search_engine import load_data, recommend_assessments
from typing import List


def recall_at_k(relevant: List[str], predicted: List[str], k: int) -> float:
    relevant_set = set(relevant)
    top_k = predicted[:k]
    retrieved_relevant = len(set(top_k) & relevant_set)
    return retrieved_relevant / len(relevant) if relevant else 0.0


def average_precision_at_k(relevant: List[str], predicted: List[str], k: int) -> float:
    relevant_set = set(relevant)
    score = 0.0
    hit_count = 0
    for i in range(min(k, len(predicted))):
        if predicted[i] in relevant_set:
            hit_count += 1
            score += hit_count / (i + 1)
    return score / min(k, len(relevant_set)) if relevant else 0.0


def evaluate_model(test_queries_path: str, catalog_path: str, k: int = 3):
    test_df = pd.read_csv(test_queries_path)
    catalog_df = load_data(catalog_path)

    recall_scores = []
    map_scores = []

    for _, row in test_df.iterrows():
        query = row['Query']
        relevant = [x.strip() for x in row['Relevant Assessments'].split(';') if x.strip()]
        results = recommend_assessments(query, catalog_df, top_k=10)
        predicted = results['Assessment Name'].tolist()

        recall_scores.append(recall_at_k(relevant, predicted, k))
        map_scores.append(average_precision_at_k(relevant, predicted, k))

    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_map = sum(map_scores) / len(map_scores)

    print(f"Mean Recall@{k}: {mean_recall:.4f}")
    print(f"MAP@{k}: {mean_map:.4f}")


if __name__ == "__main__":
    evaluate_model("evaluation/test_queries.csv", "data/shl_mock_catalog.csv", k=3)
