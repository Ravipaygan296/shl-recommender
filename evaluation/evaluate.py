import json
import pandas as pd
from utils.search_engine import load_data, recommend_assessments
from sklearn.metrics import average_precision_score

def compute_recall_at_k(y_true, y_pred, k=3):
    recall_scores = []
    for true, pred in zip(y_true, y_pred):
        pred_k = pred[:k]
        recall = len(set(true).intersection(pred_k)) / len(true) if true else 0
        recall_scores.append(recall)
    return sum(recall_scores) / len(recall_scores)

def compute_map_at_k(y_true, y_pred, k=3):
    map_scores = []
    for true, pred in zip(y_true, y_pred):
        ap = 0.0
        hits = 0
        for i, p in enumerate(pred[:k]):
            if p in true:
                hits += 1
                ap += hits / (i + 1)
        if true:
            ap /= min(len(true), k)
        map_scores.append(ap)
    return sum(map_scores) / len(map_scores)

if __name__ == "__main__":
    test_set = json.load(open("evaluation/test_queries.json"))
    df, embeddings, model = load_data()

    y_true = []
    y_pred = []

    for item in test_set:
        query = item["query"]
        expected_ids = item["relevant_assessments"]
        results = recommend_assessments(query, df, embeddings, model)
        result_ids = [rec["assessment_name"] for rec in results]

        y_true.append(expected_ids)
        y_pred.append(result_ids)

    recall = compute_recall_at_k(y_true, y_pred, k=3)
    mapk = compute_map_at_k(y_true, y_pred, k=3)

    print(f"Recall@3: {recall:.3f}")
    print(f"MAP@3: {mapk:.3f}")
