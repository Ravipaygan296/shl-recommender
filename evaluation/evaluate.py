import pandas as pd
from utils.search_engine import load_data, recommend_assessments
from sklearn.metrics import average_precision_score
import numpy as np


def recall_at_k(relevant, predicted, k):
    predicted_top_k = predicted[:k]
    hits = sum(1 for item in predicted_top_k if item in relevant)
    return hits / len(relevant) if relevant else 0.0


def apk(actual, predicted, k=10):
    if not actual:
        return 0.0
    actual_set = set(actual)
    predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual_set and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)


def mapk(actual_list, predicted_list, k=10):
    return np.mean([apk(a, p, k) for a, p in zip(actual_list, predicted_list)])


def evaluate_model(test_queries_path, catalog_path, k=3):
    # Load test queries
    test_df = pd.read_csv(test_queries_path)
    test_df = test_df.dropna(subset=["Query", "Relevant Assessments"])

    # Load catalog and model
    catalog_df

