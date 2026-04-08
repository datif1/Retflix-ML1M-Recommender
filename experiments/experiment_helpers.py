import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, List
from tqdm import tqdm


def precision_at_k(y_true, y_pred, k):
    """Calculates the precision metric at k"""
    return len(
        set(y_pred[:k]) & set(y_true)
    ) / k if y_pred else 0.0


def recall_at_k(y_true, y_pred, k):
    """Calculates the recall metric at k"""
    return len(
        set(y_pred[:k]) & set(y_true)
    ) / len(y_true) if y_true else 0.0


def dcg_at_k(relevance, k):
    """Calculates the DCG at k"""
    return sum(
        (2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(relevance[:k])
    )


def ndcg_at_k(y_true, y_pred, k):
    """Calculates the NDCG at k"""
    relevance = [1 if mid in y_true else 0 for mid in y_pred[:k]]
    ideal = sorted(relevance, reverse=True)
    dcg = dcg_at_k(relevance, k)
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_user_at_ks(
    user_id: int,
    ratings_df: pd.DataFrame,
    k_values: List[int],
    recommender: Callable[[int, pd.DataFrame, int], List[dict]]
) -> List[dict]:
    """
    General function for recommending movies to an user with a specified
    recommender model. Used for the experiments
    """
    # Get ratings for only the user id specified
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    # Get the relevant movies where the user has rated the movie 4.0 or above
    relevant = user_ratings[user_ratings["rating"] >= 4.0]["movie__movieId"].tolist()
    # No relevant movies, return empty list
    if not relevant:
        return []

    try:
        # Get recommendations for the user
        recs = recommender(user_id, ratings_df, max(k_values))
        # Get the movie ids for the predicted movies
        pred_ids = [r["movieId"] for r in recs if "movieId" in r]
    except Exception as e:
        print(f"Error generating recommendations for user {user_id}: {e}")
        return []

    # Return a list with all of the values for the specified k values
    return [
        {
            "user_id": user_id,
            "k": k,
            "precision": precision_at_k(relevant, pred_ids, k),
            "recall": recall_at_k(relevant, pred_ids, k),
            "ndcg": ndcg_at_k(relevant, pred_ids, k),
        }
        for k in k_values
    ]


def plot_precision_recall_ndcg(results_df: pd.DataFrame, title: str, filename: str):
    """
    Helper function to create a chart of the evaluation metrics
    """
    # Get the average precision, recall, and ndcg at K
    summary = results_df.groupby("k").agg({
        "precision": "mean",
        "recall": "mean",
        "ndcg": "mean"
    }).reset_index()
    plt.figure(figsize=(10, 6))
    # Create a line for each metric
    plt.plot(summary["k"], summary["precision"], marker='o', label="Precision@K")
    plt.plot(summary["k"], summary["recall"], marker='s', label="Recall@K")
    plt.plot(summary["k"], summary["ndcg"], marker='^', label="NDCG@K")
    # Configure the rest of the chart appearance
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Score")
    plt.xticks(summary["k"])
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    print(f"Plot saved to {filename}")
