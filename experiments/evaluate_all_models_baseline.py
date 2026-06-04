import os
import sys
import math
import time
import django
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")

django.setup()

from core.recommenders.content_based import recommend_movies_from_csv as content_based
from core.recommenders.collaborative_based import recommend_movies_from_csv as collaborative_based
from core.recommenders.clustering import recommend_movies_from_csv as clustering_based
from core.recommenders.svd_recommender import recommend_movies_from_csv as svd_based


DATA_PATH = os.path.join(BASE_DIR, "data")
OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "experiments",
    "results",
    "baseline_ml1m_only"
)
 
os.makedirs(OUTPUT_PATH, exist_ok=True)

K_VALUES = [5, 10, 20, 40, 80]
RELEVANCE_THRESHOLD = 4.0
MAX_USERS = 200   # use 200 first for testing; later set to None for all users


def dcg_at_k(relevances):
    return sum(
        rel / math.log2(idx + 2)
        for idx, rel in enumerate(relevances)
    )


def ndcg_at_k(recommended_ids, relevant_ids, k):
    relevances = [
        1 if movie_id in relevant_ids else 0
        for movie_id in recommended_ids[:k]
    ]

    dcg = dcg_at_k(relevances)

    ideal_relevances = sorted(relevances, reverse=True)

    idcg = dcg_at_k(ideal_relevances)

    return dcg / idcg if idcg > 0 else 0


def precision_at_k(recommended_ids, relevant_ids, k):
    recommended_k = recommended_ids[:k]

    if not recommended_k:
        return 0

    hits = len(set(recommended_k) & relevant_ids)

    return hits / k


def recall_at_k(recommended_ids, relevant_ids, k):
    if not relevant_ids:
        return 0

    recommended_k = recommended_ids[:k]

    hits = len(set(recommended_k) & relevant_ids)

    return hits / len(relevant_ids)


def extract_movie_ids(recommendations):
    movie_ids = []

    for rec in recommendations:
        if "movieId" in rec:
            movie_ids.append(rec["movieId"])

    return movie_ids


def evaluate_model(model_name, model_function, train_df, test_df, users):
    results = []

    print(f"\nEvaluating {model_name}...")

    start_time = time.time()

    for count, user_id in enumerate(users, start=1):
        if count % 25 == 0:
            print(f"{model_name}: evaluated {count}/{len(users)} users")

        user_test = test_df[test_df["userId"] == user_id]

        relevant_ids = set(
            user_test[user_test["rating"] >= RELEVANCE_THRESHOLD]["movieId"]
        )

        if not relevant_ids:
            continue

        try:
            recommendations = model_function(
                user_id=user_id,
                ratings_df=train_df,
                top_k=max(K_VALUES),
                exclude_rated=True
            )

            recommended_ids = extract_movie_ids(recommendations)

            for k in K_VALUES:
                results.append({
                    "model": model_name,
                    "userId": user_id,
                    "K": k,
                    "precision": precision_at_k(recommended_ids, relevant_ids, k),
                    "recall": recall_at_k(recommended_ids, relevant_ids, k),
                    "ndcg": ndcg_at_k(recommended_ids, relevant_ids, k),
                    "num_relevant_test_items": len(relevant_ids),
                    "num_recommendations": len(recommended_ids),
                })

        except Exception as e:
            print(f"Error for {model_name}, user {user_id}: {e}")

    elapsed = time.time() - start_time

    print(f"{model_name} completed in {elapsed:.2f} seconds.")

    return pd.DataFrame(results)


def main():
    train_df = pd.read_csv(os.path.join(DATA_PATH, "ratings_train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_PATH, "ratings_test.csv"))

    common_users = sorted(
        set(train_df["userId"].unique()) & set(test_df["userId"].unique())
    )

    if MAX_USERS is not None:
        np.random.seed(42)
        common_users = np.random.choice(
            common_users,
            size=min(MAX_USERS, len(common_users)),
            replace=False
        )

    common_users = list(common_users)

    print("Total users being evaluated:", len(common_users))

    models = {
        "Content-Based": content_based,
        "Collaborative Filtering": collaborative_based,
        "Clustering": clustering_based,
        "SVD Matrix Factorization": svd_based,
    }

    all_results = []

    for model_name, model_function in models.items():
        model_results = evaluate_model(
            model_name,
            model_function,
            train_df,
            test_df,
            common_users
        )

        model_file = os.path.join(
            OUTPUT_PATH,
            f"{model_name.lower().replace(' ', '_')}_results.csv"
        )

        model_results.to_csv(model_file, index=False)

        all_results.append(model_results)

    final_results = pd.concat(all_results, ignore_index=True)

    final_results.to_csv(
        os.path.join(OUTPUT_PATH, "all_model_results.csv"),
        index=False
    )

    summary = (
        final_results
        .groupby(["model", "K"])
        [["precision", "recall", "ndcg"]]
        .mean()
        .reset_index()
    )

    summary.to_csv(
        os.path.join(OUTPUT_PATH, "summary_results.csv"),
        index=False
    )

    print("\nEvaluation complete.")
    print(summary)


if __name__ == "__main__":
    main()