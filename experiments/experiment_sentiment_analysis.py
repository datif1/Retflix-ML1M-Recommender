import os
import sys
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


# Environment Setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
load_dotenv()
# Setup Django so the models are available
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")
import django
django.setup()
# Import the clustering model and helper functions
from core.recommenders.sentiment_analysis import recommend_movies_from_csv
from experiments.experiment_helpers import evaluate_user_at_ks, plot_precision_recall_ndcg


def load_ratings():
    path = os.path.join(PROJECT_ROOT, "data", "ratings.csv")
    ratings = pd.read_csv(path)
    return ratings.rename(columns={"userId": "user_id", "movieId": "movie__movieId"})


if __name__ == "__main__":
    # Get all ratings data
    ratings_df = load_ratings()
    # Get all unique users
    user_ids = ratings_df["user_id"].unique()
    # Initialize the list of K values to get calculations at
    k_values = [5, 10, 20, 40, 80, 160]
    # Accumulate all the results for each user
    all_results = []
    # Define the recommender using imported function
    def recommender(user_id, ratings_df, top_k):
        return recommend_movies_from_csv(user_id, ratings_df, top_k=top_k, exclude_rated=False)

    print(f"Evaluating {len(user_ids)} users...")
    # For each user, evaluate at the specified K values and save results
    for user_id in tqdm(user_ids, desc="Evaluating users"):
        user_results = evaluate_user_at_ks(
            user_id, ratings_df, k_values, recommender
        )
        all_results.extend(user_results)
    # Turn all results into a DataFrame
    results_df = pd.DataFrame(all_results)
    # Save the results to a CSV file
    results_df.to_csv("sentiment_eval_results.csv", index=False)
    print("Results saved to sentiment_eval_results.csv")
    # Plot the metrics in a line chart
    plot_precision_recall_ndcg(
        results_df,
        title="Sentiment Analysis Recommender Performance",
        filename="sentiment_precision_recall_ndcg.png"
    )
    # Group by K and compute the mean of each metric
    summary_df = (
        results_df.groupby("k")[["precision", "recall", "ndcg"]]
        .mean()
        .reset_index()
        .sort_values("k")
    )
    print(summary_df)
    # Save the summary to CSV
    summary_path = "sentiment_eval_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved to {summary_path}")
