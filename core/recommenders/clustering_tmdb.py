from typing import List, Dict
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from data.loader import clean_movie_data, format_recommendations
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data")


def load_tmdb_data():
    movies_df = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))
    ratings_df = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))
    tmdb_df = pd.read_csv(os.path.join(DATA_PATH, "tmdb_metadata.csv"))

    return movies_df, ratings_df, tmdb_df


def build_cluster_features(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    tmdb_df: pd.DataFrame
):
    movies_df = clean_movie_data(movies_df)

    avg_ratings = (
        ratings_df
        .groupby("movieId")["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "avg_rating"})
    )

    merged_df = movies_df.merge(
        avg_ratings,
        on="movieId",
        how="left"
    )

    merged_df = merged_df.merge(
        tmdb_df,
        on="movieId",
        how="left",
        suffixes=("", "_tmdb")
    )

    if "avg_rating" not in merged_df.columns:
        merged_df["avg_rating"] = 0

    merged_df["avg_rating"] = merged_df["avg_rating"].fillna(0)

    for col in ["overview", "keywords", "popularity", "vote_average"]:
        if col not in merged_df.columns:
            merged_df[col] = 0 if col in ["popularity", "vote_average"] else ""

    merged_df["popularity"] = pd.to_numeric(
        merged_df["popularity"],
        errors="coerce"
    ).fillna(0)

    merged_df["vote_average"] = pd.to_numeric(
        merged_df["vote_average"],
        errors="coerce"
    ).fillna(0)

    merged_df["genres_text"] = merged_df["genres"].apply(
        lambda g: " ".join(g) if isinstance(g, list) else str(g)
    )

    merged_df["combined_text"] = (
        merged_df["genres_text"].fillna("") + " " +
        merged_df["overview"].fillna("") + " " +
        merged_df["keywords"].fillna("")
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=1000,
        min_df=2,
        max_df=0.8
    )
    
    text_features = tfidf.fit_transform(
        merged_df["combined_text"]
    ).toarray()

    numeric_features = merged_df[
        ["avg_rating", "popularity", "vote_average"]
    ].fillna(0)

    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_features)

    final_features = np.hstack([
        text_features,
        numeric_scaled
    ])

    return merged_df, final_features

def recommend_movies_from_csv(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:

    movies_df, all_ratings_df, tmdb_df = load_tmdb_data()

    merged_df, features = build_cluster_features(
        movies_df,
        all_ratings_df,
        tmdb_df
    )

    kmeans = KMeans(
        n_clusters=20,
        random_state=42,
        n_init=3
    )

    merged_df["cluster"] = kmeans.fit_predict(features)

    user_ratings = ratings_df[
        ratings_df["userId"] == user_id
    ]

    if user_ratings.empty:
        return []

    liked_movies = user_ratings[
        user_ratings["rating"] >= 4.0
    ]["movieId"].tolist()

    liked_clusters = merged_df[
        merged_df["movieId"].isin(liked_movies)
    ]["cluster"]

    if liked_clusters.empty:
        return []

    favorite_cluster = liked_clusters.mode()[0]

    recommendations = merged_df[
        merged_df["cluster"] == favorite_cluster
    ].copy()

    if exclude_rated:
        recommendations = recommendations[
            ~recommendations["movieId"].isin(
                user_ratings["movieId"]
            )
        ]

    recommendations = recommendations.sort_values(
        by=["vote_average", "popularity"],
        ascending=False
    )

    return format_recommendations(
        recommendations,
        top_k
    )