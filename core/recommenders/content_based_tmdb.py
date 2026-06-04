from typing import Any, List, Tuple, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data.loader import clean_movie_data, format_recommendations
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data")


def load_tmdb_data():
    movies_df = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))
    tmdb_df = pd.read_csv(os.path.join(DATA_PATH, "tmdb_metadata.csv"))

    return movies_df, tmdb_df


def build_tfidf_matrix(
    movies_df: pd.DataFrame,
    tmdb_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Any, List[int]]:
    """
    Builds TF-IDF matrix using enriched TMDB metadata.
    """

    movies_df = clean_movie_data(movies_df)

    merged_df = movies_df.merge(
        tmdb_df,
        on="movieId",
        how="left",
        suffixes=("", "_tmdb")
    )

    if "poster_url_tmdb" in merged_df.columns:
        merged_df["poster_url"] = merged_df["poster_url_tmdb"]

    text_columns = [
        "genres",
        "overview",
        "keywords",
        "cast",
        "director",
        "tagline",
    ]

    merged_df["genres_text"] = merged_df["genres"].apply(
        lambda g: " ".join(g) if isinstance(g, list) else str(g)
    )

    for col in text_columns:
        if col not in merged_df.columns:
            merged_df[col] = ""

    merged_df["combined_features"] = (
        merged_df["genres_text"].fillna("") + " " +
        merged_df["overview"].fillna("") + " " +
        merged_df["keywords"].fillna("") + " " +
        merged_df["cast"].fillna("") + " " +
        merged_df["director"].fillna("") + " " +
        merged_df["tagline"].fillna("")
    )

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=15000
    )

    tfidf_matrix = tfidf.fit_transform(
        merged_df["combined_features"]
    )

    return merged_df, tfidf_matrix, list(merged_df["movieId"])


def recommend_from_rated_list(
    rated: List[Tuple[int, float]],
    movies_df: pd.DataFrame,
    tfidf_matrix,
    movie_ids: List[int],
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:

    if not rated:
        return []

    valid_ratings = [
        (mid, rating)
        for mid, rating in rated
        if mid in movie_ids
    ]

    if not valid_ratings:
        return []

    profile_vector = sum(
        float(rating) * tfidf_matrix[movie_ids.index(mid)]
        for mid, rating in valid_ratings
    )

    similarity_scores = cosine_similarity(
        profile_vector,
        tfidf_matrix
    ).flatten()

    seen_ids = [mid for mid, _ in valid_ratings]

    movies_df = movies_df.copy()
    movies_df["similarity"] = similarity_scores

    if exclude_rated:
        candidates = movies_df[
            ~movies_df["movieId"].isin(seen_ids)
        ]
    else:
        candidates = movies_df

    top_movies = candidates.sort_values(
        by="similarity",
        ascending=False
    )

    return format_recommendations(top_movies, top_k)


def recommend_movies_from_csv(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:

    movies_df, tmdb_df = load_tmdb_data()

    movies_df, tfidf_matrix, movie_ids = build_tfidf_matrix(
        movies_df,
        tmdb_df
    )

    user_ratings = ratings_df[
        ratings_df["userId"] == user_id
    ]

    rated = [
        (row["movieId"], row["rating"])
        for _, row in user_ratings.iterrows()
    ]

    return recommend_from_rated_list(
        rated,
        movies_df,
        tfidf_matrix,
        movie_ids,
        top_k,
        exclude_rated
    )