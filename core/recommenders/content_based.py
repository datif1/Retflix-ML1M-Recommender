from typing import Any, List, Tuple, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.models import UserMovieRating
from django.contrib.auth.models import User
from data.loader import load_all_data, clean_movie_data, format_recommendations


def build_tfidf_matrix(
    movies_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Any, List[int]]:
    """Builds the TF-IDF matrix using MovieLens 1M genre data only."""

    movies_df = movies_df.copy()

    movies_df["genres_str"] = movies_df["genres"].apply(
        lambda g: " ".join(g) if isinstance(g, list) else str(g).replace("|", " ")
    )

    movies_df["combined_features"] = movies_df["genres_str"].fillna("")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies_df["combined_features"])

    return movies_df, tfidf_matrix, list(movies_df["movieId"])


def recommend_from_rated_list(
    rated: List[Tuple[int, float]],
    movies_df: pd.DataFrame,
    tfidf_matrix,
    movie_ids: List[int],
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    """Recommends movies to a user from a ratings list and TF-IDF matrix."""

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

    similarity_scores = cosine_similarity(profile_vector, tfidf_matrix).flatten()

    seen_ids = [mid for mid, _ in valid_ratings]

    movies_df = movies_df.copy()
    movies_df["similarity"] = similarity_scores

    if exclude_rated:
        candidates = movies_df[~movies_df["movieId"].isin(seen_ids)]
    else:
        candidates = movies_df

    top_movies = candidates.sort_values(by="similarity", ascending=False)

    return format_recommendations(top_movies, top_k)


def recommend_movies(
    user: User,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    """Recommends a list of top K movies for a Django user."""

    movies_df, ratings_df, users_df = load_all_data()

    movies_df = clean_movie_data(movies_df)

    movies_df, tfidf_matrix, movie_ids = build_tfidf_matrix(movies_df)

    ratings = UserMovieRating.objects.filter(user=user)

    rated = [(r.movie.movieId, r.rating) for r in ratings]

    return recommend_from_rated_list(
        rated,
        movies_df,
        tfidf_matrix,
        movie_ids,
        top_k,
        exclude_rated
    )


def recommend_movies_from_csv(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    """
    Returns a list of recommended movies for a MovieLens 1M user ID,
    optionally excluding already rated movies.
    """

    movies_df, all_ratings_df, users_df = load_all_data()

    movies_df = clean_movie_data(movies_df)

    movies_df, tfidf_matrix, movie_ids = build_tfidf_matrix(movies_df)

    user_ratings = ratings_df[ratings_df["userId"] == user_id]

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