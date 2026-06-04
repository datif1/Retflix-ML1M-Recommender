from typing import Dict, List
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans

from core.models import UserMovieRating
from django.contrib.auth.models import User
from data.loader import load_all_data, clean_movie_data, format_recommendations


def normalize_ratings_columns(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts ratings into one common format.

    MovieLens 1M CSV:
        userId, movieId, rating

    Django database:
        user_id, movie__movieId, rating

    Final format:
        user_id, movie_id, rating
    """

    ratings_df = ratings_df.copy()

    if ratings_df.empty:
        return ratings_df

    if "userId" in ratings_df.columns:
        ratings_df = ratings_df.rename(columns={"userId": "user_id"})

    if "movieId" in ratings_df.columns:
        ratings_df = ratings_df.rename(columns={"movieId": "movie_id"})

    if "movie__movieId" in ratings_df.columns:
        ratings_df = ratings_df.rename(columns={"movie__movieId": "movie_id"})

    return ratings_df


def cluster_movies(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    n_clusters: int = 50
) -> pd.DataFrame:
    """
    Clusters movies based on genres and average ratings.

    Since MovieLens 1M does not contain tags, this version uses:
        - genre features
        - average movie rating
    """

    movies_df = movies_df.copy()
    ratings_df = normalize_ratings_columns(ratings_df)

    mlb = MultiLabelBinarizer()

    genre_matrix = mlb.fit_transform(movies_df["genres"])

    genre_df = pd.DataFrame(
        genre_matrix,
        columns=mlb.classes_,
        index=movies_df.index
    )

    genre_df["movieId"] = movies_df["movieId"].values
    genre_df["title"] = movies_df["title"].values

    avg_ratings = (
        ratings_df
        .groupby("movie_id")["rating"]
        .mean()
        .reset_index()
        .rename(columns={"movie_id": "movieId", "rating": "avg_rating"})
    )

    genre_df = genre_df.merge(avg_ratings, on="movieId", how="left")

    genre_df["avg_rating"] = genre_df["avg_rating"].fillna(3.0)

    combined_features = genre_df.drop(columns=["movieId", "title"])

    combined_features.columns = combined_features.columns.astype(str)

    scaled_features = StandardScaler().fit_transform(combined_features)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=1,
        n_init=10
    )

    genre_df["cluster"] = kmeans.fit_predict(scaled_features)

    return movies_df.merge(
        genre_df[["movieId", "cluster"]],
        on="movieId",
        how="left"
    )


def get_cluster_candidates(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> pd.DataFrame:
    """
    Gets recommendation candidates from clusters where the user liked movies.

    A movie is considered liked if rating >= 4.0.
    """

    movies_df, all_ratings_df, users_df = load_all_data()

    movies_df = clean_movie_data(movies_df)

    ratings_df = normalize_ratings_columns(ratings_df)

    if ratings_df.empty:
        return pd.DataFrame()

    movies_df = cluster_movies(
        movies_df,
        all_ratings_df
    )

    user_ratings = ratings_df[ratings_df["user_id"] == user_id]

    if user_ratings.empty:
        return pd.DataFrame()

    liked_ids = set(
        user_ratings[user_ratings["rating"] >= 4.0]["movie_id"]
    )

    rated_ids = set(user_ratings["movie_id"])

    if not liked_ids:
        return pd.DataFrame()

    liked_clusters = (
        movies_df[movies_df["movieId"].isin(liked_ids)]["cluster"]
        .value_counts()
        .index
        .tolist()
    )

    candidates = movies_df[movies_df["cluster"].isin(liked_clusters)].copy()

    if exclude_rated:
        candidates = candidates[~candidates["movieId"].isin(rated_ids)]

    candidates = candidates.sort_values("cluster")

    return candidates.head(top_k)


def recommend_movies(
    user: User,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    """
    Returns cluster-based recommendations for a Django user.
    """

    ratings_qs = UserMovieRating.objects.filter(user=user).values(
        "user_id",
        "movie__movieId",
        "rating"
    )

    ratings_df = pd.DataFrame(list(ratings_qs))

    candidates = get_cluster_candidates(
        user.id,
        ratings_df,
        top_k=top_k,
        exclude_rated=exclude_rated
    )

    if candidates.empty:
        return []

    return format_recommendations(candidates, top_k)


def recommend_movies_from_csv(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    """
    Returns cluster-based recommendations for a MovieLens 1M user.
    """

    candidates = get_cluster_candidates(
        user_id,
        ratings_df,
        top_k=top_k,
        exclude_rated=exclude_rated
    )

    if candidates.empty:
        return []

    return [
        {"movieId": row["movieId"]}
        for _, row in candidates.iterrows()
    ]