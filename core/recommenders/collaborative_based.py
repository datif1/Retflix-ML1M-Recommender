from typing import List, Dict
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from core.models import UserMovieRating
from django.contrib.auth.models import User
from data.loader import load_all_data, clean_movie_data, format_recommendations


DJANGO_USER_OFFSET = 1000000


def normalize_ratings_columns(ratings_df: pd.DataFrame) -> pd.DataFrame:
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


def build_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    ratings_df = normalize_ratings_columns(ratings_df)

    if ratings_df.empty:
        return pd.DataFrame()

    return ratings_df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    ).fillna(0)


def recommend_from_user_item_matrix(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:

    movies_df, _, _ = load_all_data()
    movies_df = clean_movie_data(movies_df)

    ratings_df = normalize_ratings_columns(ratings_df)

    user_item_matrix = build_user_item_matrix(ratings_df)

    if user_item_matrix.empty or user_id not in user_item_matrix.index:
        return []

    user_vector = user_item_matrix.loc[[user_id]]

    similarity_scores = cosine_similarity(user_vector, user_item_matrix)[0]

    similarity_series = pd.Series(
        similarity_scores,
        index=user_item_matrix.index
    )

    similarity_series = similarity_series.drop(labels=[user_id], errors="ignore")

    if similarity_series.sum() == 0:
        return []

    weighted_ratings = (
        user_item_matrix.drop(index=user_id, errors="ignore").T.dot(similarity_series)
        / similarity_series.sum()
    )

    if exclude_rated:
        rated_movie_ids = set(
            ratings_df[ratings_df["user_id"] == user_id]["movie_id"]
        )
        weighted_ratings = weighted_ratings.drop(
            labels=rated_movie_ids,
            errors="ignore"
        )

    top_movie_ids = weighted_ratings.sort_values(ascending=False).head(top_k).index

    top_movies = movies_df[movies_df["movieId"].isin(top_movie_ids)].copy()
    top_movies["score"] = top_movies["movieId"].map(weighted_ratings)
    top_movies = top_movies.sort_values(by="score", ascending=False)

    return format_recommendations(top_movies, top_k)


def recommend_movies(
    user: User,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:

    _, ml_ratings_df, _ = load_all_data()

    ml_ratings_df = normalize_ratings_columns(ml_ratings_df)

    django_user_id = DJANGO_USER_OFFSET + user.id

    django_ratings_qs = UserMovieRating.objects.filter(user=user).values(
        "movie__movieId",
        "rating"
    )

    django_ratings_df = pd.DataFrame(list(django_ratings_qs))

    if django_ratings_df.empty:
        return []

    django_ratings_df = django_ratings_df.rename(
        columns={"movie__movieId": "movie_id"}
    )

    django_ratings_df["user_id"] = django_user_id

    combined_ratings_df = pd.concat(
        [
            ml_ratings_df[["user_id", "movie_id", "rating"]],
            django_ratings_df[["user_id", "movie_id", "rating"]]
        ],
        ignore_index=True
    )

    return recommend_from_user_item_matrix(
        django_user_id,
        combined_ratings_df,
        top_k,
        exclude_rated
    )


def recommend_movies_from_csv(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:

    return recommend_from_user_item_matrix(
        user_id,
        ratings_df,
        top_k,
        exclude_rated
    )