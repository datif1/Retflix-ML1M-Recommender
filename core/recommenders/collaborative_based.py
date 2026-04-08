from typing import List, Dict
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from core.models import UserMovieRating
from django.contrib.auth.models import User
from data.loader import load_all_data, clean_movie_data, attach_posters, format_recommendations


def build_user_item_matrix(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds the user-item matrix where rows are users and columns are movie ids
    Ratings are filled with 0 for items that don't have any ratings
    """
    if ratings_df.empty:
        return pd.DataFrame()
    return ratings_df.pivot_table(index="user_id", columns="movie__movieId", values="rating").fillna(0)


def recommend_from_user_item_matrix(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    """
    Generates recommendations based on user similarity using a cosine 
    similarity weighted averages
    """
    # Get all the data from the MovieLens data files
    movies_df, tags_df, links_df, posters_df = load_all_data()
    # Process the data and clean to an usable format
    movies_df = clean_movie_data(movies_df)
    # Add the poster image urls to the movie data as well
    movies_df = attach_posters(movies_df, links_df, posters_df)
    # Get the user item matrix
    user_item_matrix = build_user_item_matrix(ratings_df)
    # Return empty list if user not in the matrix
    if user_id not in user_item_matrix.index:
        return []
    # Calculate cosine similarity between the target user and all other users
    user_vector = user_item_matrix.loc[[user_id]]
    similarity_scores = cosine_similarity(user_vector, user_item_matrix)[0]
    similarity_series = pd.Series(similarity_scores, index=user_item_matrix.index)
    # Weighted average of ratings by user similarity
    weighted_ratings = user_item_matrix.T.dot(similarity_series) / similarity_series.sum()
    # Optionally, exclude the already rated items from suggestions
    if not exclude_rated:
        rated_movie_ids = set(ratings_df[ratings_df["user_id"] == user_id]["movie__movieId"])
        candidates = weighted_ratings.drop(labels=rated_movie_ids, errors="ignore")
    else:
        candidates = weighted_ratings
    # Get the top-k movie recommendations based on most similar values
    top_movie_ids = candidates.sort_values(ascending=False).head(top_k).index
    top_movies = movies_df[movies_df["movieId"].isin(top_movie_ids)]
    # Format the results
    return format_recommendations(top_movies, top_k)


def recommend_movies(
    user: User,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    """
    Recommends movies for a Django user 
    """
    # Get all user-movie ratings
    ratings_qs = UserMovieRating.objects.all().values("user_id", "movie__movieId", "rating")
    # Convert to DataFrame
    ratings_df = pd.DataFrame(list(ratings_qs))
    # Perform recommendation
    return recommend_from_user_item_matrix(user.id, ratings_df, top_k, exclude_rated)


def recommend_movies_from_csv(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    """
    Recommends movies for a MovieLens user id using the collaborative model,
    optionally excluding movies already rated by the user
    """
    return recommend_from_user_item_matrix(user_id, ratings_df, top_k, exclude_rated)
