from typing import Any, List, Tuple, Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.models import UserMovieRating
from django.contrib.auth.models import User
from data.loader import load_all_data, clean_movie_data, attach_posters, format_recommendations


def build_tfidf_matrix(
    movies_df: pd.DataFrame, 
    tags_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Any, List[int]]:
    """Builds the TF-IDF matrix for tag and genre data"""
    # Group all tags by movie and join 
    tag_data = tags_df.groupby("movieId")["tag"].apply(
        lambda x: " ".join(str(i) for i in x)
    ).reset_index()
    # Combine the movies data and tag data
    movies_df = movies_df.merge(tag_data, on="movieId", how="left")
    # Combine all genres into a string
    movies_df["genres_str"] = movies_df["genres"].apply(
        lambda g: " ".join(g) if isinstance(g, list) else ""
    )
    # Put all the genre and tag data into one feature column
    movies_df["combined_features"] = movies_df["genres_str"].fillna("") + " " + movies_df["tag"].fillna("")
    # Initialize the vectorizer
    tfidf = TfidfVectorizer(stop_words="english")
    # Create the matrix using the combined features
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
    """Recommends movies to an user from a ratings list and tfidf matrix"""
    # User must have ratings to get recommendations
    if not rated:
        return []
    # Create the profile vector as a weighted sum of ratings and the tag genre matrix
    profile_vector = sum(
        float(rating) * tfidf_matrix[movie_ids.index(mid)]
        for mid, rating in rated if mid in movie_ids
    )
    # Calculate the similarity between profile vector and each movie in the matrix
    similarity_scores = cosine_similarity(profile_vector, tfidf_matrix).flatten()
    # Movie ids that were already rated
    seen_ids = [mid for mid, _ in rated]
    # Add a column for the similarity scores
    movies_df["similarity"] = similarity_scores
    # Optionally, exclude already rated movies from recommendations
    if exclude_rated:
        candidates = movies_df[~movies_df["movieId"].isin(seen_ids)]
    else:
        candidates = movies_df
    # Get the top k recommendations
    top_movies = candidates.sort_values(by="similarity", ascending=False)
    # Return the formatted list of recommendations
    return format_recommendations(top_movies, top_k)


def recommend_movies(
    user: User,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    """Recommends a list of top k movies for a django user"""
    # Get all the data from the MovieLens data files
    movies_df, tags_df, links_df, posters_df = load_all_data()
    # Process the data and clean to an usable format
    movies_df = clean_movie_data(movies_df)
    # Build the tag, genre TF-IDF matrix
    movies_df, tfidf_matrix, movie_ids = build_tfidf_matrix(movies_df, tags_df)
    # Add the poster image urls to the movie data as well
    movies_df = attach_posters(movies_df, links_df, posters_df)
    # Get the ratings from the Django user
    ratings = UserMovieRating.objects.filter(user=user)
    # Get a list of movies the user has rated
    rated = [(r.movie.movieId, r.rating) for r in ratings]
    # Get recommendations
    return recommend_from_rated_list(rated, movies_df, tfidf_matrix, movie_ids, top_k, exclude_rated)


def recommend_movies_from_csv(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    """
    Returns a list of recommended movies for a MovieLens user ID,
    optionally excluding already rated movies.
    """
    # Get all the data from the MovieLens data files
    movies_df, tags_df, links_df, posters_df = load_all_data()
    # Process the data and clean to an usable format
    movies_df = clean_movie_data(movies_df)
    # Build the tag, genre TF-IDF matrix
    movies_df, tfidf_matrix, movie_ids = build_tfidf_matrix(movies_df, tags_df)
    # Add the poster image urls to the movie data as well
    movies_df = attach_posters(movies_df, links_df, posters_df)
    # Get all movies rated by the user
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    # Convert to a list
    rated = [
        (row["movie__movieId"], row["rating"])
        for _, row in user_ratings.iterrows()
    ]
    # Get Recommendations
    return recommend_from_rated_list(rated, movies_df, tfidf_matrix, movie_ids, top_k, exclude_rated)
