from typing import Dict, List
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from core.models import UserMovieRating
from django.contrib.auth.models import User
from data.loader import load_all_data, clean_movie_data, attach_posters, format_recommendations


def cluster_movies(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    tags_df: pd.DataFrame,
    n_clusters: int = 50
) -> pd.DataFrame:
    """
    Performs clustering of movies based on the average ratings, genres, and tag strings
    and ultimately returns a DataFrame with a cluster column which contains the cluster
    id for each movie
    """
    mlb = MultiLabelBinarizer()
    # Create a matrix of each movies' genres in binary vector format
    genre_matrix = mlb.fit_transform(movies_df["genres"])
    # Create a DataFrame of the genre matrix, setting the columns to the name of the genre
    genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_)
    # Adds the movieId as a column to the genre DataFrame
    genre_df["movieId"] = movies_df["movieId"].values
    # Adds the title to the genre DataFrame
    genre_df["title"] = movies_df["title"].values
    # Group by movie and calculate the average rating for each movie
    avg_ratings = ratings_df.groupby("movie__movieId")["rating"].mean().reset_index()
    # Rename the columns for readability
    avg_ratings.columns = ["movieId", "avg_rating"]
    # Combines the genre DataFrame and the ratings DataFrame to put everything together
    genre_df = genre_df.merge(avg_ratings, on="movieId", how="left")
    # Fill any missing ratings with a value of 3.0 so we can keep things neutral
    genre_df["avg_rating"] = genre_df["avg_rating"].fillna(3.0)
    # Groups all of the tags by movie then applies a function to combine the tags to a single string
    # Also reorders by movie id and fills null values with a blank string
    tag_strings = tags_df.groupby("movieId")["tag"].apply(
        lambda tags: " ".join(tags)
    ).reindex(genre_df["movieId"]).fillna("")
    # Initialize a vocabulary of up to 100 of the most common tags for movies in vectors
    tfidf = TfidfVectorizer(max_features=100)
    # Learns the vocabulary and creates the sparse matrix of numeric values for the matrix
    tag_features = tfidf.fit_transform(tag_strings).toarray()
    # Convert the numpy arrau for tag features into DataFrame and index on the same index as the genre DataFrame
    tag_df = pd.DataFrame(tag_features, index=genre_df.index)
    # Combines the genres, avg ratings, and tag data into one vector for each movie
    combined_features = pd.concat([genre_df.drop(columns=["movieId", "title"]), tag_df], axis=1)
    # Prevent mismatched datatypes in the columns after concatenation
    combined_features.columns = combined_features.columns.astype(str)
    # Normalize the data so the ratings don't dominate the KMeans distance calculations
    scaled = StandardScaler().fit_transform(combined_features)
    # Initialize the KMeans cluster with number of clusters, random seed value, and 
    kmeans = KMeans(n_clusters=n_clusters, random_state=1, n_init=10)
    # Adds a column to the genre_df with the assigned clusters resulting from KMeans
    genre_df["cluster"] = kmeans.fit_predict(scaled)
    # Returns the movies DataFrame with the cluster values added
    return movies_df.merge(genre_df[["movieId", "cluster"]], on="movieId", how="left")


def get_cluster_candidates(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> pd.DataFrame:
    """
    Gets the recommendation options from the clusters that the user has liked movies in. The
    idea is that if an user has liked a movie in a particular cluster, then other movies in that
    cluster might be relevant to the user. 
    """
    # Get all the data from the MovieLens data files
    movies_df, tags_df, links_df, posters_df = load_all_data()
    # Process the data and clean to an usable format
    movies_df = clean_movie_data(movies_df)
    # Add the cluster column with each movies cluster id
    movies_df = cluster_movies(movies_df, ratings_df, tags_df)
    # Add the poster image urls to the movie data as well
    movies_df = attach_posters(movies_df, links_df, posters_df)
    # Get only the ratings for the user
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    # Get a set of movie ids for movies the user has rated 4.0 or above
    liked_ids = set(user_ratings[user_ratings["rating"] >= 4.0]["movie__movieId"])
    # Set of all the movie ids an user has rated
    rated_ids = set(user_ratings["movie__movieId"])
    # Return empty DataFrame if there are no movies an user has liked because
    # there would be nothing to base the clustering on
    if not liked_ids:
        return pd.DataFrame()
    # Find out which clusters the user has liked movies in
    liked_clusters = (
        movies_df[movies_df["movieId"].isin(liked_ids)]["cluster"]
        .value_counts().index.tolist()
    )
    # Get all movies that are in the liked clusters
    candidates = movies_df[movies_df["cluster"].isin(liked_clusters)]
    # Optionally, exclude movies that the user has already rated from the recommendations
    if exclude_rated:
        candidates = candidates[~candidates["movieId"].isin(rated_ids)]

    candidates = candidates.sort_values("cluster")
    return candidates.head(top_k)


def recommend_movies(user: User, top_k: int = 10) -> List[Dict]:
    """
    Returns the list of movies for recommendation to the django user. The return format 
    is a list of dicts that will be used to generate the carousel for the predictions in 
    the user interface.
    """
    # Query the database for all user-movie ratings
    ratings_qs = UserMovieRating.objects.filter(user=user).values("user_id", "movie__movieId", "rating")
    # Create a DataFrame for all the ratings
    ratings_df = pd.DataFrame(list(ratings_qs))
    # Fetch movies that are from user-liked clusters
    candidates = get_cluster_candidates(user.id, ratings_df, top_k)
    # Return empty list if there could not be any recommendations
    if candidates.empty:
        return []
    # Return the list of recommended movies in the format for the carousel
    return format_recommendations(candidates, top_k)


def recommend_movies_from_csv(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = False
) -> list[dict]:
    """
    Function the same as the recommend_movies function, but instead of using a
    Django user for the basis of recommendations, this allows for using the MovieLens
    users to make recommendations. Used only for the experiments.
    """
    candidates = get_cluster_candidates(
        user_id,
        ratings_df,
        top_k=top_k,
        exclude_rated=exclude_rated
    )
    # Returns a simplified structure with just movie id because this won't be rendered
    # in the browser
    return [
        {"movieId": row["movieId"]} 
        for _, row in candidates.iterrows()
    ]
