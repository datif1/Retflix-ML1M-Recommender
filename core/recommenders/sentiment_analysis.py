from typing import List, Dict
import pandas as pd
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from django.contrib.auth.models import User
from core.models import UserMovieRating
from data.loader import load_all_data, clean_movie_data, attach_posters, format_recommendations

# Downloads the stopwords from the vader library 
nltk.download("stopwords")
# Downloads the lexicon vader uses for sentiment analysis
nltk.download("vader_lexicon")


def clean_and_score_tags(tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans up the tag data and applies the average sentiment analysis score for each tag
    returning a DataFrame with the new column for sentiment score.
    """
    # Copy the DataFrame so the original isn't modified
    tags_df = tags_df.copy()
    # Converts tags to lowercase and strips the whitespace if there is any
    tags_df["tag"] = tags_df["tag"].astype(str).str.lower().str.strip()
    # Exclude any empty or null tags
    tags_df = tags_df[tags_df["tag"].notnull() & (tags_df["tag"].str.len() > 0)]
    # Create a set of stopwords from the NLTK library
    stop_words = set(stopwords.words("english"))
    # Extracts words from tags using regex, then removes stopwords and joins the tags back together
    tags_df["tag"] = tags_df["tag"].apply(
        lambda s: " ".join(
            [w for w in re.findall(r"\b\w+\b", s) if w not in stop_words]
        )
    )
    # Initialize the vader sentiment analysis tool
    analyzer = SentimentIntensityAnalyzer()
    # Calculates that average sentiment score for each word and creates new column in the DataFrame
    tags_df["sentiment"] = tags_df["tag"].apply(lambda s: analyzer.polarity_scores(s)["compound"])
    return tags_df


def build_sentiment_scores(movies_df: pd.DataFrame, tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the movies DataFrame with all of the average sentiment scores
    """
    # Get just the movie id and tag columns from the tag data
    tag_data = tags_df[["movieId", "tag"]]
    # Get just the movie id and title from the movies data and rename title column to tag
    title_as_tag = movies_df[["movieId", "title"]].rename(columns={"title": "tag"})
    # Convert to lowercase and strip whitespace from title
    title_as_tag["tag"] = title_as_tag["tag"].str.lower().str.strip()
    # Adds the title as tags to the actual tag data. Now the title is considered a tag as well
    # so it can handle cases where there aren't any tags for a movie
    tags_combined = pd.concat([tag_data, title_as_tag], ignore_index=True)
    # Get the sentiment column for all of the movie tags
    scored_tags = clean_and_score_tags(tags_combined)
    # Get the average sentiment score for each movie
    movie_sentiment = scored_tags.groupby("movieId")["sentiment"].mean().reset_index()
    # Reset the columns to account for the avg sentiment
    movie_sentiment.columns = ["movieId", "avg_movie_sentiment"]
    # Adds the sentiment column to the overall movie data
    return movies_df.merge(movie_sentiment, on="movieId", how="left")


def recommend_from_ratings_df(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    # Get all the data from the MovieLens data files
    movies_df, tags_df, links_df, posters_df = load_all_data()
    # Clean up the movie data 
    movies_df = clean_movie_data(movies_df)
    # Get the average sentiment scores for all of the movies
    movies_df = build_sentiment_scores(movies_df, tags_df)
    # Attach the poster urls for the movies for the display in the django app
    movies_df = attach_posters(movies_df, links_df, posters_df)
    # Get all of the ratings from just the user
    user_ratings_df = ratings_df[ratings_df["user_id"] == user_id].rename(columns={"movie__movieId": "movieId"})
    # If no user ratings return an empty list
    if user_ratings_df.empty:
        return []
    # Create a map of the movieIds and avg movie sentiment
    sentiment_map = movies_df.set_index("movieId")["avg_movie_sentiment"]
    # Create a copy of the ratings DataFrame so the original doesn't change
    user_ratings = user_ratings_df.copy()
    # Add the sentiment scores to the user rated movies
    user_ratings["avg_movie_sentiment"] = user_ratings["movieId"].map(sentiment_map)
    # Get the ratings that the user has rated 4.0 or above
    positive_ratings = user_ratings[user_ratings["rating"] >= 4.0]
    # Get the average sentiment score for all of the movies the user has rated positively
    # this is the user preference profile
    user_score = positive_ratings["avg_movie_sentiment"].mean()
    # Get a set of the movie ids that the user has rated
    rated_ids = set(user_ratings_df["movieId"])
    # Optionally, exclude movies the user has already rated
    if exclude_rated:
        candidates = movies_df[~movies_df["movieId"].isin(rated_ids)].copy()
    else:
        candidates = movies_df.copy()
    # Calculate the difference between sentiment scores and the average user sentiment
    candidates["sentiment_diff"] = abs(candidates["avg_movie_sentiment"] - user_score)
    # Sort by the least different movies
    top_movies = candidates.sort_values("sentiment_diff")
    # Return the top-k movies with least difference
    return format_recommendations(top_movies, top_k)


def recommend_movies(user: User, top_k: int = 10) -> List[Dict]:
    """
    Returns a list of dicts for the movie recommendations. The recommendations are for
    the django user. 
    """
    # Get all the ratings for the django user
    ratings_qs = UserMovieRating.objects.filter(user=user).values("user_id", "movie__movieId", "rating")
    # Convert the QuerySet to a DataFrame
    ratings_df = pd.DataFrame(list(ratings_qs))
    # Get the recommendations
    return recommend_from_ratings_df(user.id, ratings_df, top_k)


def recommend_movies_from_csv(
    user_id: int,
    ratings_df: pd.DataFrame,
    top_k: int = 10,
    exclude_rated: bool = True
) -> List[Dict]:
    """
    Same as the recommend_movies function, but meant for the users in the MovieLens dataset
    not the django user.
    """
    return recommend_from_ratings_df(user_id, ratings_df, top_k, exclude_rated=exclude_rated)
