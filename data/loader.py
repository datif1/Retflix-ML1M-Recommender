import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")
PLACEHOLDER_IMAGE = "https://via.placeholder.com/500x750?text=No+Image"

def load_all_data():
    """Reads the movie, tag, link, and poster data all in one spot"""
    return (
        pd.read_csv(os.path.join(DATA_PATH, "movies.csv")),
        pd.read_csv(os.path.join(DATA_PATH, "tags.csv")),
        pd.read_csv(os.path.join(DATA_PATH, "links.csv")),
        pd.read_csv(os.path.join(DATA_PATH, "poster_cache.csv")),
    )

def clean_movie_data(movie_df):
    """Cleans the movie data by removing the year from title and removing whitespace"""
    movie_df = movie_df.copy()
    # Remove year from title and remove whitespace
    movie_df["title"] = movie_df["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()
    # Separate the genres since it will all be one string to begin with
    movie_df["genres"] = movie_df["genres"].apply(lambda x: x.split("|") if pd.notnull(x) else [])
    # Remove any duplicate movieIds
    return movie_df.drop_duplicates(subset=["movieId"]).dropna(subset=["title"])

def attach_posters(movies_df, links_df, posters_df):
    """Adds the TMDB poster links into the movie data"""
    # Combine posters df with the links df
    links_with_posters = links_df.merge(posters_df, on="tmdbId", how="left")
    # Combine all into the movies_df
    return movies_df.merge(links_with_posters[["movieId", "poster_url"]], on="movieId", how="left")

def format_recommendations(movies_df, top_k=10):
    """Formats the recommendations so the carousels can be built"""
    return [
        {
            "movieId": row["movieId"],
            "title": row["title"],
            "poster_url": row.get("poster_url") or PLACEHOLDER_IMAGE,
            "tagline": "",
        }
        for _, row in movies_df.head(top_k).iterrows()
    ]
