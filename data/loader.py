import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")

PLACEHOLDER_IMAGE = "https://via.placeholder.com/500x750?text=No+Image"


def load_all_data():
    """
    Loads MovieLens 1M datasets.
    """

    movies_df = pd.read_csv(
        os.path.join(DATA_PATH, "movies.csv")
    )

    ratings_df = pd.read_csv(
        os.path.join(DATA_PATH, "ratings.csv")
    )

    users_df = pd.read_csv(
        os.path.join(DATA_PATH, "users.csv")
    )

    return movies_df, ratings_df, users_df


def load_tmdb_posters():
    """
    Loads TMDB poster metadata.
    """

    poster_path = os.path.join(DATA_PATH, "tmdb_posters.csv")

    if os.path.exists(poster_path):
        return pd.read_csv(poster_path)

    return pd.DataFrame(columns=["movieId", "poster_url"])


def clean_movie_data(movie_df):
    """
    Cleans movie data.
    """

    movie_df = movie_df.copy()

    movie_df["title"] = (
        movie_df["title"]
        .str.replace(r"\(\d{4}\)", "", regex=True)
        .str.strip()
    )

    movie_df["genres"] = movie_df["genres"].apply(
        lambda x: x.split("|")
        if pd.notnull(x)
        else []
    )

    movie_df = movie_df.drop_duplicates(
        subset=["movieId"]
    )

    movie_df = movie_df.dropna(
        subset=["title"]
    )

    return movie_df


def attach_posters(movies_df):
    """
    Merges TMDB poster URLs into movies dataframe.
    """

    posters_df = load_tmdb_posters()

    merged_df = movies_df.merge(
        posters_df[["movieId", "poster_url"]],
        on="movieId",
        how="left"
    )

    merged_df["poster_url"] = merged_df["poster_url"].fillna(
        PLACEHOLDER_IMAGE
    )

    return merged_df


def format_recommendations(movies_df, top_k=10):
    movies_df = movies_df.copy()

    if "poster_url" not in movies_df.columns:
        movies_df = attach_posters(movies_df)

    if "poster_url_x" in movies_df.columns and "poster_url" not in movies_df.columns:
        movies_df["poster_url"] = movies_df["poster_url_x"]

    if "poster_url_y" in movies_df.columns:
        movies_df["poster_url"] = movies_df.get("poster_url", movies_df["poster_url_y"])
        movies_df["poster_url"] = movies_df["poster_url"].fillna(movies_df["poster_url_y"])

    if "poster_url" not in movies_df.columns:
        movies_df["poster_url"] = PLACEHOLDER_IMAGE

    movies_df["poster_url"] = movies_df["poster_url"].fillna(PLACEHOLDER_IMAGE)
    movies_df["poster_url"] = movies_df["poster_url"].replace("", PLACEHOLDER_IMAGE)

    return [
        {
            "movieId": row["movieId"],
            "title": row["title"],
            "poster_url": row["poster_url"],
            "tagline": "",
        }
        for _, row in movies_df.head(top_k).iterrows()
    ]