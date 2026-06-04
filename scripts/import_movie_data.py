import os
import sys
import re
import pandas as pd
import django

# Setup Django so the models are available
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "app.settings")
django.setup()

from core.models import Movie, Genre

def load_and_clean_movie_data() -> pd.DataFrame:
    """
    Loads the data and performs some processing on the movies, 
    links, and poster data
    """
    movies_df = pd.read_csv("data/movies.csv")
    posters_df = pd.read_csv("data/tmdb_posters.csv")

    # Extract the year of the movie from the title
    movies_df["year"] = movies_df["title"].str.extract(r"\((\d{4})\)", expand=False)

    # Extract the title and strip whitespace
    movies_df["title"] = (
       movies_df["title"]
       .str.replace(r"\(\d{4}\)", "", regex=True)
       .str.strip()
    )

    # Separate genres into a list
    movies_df["genres"] = movies_df["genres"].apply(
        lambda s: s.split("|") if pd.notnull(s) else []
    )

    # Merge poster URLs
    merged_df = movies_df.merge(
    	posters_df[["movieId", "poster_url"]],
        on="movieId",
        how="left"
    )

    return merged_df

def ensure_genres_exist(all_data_df: pd.DataFrame) -> dict:
    """
    Makes sure that all the genres are created if they do not currently
    exist in the database
    """
    # Get all the genres from the data
    genre_set = set(
        g for genre_list in all_data_df["genres"] for g in genre_list
    )
    # Get all existing genres from the database
    existing = {
        g.name: g for g in Genre.objects.all()
    }
    # Find out which genres need to be added
    missing = genre_set - set(existing.keys())

    if missing:
        # Create all the missing genres
        Genre.objects.bulk_create([Genre(name=name) for name in missing])
        existing.update(
            {
                g.name: g for g in Genre.objects.filter(name__in=missing)
            }
        )
    return existing


def create_movies(all_data_df: pd.DataFrame) -> dict:
    """Creates Movie entries and returns a map of movieId -> Movie"""
    # Get existing movies
    existing_movie_ids = set(Movie.objects.values_list("movieId", flat=True))
    movie_objs = []
    # Go through the data and create the movie objects in the database
    for _, row in all_data_df.iterrows():
        if pd.isna(row["title"]) or row["movieId"] in existing_movie_ids:
            continue

        movie_objs.append(
            Movie(
                movieId=row["movieId"],
                title=row["title"],
                year=row["year"] or "",
                poster_url=row.get("poster_url") if pd.notna(row.get("poster_url")) else None,
            )
        )
    # Create movies in batches of 500
    Movie.objects.bulk_create(movie_objs, batch_size=500)
    # Return the map
    return {
        m.movieId: m for m in Movie.objects.filter(movieId__in=all_data_df["movieId"].tolist())
    }


def link_movies_to_genres(all_data_df: pd.DataFrame, movie_map: dict, genre_map: dict):
    """Creates M2M relationships between movies and genres"""
    m2m_relations = []

    for _, row in all_data_df.iterrows():
        # Get each movie
        movie = movie_map.get(row["movieId"])
        if not movie:
            continue
        # For each genre, create a many to many relation using through()
        for genre_name in row["genres"]:
            if genre_name.lower() == "(no genres listed)":
                continue

            genre = genre_map.get(genre_name.strip())
            if genre:
                m2m_relations.append(
                    Movie.genres.through(movie_id=movie.id, genre_id=genre.id)
                )
    # Create the Genre objects in the database
    Movie.genres.through.objects.bulk_create(m2m_relations, batch_size=500)
    return len(m2m_relations)


if __name__ == "__main__":
    all_data = load_and_clean_movie_data()
    genre_map = ensure_genres_exist(all_data)
    movie_map = create_movies(all_data)
    link_count = link_movies_to_genres(all_data, movie_map, genre_map)
    print(f"Imported {len(movie_map)} movies and linked {link_count} genres to movies")
