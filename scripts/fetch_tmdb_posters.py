import os
import re
import time
import requests
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data")

MOVIES_PATH = os.path.join(DATA_PATH, "movies.csv")
OUTPUT_PATH = os.path.join(DATA_PATH, "tmdb_posters.csv")

BASE_IMAGE_URL = "https://image.tmdb.org/t/p/w500"


def extract_title_and_year(title):
    match = re.search(r"^(.*)\((\d{4})\)$", title)

    if match:
        movie_title = match.group(1).strip()
        year = match.group(2)
    else:
        movie_title = title.strip()
        year = ""

    return movie_title, year


def search_movie(title, year):
    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "year": year,
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return None

        data = response.json()

        results = data.get("results", [])

        if not results:
            return None

        return results[0]

    except Exception:
        return None


def main():
    movies_df = pd.read_csv(MOVIES_PATH)

    poster_rows = []

    total_movies = len(movies_df)

    for index, row in movies_df.iterrows():
        movie_id = row["movieId"]
        full_title = row["title"]

        title, year = extract_title_and_year(full_title)

        print(f"[{index + 1}/{total_movies}] Searching: {title}")

        result = search_movie(title, year)

        if result:
            tmdb_id = result.get("id")
            poster_path = result.get("poster_path")

            if poster_path:
                poster_url = BASE_IMAGE_URL + poster_path
            else:
                poster_url = ""

            poster_rows.append({
                "movieId": movie_id,
                "title": full_title,
                "tmdb_id": tmdb_id,
                "poster_url": poster_url,
            })

        time.sleep(0.2)

    posters_df = pd.DataFrame(poster_rows)

    posters_df.to_csv(OUTPUT_PATH, index=False)

    print("\nPoster fetch complete.")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()