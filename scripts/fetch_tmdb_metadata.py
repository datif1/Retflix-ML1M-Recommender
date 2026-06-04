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
OUTPUT_PATH = os.path.join(DATA_PATH, "tmdb_metadata.csv")

BASE_IMAGE_URL = "https://image.tmdb.org/t/p/w500"


def extract_title_and_year(full_title):
    match = re.search(r"^(.*)\((\d{4})\)$", str(full_title))

    if match:
        return match.group(1).strip(), match.group(2)

    return str(full_title).strip(), ""


def search_tmdb_movie(title, year):
    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "year": year,
        "language": "en-US",
    }

    response = requests.get(url, params=params, timeout=15)

    if response.status_code != 200:
        return None

    results = response.json().get("results", [])

    if not results:
        return None

    return results[0]


def get_movie_details(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"

    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
        "append_to_response": "credits,keywords",
    }

    response = requests.get(url, params=params, timeout=15)

    if response.status_code != 200:
        return None

    return response.json()


def extract_director(credits):
    crew = credits.get("crew", []) if credits else []

    directors = [
        person.get("name", "")
        for person in crew
        if person.get("job") == "Director"
    ]

    return "|".join(directors[:3])


def extract_cast(credits, top_n=5):
    cast = credits.get("cast", []) if credits else []

    names = [
        person.get("name", "")
        for person in cast[:top_n]
        if person.get("name")
    ]

    return "|".join(names)


def extract_keywords(keywords_data):
    keywords = keywords_data.get("keywords", []) if keywords_data else []

    words = [
        item.get("name", "")
        for item in keywords
        if item.get("name")
    ]

    return "|".join(words)


def main():
    if not TMDB_API_KEY:
        raise ValueError("TMDB_API_KEY not found. Please add it to your .env file.")

    movies_df = pd.read_csv(MOVIES_PATH)

    existing_movie_ids = set()

    if os.path.exists(OUTPUT_PATH):
        existing_df = pd.read_csv(OUTPUT_PATH)
        existing_movie_ids = set(existing_df["movieId"].tolist())
        rows = existing_df.to_dict("records")
        print(f"Resuming from existing file with {len(existing_movie_ids)} movies.")
    else:
        rows = []

    total = len(movies_df)

    for idx, row in movies_df.iterrows():
        movie_id = row["movieId"]

        if movie_id in existing_movie_ids:
            continue

        original_title = row["title"]
        ml_genres = row["genres"]

        title, year = extract_title_and_year(original_title)

        print(f"[{idx + 1}/{total}] Searching: {title} ({year})")

        try:
            search_result = search_tmdb_movie(title, year)

            if not search_result:
                rows.append({
                    "movieId": movie_id,
                    "ml_title": original_title,
                    "search_title": title,
                    "year": year,
                    "tmdb_id": "",
                    "tmdb_title": "",
                    "poster_url": "",
                    "overview": "",
                    "tagline": "",
                    "tmdb_genres": "",
                    "keywords": "",
                    "cast": "",
                    "director": "",
                    "runtime": "",
                    "popularity": "",
                    "vote_average": "",
                    "vote_count": "",
                    "release_date": "",
                })
                continue

            tmdb_id = search_result.get("id")

            details = get_movie_details(tmdb_id)

            if not details:
                continue

            poster_path = details.get("poster_path")
            poster_url = BASE_IMAGE_URL + poster_path if poster_path else ""

            tmdb_genres = "|".join(
                genre.get("name", "")
                for genre in details.get("genres", [])
                if genre.get("name")
            )

            rows.append({
                "movieId": movie_id,
                "ml_title": original_title,
                "search_title": title,
                "year": year,
                "tmdb_id": tmdb_id,
                "tmdb_title": details.get("title", ""),
                "poster_url": poster_url,
                "overview": details.get("overview", ""),
                "tagline": details.get("tagline", ""),
                "tmdb_genres": tmdb_genres,
                "keywords": extract_keywords(details.get("keywords", {})),
                "cast": extract_cast(details.get("credits", {})),
                "director": extract_director(details.get("credits", {})),
                "runtime": details.get("runtime", ""),
                "popularity": details.get("popularity", ""),
                "vote_average": details.get("vote_average", ""),
                "vote_count": details.get("vote_count", ""),
                "release_date": details.get("release_date", ""),
            })

            if len(rows) % 25 == 0:
                pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
                print(f"Saved progress: {len(rows)} rows")

            time.sleep(0.25)

        except Exception as e:
            print(f"Error for {original_title}: {e}")
            time.sleep(1)

    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)

    print("\nTMDB metadata fetch complete.")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()