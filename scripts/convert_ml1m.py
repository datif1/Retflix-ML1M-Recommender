import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = "Data/ml-1m"
OUTPUT_DIR = "Data"

movies = pd.read_csv(
    os.path.join(INPUT_DIR, "movies.dat"),
    sep="::",
    engine="python",
    names=["movieId", "title", "genres"],
    encoding="latin-1"
)

ratings = pd.read_csv(
    os.path.join(INPUT_DIR, "ratings.dat"),
    sep="::",
    engine="python",
    names=["userId", "movieId", "rating", "timestamp"],
    encoding="latin-1"
)

users = pd.read_csv(
    os.path.join(INPUT_DIR, "users.dat"),
    sep="::",
    engine="python",
    names=["userId", "gender", "age", "occupation", "zip_code"],
    encoding="latin-1"
)

movies.to_csv(os.path.join(OUTPUT_DIR, "movies.csv"), index=False)
ratings.to_csv(os.path.join(OUTPUT_DIR, "ratings.csv"), index=False)
users.to_csv(os.path.join(OUTPUT_DIR, "users.csv"), index=False)

train, test = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

train.to_csv(os.path.join(OUTPUT_DIR, "ratings_train.csv"), index=False)
test.to_csv(os.path.join(OUTPUT_DIR, "ratings_test.csv"), index=False)

print("MovieLens 1M converted successfully.")
print("Movies:", movies.shape)
print("Ratings:", ratings.shape)
print("Users:", users.shape)
print("Train ratings:", train.shape)
print("Test ratings:", test.shape)