from .content_based import recommend_movies as recommend_movies_content_based
from .collaborative_based import recommend_movies as recommend_movies_collaborative_based
from .clustering import recommend_movies as recommend_movies_clustering
from .svd_recommender import recommend_movies as recommend_movies_svd


registry = {
    "content_based": recommend_movies_content_based,
    "collaborative_based": recommend_movies_collaborative_based,
    "clustering": recommend_movies_clustering,
    "svd": recommend_movies_svd,
}