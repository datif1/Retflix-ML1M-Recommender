from .clustering import recommend_movies as recommend_movies_clustering
from .collaborative_based import recommend_movies as recommend_movies_collaborative_based
from .content_based import recommend_movies as recommend_movies_content_based
from .sentiment_analysis import recommend_movies as recommend_movies_sentiment_analysis

# Holds the mapping to the strategy used for recommendation and its function for generating recommendations
registry = {
    "collaborative_based": recommend_movies_collaborative_based,
    "clustering": recommend_movies_clustering,
    "content_based": recommend_movies_content_based,
    "sentiment_analysis": recommend_movies_sentiment_analysis 
}
