from django.db.models import Model, FloatField, ForeignKey, DateTimeField, CASCADE
from django.contrib.auth.models import User

from core.models.movie import Movie


class UserMovieRating(Model):
    """
    Model for associating movie ratings and users that are
    collected through the UI
    """
    user = ForeignKey(User, on_delete=CASCADE)
    movie = ForeignKey(Movie, on_delete=CASCADE)
    rating = FloatField()
    rated_at = DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "movie")
