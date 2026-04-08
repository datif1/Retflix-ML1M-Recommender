from django.db.models import Model, IntegerField, CharField, ManyToManyField, URLField
from core.models.genre import Genre


class Movie(Model):
    """
    Model for the movie database table. Allows us to store all of 
    the movie data in one spot after processing title, year, genres, and
    fetching the poster urls.
    """
    movieId = IntegerField(unique=True)
    title = CharField(max_length=500)
    year = CharField(max_length=10, null=True, blank=True)
    genres = ManyToManyField(Genre, related_name="movies")
    poster_url = URLField(null=True, blank=True)

    def __str__(self):
        return f"Movie(Id={self.movieId} title={self.title} year={self.year})"
    