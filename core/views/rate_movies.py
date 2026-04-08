from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from core.models import Movie, UserMovieRating

import random


class RateMoviesView(LoginRequiredMixin, TemplateView):
    template_name = "rate_movies.html"
    login_url = "/login/"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Get the ids of the movies that the user has already rated
        rated_ids = UserMovieRating.objects.filter(user=self.request.user).values_list("movie_id", flat=True)
        # Get the movies the user has not rated yet
        unrated_movies = Movie.objects.exclude(id__in=rated_ids)
        # Gets a random sample of movies. Will get either 12 movies or less if the unrated movies is less than 12
        sample_movies = random.sample(list(unrated_movies), min(12, unrated_movies.count()))
        # Sets the movies in the context
        context["movies"] = sample_movies
        context["title"] = "Rate Movies"
        # Sets the options for ratings
        context["rating_choices"] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        return context

    def post(self, request, *args, **kwargs):
        # Go through each of the form elements
        for key, value in request.POST.items():
            # Check for ratings fields and make sure a value was selected
            if key.startswith("rating_") and value:
                # Get the movie id from the form input name
                movie_id = key.split("_")[1]
                try:
                    movie = Movie.objects.get(id=movie_id)
                    # Creates the movie rating if doesn't exist or updates it if it does
                    UserMovieRating.objects.update_or_create(
                        user=request.user,
                        movie=movie,
                        defaults={"rating": float(value)}
                    )
                except Movie.DoesNotExist:
                    continue
        # Return back to the rate movies page to rate more movies
        return redirect("rate_movies")
