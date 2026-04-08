from django.urls import path

from .views.register import RegisterView
from .views.rate_movies import RateMoviesView
from .views.homepage import HomepageView


urlpatterns = [
    path('', HomepageView.as_view(), name='homepage'),
    path("rate/", RateMoviesView.as_view(), name="rate_movies"),
    path('register/', RegisterView.as_view(), name='register'),
]