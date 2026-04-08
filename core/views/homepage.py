from typing import Callable
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin

from core.models import UserMovieRating
from core.recommenders.registry import registry

class HomepageView(LoginRequiredMixin, TemplateView):
    template_name = "homepage.html"
    login_url = "/login/"
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["title"] = "Home"
        # Check if user has rated any movies
        context["has_rated"] = UserMovieRating.objects.filter(user=self.request.user).count() > 0
        # Get all of the recommender model functions
        sentiment_analysis: Callable = registry.get("sentiment_analysis")
        collaborative_based: Callable = registry.get("collaborative_based")
        content_based: Callable = registry.get("content_based")
        clustering: Callable = registry.get("clustering")
        # Build the data for each carousel
        recommendation_sections = [
            {
                "title": "Recommended with Collaborative-Based Filtering",
                "movies": collaborative_based(self.request.user, top_k=10)
            },
            {
                "title": "Recommended with Content-Based Filtering",
                "movies": content_based(self.request.user, top_k=10)
            },
            {
                "title": "Recommended with Sentiment Analysis",
                "movies": sentiment_analysis(self.request.user, top_k=10)
            },
            {
                "title": "Recommended with Clustering",
                "movies": clustering(self.request.user, top_k=10)
            },
        ]
        context["recommendation_sections"] = recommendation_sections
        return context
    