from django.db.models import Model, CharField


class Genre(Model):
    """
    Model for the genre database table. Allows genres to be split up and 
    associated to movies through a foreign key relation.
    """
    name = CharField(max_length=100, unique=True)

    def __str__(self):
        return f"Genre(name={self.name})"
    