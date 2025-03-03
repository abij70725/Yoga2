from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Yoga_Images(models.Model):
    img = models.ImageField()