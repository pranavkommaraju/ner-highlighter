from django.urls import path

from . import views

urlpatterns = [
    path('extract_entities/', views.extract_entities, name='extract_entities'),
]