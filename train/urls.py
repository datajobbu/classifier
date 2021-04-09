from django.urls import path

from . import views


app_name = "train"

urlpatterns = [
    path('', views.index, name="index"),
    path('upload/', views.upload, name="upload"),
    path('train/', views.train, name="train"),
]