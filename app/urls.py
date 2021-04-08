from django.urls import path

from app import views


app_name = "app"

urlpatterns = [
    path('', views.index, name="index"),
    path('upload/', views.upload, name="upload"),
    path('train/', views.train, name="train"),
    path('predict/', views.predict, name="predict"),
]