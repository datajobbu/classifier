from django.urls import path

from app import views


app_name = "app"

urlpatterns = [
    path('', views.index, name="index"),
    path('train_view/', views.train_view, name="train_view"),
    path('train_view/upload/', views.upload, name="upload"),
    path('train_view/train/', views.train, name="train"),
    path('train_view/upload/train/', views.train, name="train"),
    path('predict_view/', views.predict_view, name="predict_view"),
    path('predict_view/predict/', views.predict, name="predict"),
]