from django.urls import path
from . import views

urlpatterns = [
    path('', views.home,name='home'),
    path('graph',views.graph,name='graph'),
    path('rest_api',views.rest_api,name='rest_api'),
]
