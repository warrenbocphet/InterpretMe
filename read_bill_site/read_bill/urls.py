from django.urls import path

from . import views

app_name = 'read_bill'

urlpatterns = [
    path("", views.index)
]
