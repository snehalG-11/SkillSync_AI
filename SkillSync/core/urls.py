from django.urls import path
from . import views
from .views import chatbot_view

urlpatterns = [
    path("", views.home_redirect, name="home"),
    path("analyzer/", views.analyzer_view, name="analyzer"),

    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("chatbot/", views.chatbot_view, name="chatbot"),

]
