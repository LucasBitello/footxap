"""footxap URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from web.views.home import index
#from api.views.tests import urlTesteMetodos
from api.views.api import obterCountries, obterLeagues, obterSeasons, atualizarSeasonsTeams, searchTeams, \
    obterStatistics, obterStatistcsTeamsPlay

nivelAPIatual = "api/v1"

urlpatterns = [
    #Somente para testes
    #path('test/', urlTesteMetodos),


    path('admin/', admin.site.urls),
    path('home/', index),

    #Urls para obter a base dos itens
    path(nivelAPIatual + '/countries', obterCountries),
    path(nivelAPIatual + '/leagues', obterLeagues),
    path(nivelAPIatual + '/seasons', obterSeasons),
    path(nivelAPIatual + '/seasons/teams', atualizarSeasonsTeams),
    path(nivelAPIatual + '/teams/search', searchTeams),
    #path(nivelAPIatual + '/statistics', obterStatistics),
    path(nivelAPIatual + '/statistics', obterStatistcsTeamsPlay)

]
