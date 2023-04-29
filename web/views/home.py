from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import Context, loader
from json import loads
from api.regras.countriesRegras import CountriesRegras
# Create your views here.

def index(request):
    urlHTML = "web/templates/home.html"

    return render(request, urlHTML, {"arrCountries": [], "flag": ""})