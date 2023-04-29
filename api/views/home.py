from json import loads, JSONDecoder, JSONEncoder
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

# Create your views here.

def index(request):
    return render(request, "home.html")