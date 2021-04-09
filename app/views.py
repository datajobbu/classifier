from django.shortcuts import render, get_object_or_404


def index(request):
    """ render main page """
    context = {'what': 'Auto Image Classifier(CNN)'}
    return render(request, 'app/home.html', context)
