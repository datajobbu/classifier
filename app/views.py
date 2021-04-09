from django.shortcuts import render, get_object_or_404


def index(request):
    """ render main page """
    context = {'what': 'Django File Upload'}
    return render(request, 'app/home.html', context)
