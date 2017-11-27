import json
import os.path
from classification import classifier
from django.http import HttpResponse
from django.shortcuts import render


CURRENT_FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))


def index(request):
    return render(request, 'base.html', {})


def predict(request):
    result = {
        'result': 'not found',
    }
    if (request.method == 'POST'):
        alogrithm = request.POST.get('alogrithm', None)
        content = request.POST.get('content', None)
        print(content)
        text_extraction = request.POST.get('text_extraction', None)
        tag = classifier.excute(text_extraction,[content])

        result['result'] = tag
    return HttpResponse(json.dumps(result), content_type='application/json')

