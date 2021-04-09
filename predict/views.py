import os

from django.shortcuts import render, get_object_or_404

from model.cnn import cnn_model
from config.settings import STATIC_URL


def index(request):
    """ render predict main page """
    context = {'content': 'Upload A Test Image And Predict'}
    return render(request, 'predict/predict.html', context)


def predict(request):
    import numpy as np
    import matplotlib.pyplot as plt

    from skimage.segmentation import mark_boundaries
    from keras.preprocessing import image
    from keras.models import load_model


    model = cnn_model()
    model.summary()

    if request.method == 'POST' and request.FILES['test']:
        if not os.path.exists(os.path.join(STATIC_URL, 'img/test/')):
            os.mkdir(os.path.join(STATIC_URL, 'img/test/'))

        test = request.FILES['test']
        with open(os.path.join(STATIC_URL, 'img/test/', 'test.jpg'), 'wb+') as destination:
            for chunk in test.chunks():
                destination.write(chunk)
    
        img = image.load_img(os.path.join(STATIC_URL, 'img/test/', 'test.jpg'),
                             target_size=(128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        o_img = img / 255

        model.load_weights('./model/cnn_model.h5')
        guess = np.argmax(model.predict(o_img), axis=-1)
        out = 'dog' if guess == 1 else 'cat'
        context = {
            'content': out,
            'prob_cat': model.predict(o_img)[0][0],
            'prob_dog': model.predict(o_img)[0][1],
        }
        return render(request, 'predict/predict.html', context)

    return render(request, 'predict/predict.html', {'content': 'wrong access'})
