import os

from django.shortcuts import render, get_object_or_404

from model.cnn import cnn_model
from config.settings import STATIC_URL


def index(request):
    """ render predict main page """
    context = {'content': 'Upload A Test Image And Predict'}
    return render(request, 'predict/predict.html', context)


def predict(request):
    """ Predict - Show Image(with lime) and Probabilities """ 
    import numpy as np
    import matplotlib.pyplot as plt

    from skimage.segmentation import mark_boundaries
    from keras.preprocessing import image
    from keras.models import load_model
    from lime.lime_image import LimeImageExplainer
    from lime.wrappers.scikit_image import SegmentationAlgorithm


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

        t_img = o_img[0]               #for lime (4D -> 3D)
        t_img = t_img.astype('double')

        model = cnn_model()
        model.load_weights('./model/cnn_model.h5')
        guess = np.argmax(model.predict(o_img), axis=-1)
        out = 'dog' if guess == 1 else 'cat'

        lime_explainer = LimeImageExplainer()
        segmenter = SegmentationAlgorithm(
                        'slic',
                        n_segments=100,
                        compactness=1,
                        sigma=1
                    )
        explanation = lime_explainer.explain_instance(
                            t_img,
                            model.predict,
                            segmentation_fn=segmenter
                      )
        temp, mask = explanation.get_image_and_mask(
                        model.predict(o_img).argmax(axis=1)[0],
                        positive_only=True,
                        hide_rest=False
                     )
        
        fig = plt.figure()
        plt.imshow(mark_boundaries(temp, mask))
        plt.axis('off')
        plt.savefig(os.path.join(STATIC_URL, 'img/test/', 'lime.jpg'),)
        plt.close(fig)

        context = {'content': out,
                   'prob_cat': model.predict(o_img)[0][0],
                   'prob_dog': model.predict(o_img)[0][1],
                   }

        return render(request, 'predict/predict.html', context)

    return render(request, 'predict/predict.html', {'content': 'wrong access'})
