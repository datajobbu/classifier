import os

from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_protect

from model.cnn import cnn_model


def index(request):
    """ render main page """
    context = {'what': 'Django File Upload'}
    return render(request, 'app/index.html', context)


def train_view(request):
    """ render train page """
    context = {'what': 'Upload Images And Train Model'}
    return render(request, 'app/train.html', context)


def predict_view(request):
    """ render predict page """
    context = {'content': 'Upload A Test Image And Predict'}
    return render(request, 'app/predict.html', context)


def _handle_uploaded_file(file, filename):
    """ help file uploading """
    if not os.path.exists('static/img/data/'):
        os.mkdir('static/img/data/')

    with open('static/img/data/' + filename, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)


@csrf_protect
def upload(request):
    """ file upload to static/img/data/ """
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        for afile in files:
            _handle_uploaded_file(afile, str(afile))
        
        context = {'what': 'Upload Successed. Ready to train.'}
        return render(request, 'app/train.html', context)

    return HttpResponse("Failed")


def train(request):
    """ 일단 날코딩으로 짜고 후에 코드 분리 및 비동기로 """
    import PIL
    import numpy as np
    import pandas as pd

    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.preprocessing.image import ImageDataGenerator
    from sklearn.model_selection import train_test_split


    FAST_RUN = False
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 3

    filenames = os.listdir("./static/img/data/")
    print("file num => ", len(filenames))
    print("-"*50)
    categories = []
    for filename in filenames:
        """TODO: Change not only dog/cat""" 
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    model = cnn_model()
    model.summary() #log?

    print("### MODEL READY ###")
    earlystop = EarlyStopping(patience=10)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                                patience=2, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.0001)

    callbacks = [earlystop, learning_rate_reduction]

    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
    
    print("### TRAIN TEST SPLIT ###")
    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]
    batch_size=15

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        "./static/img/data/",
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        "./static/img/data/", 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    print("### MODEL TRAIN ###")
    epochs=3 if FAST_RUN else 100
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//batch_size,
        steps_per_epoch=total_train//batch_size,
        callbacks=callbacks
    )

    print("### MODEL SAVE ###")
    model.save_weights("model/cnn_model.h5")
    
    context = {"what": "Train Finished! Ready To Predict."}
    return render(request, 'app/predict.html', context)


def predict(request):
    import numpy as np
    import matplotlib.pyplot as plt

    import lime
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    from keras.preprocessing import image
    from keras.models import load_model


    model = cnn_model()
    model.summary() #log?

    if request.method == 'POST' and request.FILES['test']:
        if not os.path.exists('static/img/test/'):
            os.mkdir('static/img/test/')

        test = request.FILES['test']
        with open('static/img/test/' + "test.jpg", 'wb+') as destination:
            for chunk in test.chunks():
                destination.write(chunk)
        
        img = image.load_img("./static/img/test/test.jpg", target_size=(128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255

        model.load_weights('./model/cnn_model.h5')

        """TODO LIME Imple
        explainer = lime_image.LimeImageExplainer(random_state=42)
        print(img)
        print(model.predict)
        explanation = explainer.explain_instance(
            img, model.predict
        )

        image, mask = explanation.get_image_and_mask(
            model.predict(
                img
            ).argmax(aixs=1)[0],
            positive_only=True,
            hide_rest=False
        )
        plt.imshow(mark_boundaries(image, mask))
        plt.savefig('./immg.jpg')"""
        

        guess = np.argmax(model.predict(img), axis=-1)
        out = 'dog' if guess == 1 else 'cat'
        context = {
            'content': out,
            'prob_cat': model.predict(img)[0][0],
            'prob_dog': model.predict(img)[0][1],
        }
    
        return render(request, 'app/predict.html', context)

    return render(request, 'app/predict.html', {'content': 'wrong access'})
