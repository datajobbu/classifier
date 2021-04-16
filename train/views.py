import os

from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_protect

from config.settings import STATIC_URL
from model.cnn import cnn_model


def index(request):
    """ render train main page """
    context = {'what': 'Upload Images And Train Model'}
    return render(request, 'train/train.html', context)


def _handle_uploaded_file(file, filename):
    """ save files """
    if not os.path.exists(os.path.join(STATIC_URL, 'img/data/')):
        os.mkdir(os.path.join(STATIC_URL, 'img/data/'))

    with open(os.path.join(STATIC_URL, 'img/data/', filename), 'wb+') as destination:
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
        return render(request, 'train/train.html', context)

    return HttpResponse("Failed")


def train(request):
    """ 일단 날코딩으로 짜고 후에 코드 분리 및 비동기로 """
    import PIL
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    

    FAST_RUN = False
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS = 3

    filenames = os.listdir(os.path.join(STATIC_URL, 'img/data/'))
    print("file num => ", len(filenames))     #TODO: logging
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
    result = []
    summary = model.summary(print_fn=lambda x: result.append(x))

    print("### MODEL READY ###")
    earlystop = EarlyStopping(patience=10)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                                patience=2, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.001)

    callbacks = [earlystop, learning_rate_reduction]

    df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})
    
    print("### TRAIN TEST SPLIT ###")
    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]
    batch_size= 50

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
                            os.path.join(STATIC_URL, 'img/data/'),
                            x_col='filename',
                            y_col='category', 
                            target_size=IMAGE_SIZE,
                            class_mode='categorical',
                            batch_size=batch_size
                      )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_dataframe(
                            validate_df,
                            os.path.join(STATIC_URL, 'img/data/'),
                            x_col='filename',
                            y_col='category', 
                            target_size=IMAGE_SIZE,
                            class_mode='categorical',
                            batch_size=batch_size
                           )

    print("### MODEL TRAIN ###")
    epochs = 3 if FAST_RUN else 100
    history = model.fit(train_generator,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=total_validate//batch_size,
                        steps_per_epoch=total_train//batch_size,
                        callbacks=callbacks)
    
    result.append(history.history['loss'][-1])
    result.append(history.history['accuracy'][-1])
    print("### MODEL SAVE ###")
    model.save_weights("model/cnn_model.h5")
    
    context = {
                "what": "Train Finished! Ready To Predict.",
                "result": result,
              }

    return render(request, 'train/train.html', context)
