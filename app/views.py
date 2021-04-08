import os

from django.http import HttpResponse
from django.shortcuts import render, get_object_or_404
from django.views.decorators.csrf import csrf_protect


def index(request):
    """ render main page """
    context = {'what': 'Django File Upload'}
    return render(request, 'app/index.html', context)


def _handle_uploaded_file(file, filename):
    """ help file uploading """
    if not os.path.exists('data/'):
        os.mkdir('data/')

    with open('data/' + filename, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)


@csrf_protect
def upload(request):
    """ file upload to /data/ """
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        for afile in files:
            _handle_uploaded_file(afile, str(afile))
        
        context = {'what': 'Upload Successed. Ready to train.'}
        return render(request, 'app/index.html', context)

    return HttpResponse("Failed")


def train(request):
    """ 일단 날코딩으로 짜고 후에 코드 분리 및 비동기로 """
    import PIL
    import numpy as np
    import pandas as pd

    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from keras.preprocessing.image import ImageDataGenerator
    from sklearn.model_selection import train_test_split


    FAST_RUN = False
    IMAGE_WIDTH=128
    IMAGE_HEIGHT=128
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
    IMAGE_CHANNELS=3

    filenames = os.listdir("./data/")
    print("file num => ", len(filenames))
    print("-"*50)
    categories = []
    for filename in filenames:   
        category = filename.split('.')[0]
        if category == 'dog':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))

    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten()) 

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy']) 

    model.summary() #log?

    print("### MODEL READY ###")
    earlystop = EarlyStopping(patience=10)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                                patience=2, 
                                                verbose=1, 
                                                factor=0.5, 
                                                min_lr=0.00001)

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
        "./data/",
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        "./data/", 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=batch_size
    )
    print("### MODEL TRAIN ###")
    epochs=3 if FAST_RUN else 5
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//batch_size,
        steps_per_epoch=total_train//batch_size,
        callbacks=callbacks
    )
    print("### MODEL SAVE ###")
    model.save_weights("./model/cnn_model.h5")
    
    context = {"what": "Train Finished! Ready To Predict."}
    return render(request, 'app/index.html', context)


def predict(request):
    """TODO"""
    pass
