import tensorflow as tf
#from tensorflow.keras 로 작성하면 오류가 발생하니까 from keras만 쓰기
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet, preprocess_input
import math
import os
TRAIN_DATA_DIR = 'data/train/'
VALIDATION_DATA_DIR = 'data/val/'
TRAIN_SAMPLES = 500
VALIDATION_SAMPLES = 500
NUM_CLASSES = 2
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 64
print("2018250051 차수진")
print("\n\nLoad and augment the data")
'''
Colored images usually have 3 channels viz. red, green and blue, each with intensity value
ranging from 0 to 255. To normalize it (i.e. bring the value between 0 to 1), we can rescale
the image by dividing each pixel by 255. Or, we can use the default preprocess_input function
in Keras which does the preprocessing for us.
'''
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rotation_range=20,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range=0.2)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory(TRAIN_DATA_DIR,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=12345,
                                                    class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(VALIDATION_DATA_DIR,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=False,
                                                        class_mode='categorical')
def model_maker():
    base_model = MobileNet(include_top=False,
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False
    input = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input)
    custom_model = GlobalAveragePooling2D()(custom_model)
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dropout(0.5)(custom_model)
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)
    return Model(inputs=input, outputs=predictions)

model = model_maker()
model.compile(loss='categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=['acc'])
model.fit_generator(train_generator,
                    steps_per_epoch=math.ceil(float(TRAIN_SAMPLES) / BATCH_SIZE),
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=math.ceil(float(VALIDATION_SAMPLES) / BATCH_SIZE))

model.save('dogcat.h5')

print("\n\nModel Prediction")

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
model = load_model('dogcat.h5')

img_path = 'data/val/dog/dog.250.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = expanded_img_array / 255. # Preprocess the image
prediction = model.predict(preprocessed_img)
print(prediction)
print(validation_generator.class_indices)
print("2018250051 차수진")