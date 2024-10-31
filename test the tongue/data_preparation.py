import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout 
from sklearn.model_selection import train_test_split




data_dir = 'C:/Users/krith/OneDrive/Documents/Desktop/cg project/images'



IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32


train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,  
    horizontal_flip=True,
    zoom_range=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)


images, labels = next(train_gen)
print(f'Batch shape: {images.shape}')
print(f'Labels shape: {labels.shape}')


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

val_loss, val_acc = model.evaluate(val_gen)
print(f'Validation Accuracy: {val_acc}')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']




model_save_path ='C:/Users/krith/OneDrive/Documents/Desktop/cg project/saved_model.keras'
model.save('saved_model.keras')

print(f'Model saved to {model_save_path}')
