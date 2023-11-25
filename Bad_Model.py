
import keras
import numpy as np
import tensorflow as tf 
from keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from numpy.random import seed
seed(4)


data_dir = "Data"
train_dir = data_dir + "/Train"
val_dir = data_dir + "/Validation"

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(100, 100),
    shuffle=True,
    seed=4,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)


class_names = train_dataset.class_names
print(class_names)



validate_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(100, 100),
    shuffle=True,
    seed=4,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)

plt.figure(figsize = (10,10))
for images, labels in train_dataset.take(1):
    plt.imshow(images[1].numpy().astype("uint8"))
    plt.axis("off")

data_augmentation_train = keras.Sequential(
    [
        layers.Rescaling(1./255),
        layers.RandomZoom((0.1), seed = 4),
    ]
)

data_augmentation_val = keras.Sequential(
    [
        layers.Rescaling(1./255),
    ]
)

train_dataset = train_dataset.map(lambda x, y : (data_augmentation_train(x),y))
validate_dataset = validate_dataset.map(lambda x, y : (data_augmentation_val(x),y))


plt.figure(figsize = (10,10))
for images, labels in train_dataset.take(1):
    plt.imshow(images[1].numpy().astype("uint8"))
    plt.axis("off")



data_aug = ImageDataGenerator(
    shear_range=0.2  # Adjust the shear range as needed
)

# BAD MODEL 

model = Sequential([
  layers.Conv2D(2, 2,strides=(1, 1),padding='same', activation='relu', ),
  layers.MaxPooling2D(),
  layers.Dropout(0.1),
  layers.Flatten(),
  layers.Dense(32, activation= 'relu'),
  layers.Dense(4, activation = 'softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],)


epochs=10
history = model.fit(
  train_dataset,
  validation_data = validate_dataset,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


