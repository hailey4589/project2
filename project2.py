
import keras
import tensorflow as tf 
import glob
from keras import layers
import matplotlib.pyplot as plt
from keras import preprocessing



data_dir = "Data"
train_dir = data_dir + "/Train"
val_dir = data_dir + "/Validation"

data_aug = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1./255,
    )


train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(100, 100),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False
)

validate_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='int',
    class_names=None,
    color_mode='rgb',
    batch_size=32,
    image_size=(100, 100),
    shuffle=True,
    seed=None,
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

data_augmentation = keras.Sequential(
    [
        layers.Rescaling(1./255),
        layers.RandomZoom(0.2),
    ]
)
train_dataset = train_dataset.map(lambda x, y : (data_augmentation(x),y))

plt.figure(figsize = (10,10))
for images, labels in train_dataset.take(1):
    plt.imshow(images[1])
    plt.axis("off")







