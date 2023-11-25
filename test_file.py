
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
test_dir = data_dir + "/Test"

med_dir = test_dir + "/Medium"
lar_dir = test_dir + "/Large"

img1_dir = med_dir +"/Crack__20180419_06_19_09,915.bmp"

img2_dir = lar_dir + "/Crack__20180419_13_29_14,846.bmp"

img1 = tf.keras.utils.load_img(
    img1_dir, target_size=(100, 100)
)

input_arr1 = tf.keras.utils.img_to_array(img1); 
input_arr1 /= 255.0;
plt.imshow(input_arr1);
plt.axis("off")

print(input_arr1.shape)

input_arr1 = np.expand_dims(input_arr1, axis=0);
print(input_arr1.shape)


img2 = tf.keras.utils.load_img(img2_dir, target_size= (100, 100))

input_arr2 = tf.keras.utils.img_to_array(img2);
input_arr2 = np.expand_dims(input_arr2, axis=0);
input_arr2 /= 255.0;
print(input_arr2.shape)


model = tf.keras.models.load_model('my_model.h5')

predictions1 = model.predict(input_arr1)
score1 = tf.nn.softmax(predictions1)
print(score1)

predictions2 = model.predict(input_arr2)
score2 = tf.nn.softmax(predictions2)
print(score2)

plt.figure(figsize=(10,10))
plt.imshow(img1)
text = 'Large Crack: {:.2f}%'.format(score1[0, 0])
text1 = 'Medium Crack: {:.2f}%'.format(score1[0,1])
text2 = 'No Crack: {:.2f}%'.format(score1[0,2])
text3 = 'Small Crack: {:.2f}%'.format(score1[0,3])
x = 40;
y  = 70;
plt.title("True Crack Classification: Medium\n Prediction: Medium Crack ", fontsize = 20)
plt.text(x,y, text, color = "green" , fontsize = 30)
plt.text(x, 78, text1, color = "green" , fontsize = 30)
plt.text(x, 86, text2, color = "green" , fontsize = 30)
plt.text(x, 94, text3, color = "green" , fontsize = 30)
plt.axis("off")


plt.figure(figsize=(10,10))
plt.imshow(img2)
text = 'Large Crack: {:.2f}%'.format(score2[0, 0])
text1 = 'Medium Crack: {:.2f}%'.format(score2[0,1])
text2 = 'No Crack: {:.2f}%'.format(score2[0,2])
text3 = 'Small Crack: {:.2f}%'.format(score2[0,3])
x = 40;
y  = 70;
plt.title("True Crack Classification: Large\n Prediction: Large Crack ", fontsize = 20)
plt.text(x,y, text, color = "green" , fontsize = 30)
plt.text(x, 78, text1, color = "green" , fontsize = 30)
plt.text(x, 86, text2, color = "green" , fontsize = 30)
plt.text(x, 94, text3, color = "green" , fontsize = 30)
plt.axis("off")





