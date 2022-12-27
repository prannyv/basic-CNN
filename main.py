import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import numpy as np
import cv2
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist #set of handwritten characters

(x_train,y_train),(x_test,y_test) = mnist.load_data()

print(x_test.shape)
# plt.imshow(x_train[5123])
# plt.show()
# plt.imshow(x_train[0],cmap=plt.cm.binary)
# plt.show()
#0 is black, 255 is white

#normalizes all the values in the matrix such that 0 is black & 1 is white
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1,IMG_SIZE, IMG_SIZE,1)
x_testr = np.array(x_test).reshape(-1,IMG_SIZE, IMG_SIZE,1)

model = Sequential()

#first convolutional layer
model.add(Conv2D(64, (3,3), input_shape = x_trainr.shape[1:]))
model.add(Activation("relu"))#relu activation, if x<0,y=0, if x>0,y=x
model.add(MaxPooling2D(pool_size=(2,2)))
#second covolutional layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
#third convolutional layer
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
#fully connected layer1
model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))
#fully connected layer2 (final layer of 10 digits)
model.add(Dense(32))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
print(len(x_trainr))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(x_trainr,y_train,epochs=2,validation_split=0.3)#training the model

test_loss,test_acc = model.evaluate(x_testr,y_test)

predictions = model.predict([x_testr])

print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()
print(np.argmax(predictions[300]))
plt.imshow(x_test[300])
plt.show()

img = cv2.imread("img3.jpeg")
plt.imshow(img)
plt.show()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
resized = cv2.resize(gray,(28,28),interpolation = cv2.INTER_AREA)
newimg = tf.keras.utils.normalize (resized,axis=1)
newimg = np.array(newimg).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print(np.argmax(model.predict(newimg)))