# https://keras.io/examples/cifar10_cnn/
# feedforward convolutional neural net of 11 layers
# preprocess images by compressed sensing

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import numpy as np

batch_size = 32
num_classes = 10
epochs = 10

# create sensing matrices
k = 400
sensingMatrices = np.random.normal(size = (k,32,32,3))

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
trainSize = x_train.shape[0]
testSize = x_test.shape[0]

# compressive sampling
x_train_new = np.zeros((trainSize,k))
x_test_new = np.zeros((testSize,k))
for i in range(k):
	sense = sensingMatrices[i].flatten()
	for j in range(trainSize):
		x_train_new[j,i] = np.inner(x_train[j].flatten(),sense)
	
	for k in range(testSize):
		x_test_new[k,i] = np.inner(x_train[k].flatten(),sense)

x_train = np.reshape(x_train_new,(trainSize,20,20,1))
x_test = np.reshape(x_test_new,(testSize,20,20,1))

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)
scores = model.evaluate(x_test, y_test, verbose=1)
print('Classification error: ',1-scores[1])