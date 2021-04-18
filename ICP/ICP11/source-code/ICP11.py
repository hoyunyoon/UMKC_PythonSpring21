#Mallikarjun Edara
#importing libraries
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras import backend as K
K.image_data_format()
#1. Follow the instruction below and then report how the performance changed.(apply all at once)
#adding random seed
seed = 7
numpy.random.seed(seed)
#Loading the cifar10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train.shape[1:]
#Normalizing the inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

#one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# creating the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(X_train.shape[1:]), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
#Compiling the model
epochs = 4
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
#fitting the model
history= model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
#Final evaluation of the model & observing performance changes
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
#Adding layers as per the given instructions
mo1 = Sequential()
mo1.add(Conv2D(32, (3, 3), input_shape=(X_train.shape[1:]), padding='same', activation='relu'))
mo1.add(Dropout(0.2))
mo1.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
mo1.add(MaxPooling2D(pool_size=(2, 2)))
mo1.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
mo1.add(Dropout(0.2))
mo1.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
mo1.add(MaxPooling2D(pool_size=(2, 2)))
mo1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
mo1.add(Dropout(0.2))
mo1.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
mo1.add(MaxPooling2D(pool_size=(2, 2)))
mo1.add(Flatten())
mo1.add(Dropout(0.2))
mo1.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
mo1.add(Dropout(0.2))
mo1.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
mo1.add(Dropout(0.2))
mo1.add(Dense(num_classes, activation='softmax'))
epochs = 4
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
mo1.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
#Fitting the model
history=mo1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=128)


#Evaluating the model and observing the performance changes
scores = mo1.evaluate(X_test, y_test, verbose=1)
print('Test loss score is :', scores[0])
print('Test accuracy score is :', scores[1])
print("Accuracy is : %.2f%%" % (scores[1]*100))
##2
#2.Predict the first 4 images of the test data using the above model. Then, compare with the actual label for those 4
#images to check whether or not the model has predicted correctly.
import matplotlib.pyplot as plt
for k in range(1,5):
    plt.imshow(X_test[k,:,:])
    plt.show()
    y=model.predict_classes(X_test[[k],:])
    print("actual",y_test[k],"predicted",y[0])
#saving the h5 model
mo1.save("my_model.h5")
import tensorflow as tf
from tensorflow import keras
new_model = tf.keras.models.load_model('my_model.h5')

import matplotlib.pyplot as plt
for i in range(1,5):
  plt.imshow(X_test[i,:,:])
  plt.show()
  y=new_model.predict_classes(X_test[[i],:])
  print("actual",yp[i],"predicted",y[0])

#3. Visualize Loss and Accuracy using the history object
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy','loss','val_loss'], loc='upper right')
plt.show()