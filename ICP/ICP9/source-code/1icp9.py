#1.	Plot the loss and accuracy for both training data and validation data using the history object in the source code
from keras import Sequential
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.utils import to_categorical

(train_images,train_labels),(test_images, test_labels) = mnist.load_data()

#display the first image in the training data
#plt.imshow(train_images[0,:,:],cmap='gray')
#plt.title('Ground Truth : {}'.format(train_labels[0]))
# plt.show()

#process the data
#1. convert each image of shape 28*28 to 784 dimensional which will be fed to the network as a single feature
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0],dimData)
test_data = test_images.reshape(test_images.shape[0],dimData)

#convert data to float and scale values between 0 and 1
train_data = train_data.astype('float')
test_data = test_data.astype('float')
#scale data
train_data /=255.0
test_data /=255.0
#change the labels frominteger to one-hot encoding
train_labels_one_hot = to_categorical(train_labels)
test_labels_one_hot = to_categorical(test_labels)

#creating network
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(dimData,)))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=4, verbose=1,
                   validation_data=(test_data, test_labels_one_hot))

[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))
## 1
# plotting the loss

plt.plot(history.history['loss'])                 # Loss for training data

plt.plot(history.history['val_loss'])             # Loss for validation data
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('Model Loss')

plt.legend(['train', 'test'], loc='upper left')
plt.show()

# plotting the accuracy
plt.plot(history.history['accuracy'])             # Accuracy for training data

plt.plot(history.history['val_accuracy'])         # Accuracy for validation data

plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('Model Accuracy')

plt.legend(['train', 'test'], loc='upper right')
plt.show()