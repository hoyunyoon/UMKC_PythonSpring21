#importing packages
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# reading the data
df = pd.read_csv('imdb_master.csv',encoding='latin-1')
print(df.head())
sentences = df['review'].values
y = df['label'].values

#tokenizing data
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(sentences)
#getting the vocabulary of data
sentences = tokenizer.texts_to_matrix(sentences)
#encoding the target column
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)
#splitting into testing and train data
X_train, X_test, y_train, y_test = train_test_split(sentences, y, test_size=0.25, random_state=1000)

# Number of features
# print(input_dim)
#creating and fitting model
#1.	In the code provided, there are three mistake which stop the code to get run successfully;
# find those mistakes and explain why they need to be corrected to be able to get the code run
model = Sequential()
model.add(layers.Dense(300,input_dim=2000, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

[testloss, testaccuracy] = model.evaluate(X_test,y_test)
print("Test Data evaluation: Loss = {}, accuracy = {}".format(testloss, testaccuracy))

print(history.history.keys())

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'validation accuracy'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'validation loss'], loc='upper left')
plt.show()

#2.	Add embedding layer to the model, did you experience any improvement?
from keras.preprocessing.sequence import pad_sequences
sentences=df['review']
max_review_len = max([len(s.split()) for s in sentences])
vocab_size = len(tokenizer.word_index)+1
sentences = tokenizer.texts_to_sequences(sentences)
padded_docs = pad_sequences(sentences,maxlen=max_review_len)

#converting Categorical data to Numerical data using Label Encoding
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)#applying label encoding on label matrix

#splitting into test and train data
X_train, X_test, y_train, y_test = train_test_split(padded_docs, y, test_size=0.25, random_state=1000)

print("vocab_size:",vocab_size)
print("max_review_len:",max_review_len)

from keras.layers import Embedding, Flatten

model = Sequential()#Sequential Neural Network model
model.add(Embedding(vocab_size, 50, input_length=max_review_len))#Embedding layer with 2470 neurons
model.add(Flatten())#Flattening the Network
model.add(layers.Dense(300, activation='relu',input_dim=max_review_len))#hidden layer with 300neurons, input layer with 2470 Neurons
model.add(layers.Dense(3, activation='softmax'))#Output layer with 3neurons, softmax as activation
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(X_train,y_train, epochs=5, verbose=True, validation_data=(X_test,y_test), batch_size=256)

print(history.history.keys())

#Calculating loss & accuracy
[testloss1, testaccuracy1] = model.evaluate(X_test,y_test)
print("Test Data result with Embedded layer: Loss = {}, accuracy = {}".format(testloss1, testaccuracy1))

"""plot"""

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'validation accuracy','loss','validation loss'], loc='upper left')
plt.show()

print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(testloss, testaccuracy))
print("Evaluation result on Test Data with Embedded layer: Loss = {}, accuracy = {}".format(testloss1, testaccuracy1))
print("Actual Value:",y_test[7],"Predicted Value",model.predict_classes(X_test[[7],:]))