#2. Implement Naïve Bayes method using scikit-learn library
# Importing modules and libraries(pandas,sklearn).
import pandas as pd
from sklearn import model_selection as ms
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics as mt
# Reading from the given .csv file
glass = pd.read_csv('glass.csv')
x = glass.drop('Type', axis=1)
y = glass['Type'].values
# Splitting the dataset into training and testing.
X_train, X_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.4, random_state=0)
# Using the Naive Bayes algorithm to the training set.
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy: ", mt.accuracy_score(y_test, y_pred))  # Outputting the accuracy to the user.
print("classification_report\n")
print(mt.classification_report(y_test, y_pred))