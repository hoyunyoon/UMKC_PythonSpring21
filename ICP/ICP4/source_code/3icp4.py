# 3. Implement linear SVM method using scikit library
# Importing modules and libraries(pandas,sklearn)
import pandas as pd
from sklearn import model_selection as ms
from sklearn import svm
from sklearn import metrics
# Reading the .csv file.
glass = pd.read_csv('glass.csv')
x = glass.drop('Type', axis=1)
y = glass['Type']
# Splitting the dataset into training and testing.
X_train, X_test, y_train, y_test = ms.train_test_split(x, y, test_size=0.4, random_state=0)
# Using the SVM method to the training
svc = svm.SVC(kernel="linear")
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))  # Outputting the accuracy to the user.
print("Classification_report\n")
print(metrics.classification_report(y_test, y_pred, zero_division=1))
