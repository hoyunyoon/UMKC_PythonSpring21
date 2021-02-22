import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

#importing here data of given

dframe = pd.read_csv('data.csv')

convert = {"City Group": {"Big Cities": 0, "Other": 1}, "Type" : {"FC" : 0, "IL" : 1, "DT" : 2}}

data = dframe.replace(convert)

#check types of the data given

data.revenue.describe()

y = np.log(data.revenue)

x = data.drop(['revenue', 'Id'], axis=1)

#import the train test split data given

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=37, test_size=.29)

from sklearn import linear_model

lr = linear_model.LinearRegression()

model = lr.fit(x_train, y_train)

#Evaluate with RMSE and R2 score models

print ("R squared error : \n", model.score(x_test, y_test))

predictions = model.predict(x_test)

from sklearn.metrics import mean_squared_error

print ('RMSE error is : \n', mean_squared_error(y_test, predictions))

#plot of data prediction here

plt.scatter(predictions, y_test, alpha=.57,color='y')

plt.title('Linear Regression Model')

plt.xlabel('actual price')

plt.ylabel('predicted Price')

plt.show()