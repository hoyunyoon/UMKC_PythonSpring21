#2.Create Multiple Regression for the “Restaurant Revenue Prediction” dataset.
#Evaluate the model using RMSE and R2 score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

dframe = pd.read_csv('data.csv')#imported data

convert = {"City Group":{"Big Cities": 0, "Other": 1}, "Type" : {"FC" : 0, "IL" : 1, "DT" : 2}}

data = dframe.replace(convert)
y = np.log(data.revenue)

x = data.drop(['revenue', 'Id'], axis=1)


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=.25)

#i used here the linear regression named 'lr'

from sklearn import linear_model

lr = linear_model.LinearRegression()

model = lr.fit(x_train, y_train)

#using RMSE and R2 score model value
print ("R squared error : \n", model.score(x_test, y_test))

predictions = model.predict(x_test)

print('RMSE error : \n', mean_squared_error(y_test, predictions))

plt.figtext(.5, .8, ('RMSE is: {}'.format(mean_squared_error(y_test, predictions))))
plt.figtext(.5, .75, ('R^2 is: {}'.format(model.score(x_test, y_test))))
plt.scatter(predictions, y_test, alpha=.52,color='r')#plot given

plt.title('Multiple Linear Regression Model')

plt.xlabel('Actual price')#Actual price

plt.ylabel('Predicted price')#Predicted Price

plt.show()