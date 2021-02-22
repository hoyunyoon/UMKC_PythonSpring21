#1.Delete all the outlier data for the GarageArea field (for the same data set in the use case: House Prices).
import pandas as pd
import matplotlib.pyplot as plt

houseprice = pd.read_csv('train.csv')# Reading data

plt.style.use(style='ggplot')# Setting Style for graph
plt.rcParams['figure.figsize'] = (10, 4)
print(houseprice[['GarageArea']])

# Plotting before removing outliers
plt.scatter(houseprice.GarageArea, houseprice.SalePrice)
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Plot before removing outliers')
plt.show()

# Plotting after removing outliers
filtered_entries = houseprice[(houseprice.GarageArea < 1000) & (houseprice.GarageArea > 200)]
plt.scatter(filtered_entries.GarageArea, filtered_entries.SalePrice)
plt.xlabel('Garage Area')
plt.ylabel('Sale Price')
plt.title('Plot after removing outliers')
plt.show()