#1. find the correlation between ‘survived’ (target column) and ‘sex’ column for the Titanic use case in class.
# Do you think we should keep this feature?
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Loading the train.csv file data using pandas
train_dataFrame = pd.read_csv("train.csv")
# Finding the correlation between the Sex Feature and Survived target variable.
corr = train_dataFrame[["Survived", "Sex"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print(corr)  # Printing the Correlation
# Plotting using the FacetGrid of seaborn library between Sex and Survived
g = sns.FacetGrid(train_dataFrame, col="Sex", row="Survived", margin_titles=False)
g.map(plt.hist, "Age", color="black")  # Plotting Against the age.
plt.show()  # Showing the plot

# The survival rate of Female is very higher when compared to Male.
