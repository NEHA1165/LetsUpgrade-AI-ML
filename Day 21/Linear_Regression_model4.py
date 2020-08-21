#Import numpy as np, pandas as pd,matplotlib.pyplot as plt and seaborn as sns.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Import the data using pandas
data=pd.read_excel("Linear Regression.xlsx",sheet_name=0)

#dropping unimportant columns
data=data.drop('sqft_living',axis=1)
data=data.drop('bedrooms',axis=1)
data=data.drop('bathrooms',axis=1)

print(data.head())
print()
print(data.dtypes)
print()

#EXploratory Data Analysis-EDA
data.hist()
print(data.corr())
sns.scatterplot(data['floors'],data['price'])

#assign features to X and Y
X=data.iloc[:,1:]
Y=data.iloc[:,:1]

#visualize dataset
plt.scatter(X,Y)
plt.title('price v/s floors')
plt.xlabel('floors')
plt.ylabel('price')
plt.show()

#split records for training and testing
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
print(X_train.shape)
print(X_test.shape)

#model building with sklearn
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()

#train the model
lin_reg.fit(X_train,Y_train)
print(lin_reg.coef_)
print(lin_reg.intercept_)

#visualize training set result
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train,lin_reg.predict(X_train),color='green')
plt.title('Linear Regression price v/s floors(training set)')
plt.xlabel('floors')
plt.ylabel('price')
plt.show()

#test the model(for X_test value predict y value)
y_pred=lin_reg.predict(X_test)
print(y_pred)
print(X_test.head())

#compare initial data to predicted data
print(data.head())

#vizualise the test dataset
plt.scatter(X_test,Y_test,color='green')
plt.plot(X_test,lin_reg.predict(X_test),color='blue')
plt.title('Linear Regression price v/s floors(testing set)')
plt.xlabel('floors')
plt.ylabel('price')
plt.show()

#Estimate the cost
from sklearn.metrics import mean_squared_error,r2_score
RSME=np.sqrt(mean_squared_error(Y_test, y_pred))
r_square=r2_score(Y_test, y_pred)
print("R square is the deciding factor - how much IDV affects DV:",r_square,"RSME value: ",RSME)

#predicting unseen value
unseen_predict=lin_reg.predict(np.array([[2.25]]))
print("predicted value is: ",unseen_predict)
