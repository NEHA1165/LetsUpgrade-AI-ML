#import packages for Multiple Linear Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load data
data=pd.read_excel("Linear Regression.xlsx",sheet_name=0)

#check for string value
print(data.dtypes)

#check for null values in dataset
print(data.isnull().sum())

print(data.describe())

data.hist()

#data normalization(not required as no bid diff in data values)
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#data=sc.fit_transform(data)

#before splitting assign input variables(IDV) to X and output variable(DV) to Y
X=data.iloc[:,1:5]
Y=data.iloc[:,:1]

#split records for training and testing in ration 75:25
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=2)

#training the model by calling linear regression algorithm from sklearn
from sklearn.linear_model import LinearRegression
mul_reg=LinearRegression()
mul_reg.fit(x_train,y_train)

#testing the model
ypred=mul_reg.predict(x_test)

#forecasting by trained data
unseen_pred=mul_reg.predict(np.array([[1180,3,1,1]]))
print(unseen_pred)

#evaluation
from sklearn.metrics import r2_score,mean_squared_error
RSME=np.sqrt(mean_squared_error(y_test, ypred))
r_square=r2_score(y_test, ypred)
print("R square is the deciding factor - how much IDV affects DV:",r_square,"RSME value: ",RSME)

#inferences
print(data.corr())
sns.pairplot(data)