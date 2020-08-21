#import packages
import pandas as pd
import numpy as np
from sklearn import preprocessing

#load dataset
dataset=pd.read_csv("general_data.csv")

#drop unimportant columns
dataset=dataset.drop('EmployeeCount',axis=1)
dataset=dataset.drop('EmployeeID',axis=1)
dataset=dataset.drop('Over18',axis=1)
dataset=dataset.drop('StandardHours',axis=1)

dataset.info()

#converting string variables into Numerical
le=preprocessing.LabelEncoder()
le.fit(dataset["Gender"])
dataset["Gender"]=le.transform(dataset["Gender"])
le.fit(dataset["Attrition"])
dataset["Attrition"]=le.transform(dataset["Attrition"])
le.fit(dataset["BusinessTravel"])
dataset["BusinessTravel"]=le.transform(dataset["BusinessTravel"])
le.fit(dataset["Department"])
dataset["Department"]=le.transform(dataset["Department"])
le.fit(dataset["EducationField"])
dataset["EducationField"]=le.transform(dataset["EducationField"])
le.fit(dataset["JobRole"])
dataset["JobRole"]=le.transform(dataset["JobRole"])
le.fit(dataset["MaritalStatus"])
dataset["MaritalStatus"]=le.transform(dataset["MaritalStatus"])
le.fit(dataset["MaritalStatus"])
dataset["MaritalStatus"]=le.transform(dataset["MaritalStatus"])

#checking for null values
print(dataset.isna().sum())

#replacing null values with average of variable
print("mean of NumCompaniesWorked: ",dataset["NumCompaniesWorked"].mean())
new_NCW_var=np.where(dataset["NumCompaniesWorked"].isnull(),2,dataset["NumCompaniesWorked"])
dataset["NumCompaniesWorked"]=new_NCW_var

print("mean of TotalWorkingYears: ",dataset["TotalWorkingYears"].mean())
new_TWY_var=np.where(dataset["TotalWorkingYears"].isnull(),11,dataset["TotalWorkingYears"])
dataset["TotalWorkingYears"]=new_TWY_var

#recheck for null values
print(dataset.isna().sum())

print(dataset.columns)
#declaring dependent variable
Y=dataset['Attrition']

#declaring independent variable 
X=dataset[['Age','BusinessTravel','Department','DistanceFromHome','Education','EducationField', 'Gender', 'JobLevel', 'JobRole','MaritalStatus','MonthlyIncome','NumCompaniesWorked','PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears','TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion','YearsWithCurrManager']]

#import statsmodels package to apply Logistic Regression
import statsmodels.api as sm

#adding constant to the independent variables
X1=sm.add_constant(X)

#perform Logistic regression
Logistic=sm.Logit(Y,X1)

#developing the model
result=Logistic.fit()

#viewing the result
print(result.summary())