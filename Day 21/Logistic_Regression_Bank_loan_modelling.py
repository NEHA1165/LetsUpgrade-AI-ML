#import pandas
import pandas as pd

#Load dataset
dataset=pd.read_excel("Bank_Personal_Loan_Modelling.xlsx",sheet_name=1)

#checking for null values
print(dataset.isna().sum())

#removing unimportant columns
dataset=dataset.drop('ID',axis=1)
dataset=dataset.drop('ZIP Code',axis=1)

#declaring dependent variable
y=dataset['Personal Loan']

#declaring independent variable 
x=dataset[['Age','Experience','Income','Family','CCAvg','Education','Mortgage','Securities Account','CD Account','Online','CreditCard']]

#import statsmodels package to apply Logistic Regression
import statsmodels.api as sm

#adding constant to the independent variables
x1=sm.add_constant(x)

#perform Logistic regression
Logistic=sm.Logit(y,x1)

#developing the model
result=Logistic.fit()

#viewing the result
print(result.summary())
