#import packages
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#load dataset
dataset=pd.read_csv("train.csv")

#dropping unimportant variables
dataset=dataset.drop(["Name","PassengerId","Ticket","Cabin","Embarked"],axis=1)

#ckeck for null values
print(dataset.isna().sum())

#convert text into numerical
le=preprocessing.LabelEncoder()
le.fit(dataset["Sex"])
dataset["Sex"]=le.transform(dataset["Sex"])

#assigning DV to y and IDV to x
y=dataset["Sex"]
X=dataset[["Survived","Pclass","Age","SibSp","Parch","Fare"]]

print(y.count())

#training the model
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

#applying naive bayes algorithm
from sklearn.naive_bayes import BernoulliNB
clf=BernoulliNB()

#prediction
y_pred=clf.fit(X_train,y_train).predict(X_test)

#accuracy score
print("The accuracy score is : ",accuracy_score(y_test, y_pred, normalize=True))

#confusion matrix
print("The confusion matrix is: \n",confusion_matrix(y_test, y_pred))
