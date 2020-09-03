#import packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#load dataset
data=pd.read_csv("train.csv")

#drop unimportant variables
data=data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

#convert string to numerical values
le=preprocessing.LabelEncoder()
data["Sex"]=le.fit_transform(data["Sex"])
data["Embarked"]=le.fit_transform(data["Embarked"])

from sklearn import neighbors

#assign DV-->y and IDV-->X
y=data["Pclass"]
X=data.drop(["Pclass"],axis=1)

#split the dataset into train and test dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

error = []

# Calculating error for K values between 1 and 267
for i in range(1,267):
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train).score(X_test,y_test)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
 
#plot the graph
plt.figure(figsize=(12, 6))
plt.plot(range(1,267),error,color='red',linestyle='dashed',marker='o',markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

#prediction
y_pred=knn.predict(X_test)

#confusion matrix
print(confusion_matrix(y_test,y_pred))

