# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given data.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: AdhithyaRam D
RegisterNumber:  212222230008
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
print(df.iloc[3])
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
print(Y_pred)
print(Y_test)

plt.scatter(X_train,Y_train,color='orange')
plt.plot(X_train,regressor.predict(X_train),color='red')
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,regressor.predict(X_test),color='black')
plt.title("Hourse vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print("Mean Square Error =",mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("Mean Square Error =",mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error =",rmse)
```

## Output:

## 1.df.head()
![image](https://github.com/Adhithyaram29D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393540/21f49b6f-f73a-4e88-b62c-f6cb759f56af)

## 2.df.tail()
![image](https://github.com/Adhithyaram29D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393540/ef24964a-9964-4929-9e7d-5a897e77ebcb)

## 3.X and Y values
![image](https://github.com/Adhithyaram29D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393540/22e09399-26b3-43ae-a284-d7457dcb3346)

## 4.Values of Y Prediction
![image](https://github.com/Adhithyaram29D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393540/28718469-5a74-4b0c-a7e7-3e05511bf120)

## 5.Values of Y Test
![image](https://github.com/Adhithyaram29D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393540/0f4d3c24-c60b-4bf8-8dd3-a198a20610fc)

## 6.Training Set
![image](https://github.com/Adhithyaram29D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393540/ced09711-88b1-4eeb-acb1-5075c5fcf071)

## 7.Test Set
![image](https://github.com/Adhithyaram29D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393540/afd484ee-875c-4059-b41b-7d40e560e959)

## Error Calculation
![image](https://github.com/Adhithyaram29D/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393540/fb3c225e-f55a-4698-bfe1-d0608c09cf94)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
