# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Barathraj B
RegisterNumber:  212222230019
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()

#segregating data to variables
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='yellow')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![image](https://github.com/bharathraj1905/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121490575/81c6c2e6-39de-4a66-a0bc-b5e023b9eb47)

### Array value of X
![image](https://github.com/bharathraj1905/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121490575/41a96bb8-2732-48c3-8924-9a68a70fead3)

### Array value of Y
![image](https://github.com/bharathraj1905/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121490575/3365d2b8-9614-4210-baa4-804515dc74b3)

### Values of Y prediction
![image](https://github.com/bharathraj1905/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121490575/6c04d5c4-ebab-48a1-94dc-c5f38be1af55)

### Values of Y test
![image](https://github.com/bharathraj1905/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121490575/0a10ad94-fb12-4703-ab91-33ecf10fb9c4)

### Training Set Graph
![image](https://github.com/bharathraj1905/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121490575/4d94d187-bfaa-4b3b-bad8-a94068d813b5)

### Test Set Graph
![image](https://github.com/bharathraj1905/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121490575/c722f2ef-f012-418d-b677-fef6b8b41619)

### Values of MSE, MAE and RMSE
![image](https://github.com/bharathraj1905/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/121490575/883b205b-cbc8-407c-b7b5-f585aac7a590)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
