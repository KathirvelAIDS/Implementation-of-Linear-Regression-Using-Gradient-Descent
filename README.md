# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the linear regression using gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
Step 1:
Use the standard libraries such as numpy, pandas, matplotlib.pyplot in python for the Gradient Descent.

Step 2:
Upload the dataset conditions and check for any null value in the values provided using the .isnull() function.

Step 3:
Declare the default values such as n, m, c, L for the implementation of linear regression using gradient descent.

Step 4:
Calculate the loss using Mean Square Error formula and declare the variables y_pred, dm, dc to find the value of m.

Step 5:
Predict the value of y and also print the values of m and c.

Step 6:
Plot the accquired graph with respect to hours and scores using the scatter plot function.

Step 7:
End the program.


## Program:
```
NAME:KATHIRVEL.A
REG NO:2122212300447
```



```
*/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(x,y,theta)
def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions -y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))

  return theta,J_history
  theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) *"+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\theta)$")
plt.title("Cost frunction using Gradient Descent")

plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0]for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.ylabel("Profit predictions")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
/*
```


OUTPUT:


data:

![image](https://github.com/KathirvelAIDS/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94911373/571b3612-2357-4807-b9f4-d51c7aec3074)


X values:


![image](https://github.com/KathirvelAIDS/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94911373/ef6eba23-9482-42ba-8646-c0718321c5d7)




Y values:



![image](https://github.com/KathirvelAIDS/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94911373/3f046fad-7ef0-4161-944b-bb8ebd9bffa6)





X scaled


![image](https://github.com/KathirvelAIDS/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94911373/31dffe8f-ff6f-470d-b750-d5c9cffe465a)


y scaled




![image](https://github.com/KathirvelAIDS/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94911373/23e78892-0cec-4149-a007-74b5a826a25e)




Predicted value



![image](https://github.com/KathirvelAIDS/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/94911373/aea00954-fe9f-4e14-8f4a-dc521e9ac57e)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
