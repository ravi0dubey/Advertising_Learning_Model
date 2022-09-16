import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.linear_model import LinearRegression

#Loading dataset
df= pd.read_csv("D:\\Study\\Data Science\\Python\\ineuron\\Data_Set\\advertising.csv")
pf=ProfileReport(df)
pf.to_file('report_'
           'advertising_profiling1.html')

linear=LinearRegression()
x= df[["TV"]]
y= df[["Sales"]]
linear.fit(x,y)
c = linear.intercept_
m= linear.coef_

file = "linear_regression.sav"
pickle.dump(linear,open(file,'wb'))
print(linear.predict([[45]]))

#we can read the model created by us

saved_linear_model = pickle.load(open(file,'rb'))

#testing linear model against a dataset
dataset_provided =  [52,31,4,95,66,723]
for i in dataset_provided:
    print(saved_linear_model.predict([[i]]))


#dataset to test the accuracy of the saved_linear_model
dataset_accuracy_test = pd.read_csv("D:\\Study\\Data Science\\Python\\ineuron\\Data_Set\\advertising_new.csv")
x_new= df[["TV"]]
y_new= df[["Sales"]]

# getting the accurracy of the linear model
#This is R2 statistics
print(f"accurracy of the model is {linear.score(x_new,y_new)}")



# we can use below function of gradient descent  instead of scikit class LinearRegression

# def gradient_descent(x,y,n):
#     m = 6
#     c = 3
#     eeta = 0.8
#     totalsize = x.shape[0]
#     for i in range(n):
#         y_pred = m*x + c
#         residual = (1/totalsize)*((y-y_pred)**2).sum()
#         print(residual)
#         delta_m = ((y-y_pred)*x).sum()
#         delta_c = ((y-y_pred)*x).sum()
#         m_new = m  - ((eeta/totalsize)*delta_m)
#         c_new = c  - ((eeta/totalsize)*delta_c)
#         m = m_new
#         c = c_new
#         print(f" new value of m and c are {m} and {c} respectively")
#         if residual < 1:
#              plt.scatter(x,y)
#              plt.plot(x,y_pred)