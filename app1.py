import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.linear_model import LinearRegression

#Loading dataset
df= pd.read_csv("D:\\Study\\Data Science\\Python\\ineuron\\Data_Set\\advertising.csv")
pf=ProfileReport(df)
pf.to_file('report_advertising_profiling_app1.html')


model = LinearRegression()
X, y = df[['TV','Radio','Newspaper']], df.Sales

model.fit(X, y)

c = model.intercept_
m = model.coef_

# manual computation with formulas
yhat = model.predict(X)
SS_Residual = sum((y-yhat)**2)
SS_Total = sum((y-np.mean(y))**2)
r_squared = 1 - (float(SS_Residual))/SS_Total
adjusted_r_squared = 1 - (1-r_squared)*(len(y)-1)/(len(y)-X.shape[1]-1)
print(f"R_square {r_squared}, Adjusted R_Square {adjusted_r_squared}")


# compute with sklearn linear_model
print(f"R_square {model.score(X, y)}, Adjusted R-Square {1 - (1-model.score(X, y))*(len(y)-1)/(len(y)-X.shape[1]-1)}")




#Feature Selection with Just TV
import statsmodels.formula.api as smf
linear3= smf.ols(formula ='Sales ~TV',data=df).fit()
print(f"R.Squared {round(linear3.rsquared,3)}, Adjusted_R_Square: {round(linear3.rsquared_adj,3)}")
linear3.summary()