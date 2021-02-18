import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("data2.csv")
# array = []
x = data[['TV','radio','newspaper']]
y= data.sales # EQUALS TO data['sales']

X_train,X_test,y_train,y_test = train_test_split(x,y,random_state=50)
linearreg = LinearRegression()
linearreg.fit(X_train,y_train)

y_predict = linearreg.predict(X_test)
print("Root Mean Square Error : ",np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
print("Intercept : ",linearreg.intercept_)
print("Slopes : ",linearreg.coef_)

y_sto = np.square(y_test - np.mean(y_test))
y_se = np.square(y_predict - y_test)
print("The Coefficient of determination (Accuracy) : ",1 - (np.sum(y_se)/np.sum(y_sto)))

x1 = np.linspace(0,250)

print("The Degrees of the lines : ",np.arctan(linearreg.coef_) * (180/np.pi))
y1 = linearreg.coef_[0] * x1 + linearreg.intercept_
y2 = linearreg.coef_[1] * x1 + linearreg.intercept_
y3 = linearreg.coef_[2] * x1 + linearreg.intercept_

plt.plot(x1,y1,linestyle = "-",label = f"{linearreg.coef_[0]} x + {linearreg.intercept_}")
plt.plot(x1,y2,linestyle = "-",label = f"{linearreg.coef_[0]} x + {linearreg.intercept_}")
plt.plot(x1,y3,linestyle = "-",label = f"{linearreg.coef_[0]} x + {linearreg.intercept_}")
plt.plot(X_test,y_predict,'k*')
plt.legend(loc = "upper left")

plt.show()

