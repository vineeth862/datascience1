
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data_score = pd.read_csv("C:/Users/Loki/Desktop/ML/Datasets/student_scores - student_scores.csv")

y=data_score['Scores'].values

x=data_score['Hours'].values

x = x.reshape(-1,1)
y = y.reshape(-1,1)


train_x,test_x,train_y,test_y= train_test_split(x,y,test_size=0.2,random_state=2)

# To get base Root mean square error
base_pred = np.mean(test_y)
base_pred = np.repeat(base_pred, len(test_y))
brmse = np.sqrt(mean_squared_error(test_y,base_pred))

#Model fitting
lgr = LinearRegression(fit_intercept=True)

model1 = lgr.fit(train_x,train_y)

model1_pred = lgr.predict(test_x)

#Final RMSE
mse1 = mean_squared_error(test_y, model1_pred)
rmse1 = np.sqrt(mse1)

#Plot train and test data over predicted model
plt.scatter(train_x, train_y, color = "red")
plt.plot(train_x, lgr.predict(train_x), color = "green")
plt.title("Hours vs Score (Training set)")
plt.xlabel("Hours of study")
plt.ylabel("Obtained perc")
plt.show()

plt.scatter(test_x, test_y, color = "red")
plt.plot(train_x, lgr.predict(train_x), color = "green")
plt.title("Hours vs Score (Training set)")
plt.xlabel("Hours of study")
plt.ylabel("Obtained perc")
plt.show()


#Predict the score if a student studies for 9.25 hrs/day
x1 = [[9.25]]
n1= lgr.predict(x1)
print("Predicted score if a student studies for 9.25 hrs/ day is %s "%n1[0][0])

#R2 score 
#r2 = model1.score(test_x,test_y)
#print("R2 value %s"%(r2))







