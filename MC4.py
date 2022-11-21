import pandas as pd
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv("E:/Repo/MachineLearning/USA_Housing.csv")
dt_Train, dt_Test = train_test_split(data, test_size = 0.3, shuffle = False)

X_train = dt_Train.iloc[:, :5]
Y_train = dt_Train.iloc[:, 5]
X_test = dt_Test.iloc[:, :5]
Y_test = dt_Test.iloc[:, 5]

reg = LinearRegression().fit(X_train, Y_train)

Y_pred = reg.predict(X_test)
Y = np.array(Y_test)

print("Coefficient of determination: %.2f" % r2_score(Y_test, Y_pred))

print("Thuc te \t Du doan \t Chenh lech")
for i in range(0, len(Y)):
    print("%.2f" % Y[i], "\t", "%.2f" % Y_pred[i], "\t", "%.2f" % abs(Y[i] - Y_pred[i]))