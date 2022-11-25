import numpy as np
from sklearn.linear_model import LinearRegression

X_train = np.array([[60, 2, 10], [40, 2, 5], [100, 3 ,7]])
Y_train = np.array([10, 12, 20])
reg = LinearRegression().fit(X_train, Y_train)

print('w=', reg.coef_)
print('w0=', reg.intercept_)

X_test = np.array([[50, 2, 8]])
f = reg.predict(X_test)

print('f=', f)
