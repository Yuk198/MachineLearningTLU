import numpy as np
import math
from sklearn.metrics import mean_absolute_error as mae

#NSE 
def nse(targets,predictions):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(predictions))**2))
a =np.array([[1, 2, 3, 4, 5]])
b =np.array([[1, 2.5, 3, 4.9, 5.1]])
print("NSE: " + str(nse(a,b)))

#R2
actual = [1,2,3,4,5]
predict = [1, 2.5, 3, 4.9, 5.1]
 
corr_matrix = np.corrcoef(actual, predict)
corr = corr_matrix[0,1]
R_sq = corr**2

#MAE
actual = [2, 3, 5, 5, 9]
calculated = [3, 3, 8, 7, 6]

error = mae(actual, calculated)

print("Mean absolute error 2 : " + str(error))
print('Coefficient of Determination 2:',R_sq)

#RMSE
y_actual = [1,2,3,4,5]
y_predicted = [1.6,2.5,2.9,3,4.1]
 
MSE = np.square(np.subtract(y_actual,y_predicted)).mean() 
 
RMSE = math.sqrt(MSE)
print("Root Mean Square Error 1:\n")
print(RMSE)
