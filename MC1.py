#Matrix and derivative
from operator import truediv
import numpy as np
import sympy as sp

#-----Matrix-----
mtA = np.array([[1, 4, -1], [2, 0, 1]])
mtB = np.array([[-1, 0], [1, 3], [-1, 1]])

#A+B.T
print(mtA + mtB.T)
#A-B.T
print(mtA - mtB.T)
#A*2
print(mtA * 2) 
#A*B
try:
    print(mtA @ mtB)
except:
    print("ValueError!")
#A.A^-1
try:
    print(mtA * np.linalg.inv(mtA)) #ValueError: shapes
except:
    print("ValueError!")

#-----Derivative-----
#Using Gradient Descent
def grad(x):
    return 2*x+ 5*np.cos(x)

def GD(eta, x):
    while abs(grad(x)) > eta:
        x = x - grad(x) * 0.1
    return  x

print('Solution x1 = %f'%GD(0.00001, -10))
