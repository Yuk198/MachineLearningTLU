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
#Using Gradient Desent
def grad(x):
    return 2*x+ 5*np.cos(x)

def GD(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = GD(.1, -10)
(x2, it2) = GD(.1, 10)
print('Solution x1 = %f, obtained after %d iterations'%(x1[-1],  it1))
print('Solution x2 = %f, obtained after %d iterations'%(x2[-1],  it2))
