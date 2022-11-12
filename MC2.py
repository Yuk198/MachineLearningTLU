import numpy as np

#f(x) = x**2 - 2
def grad(x, n):
    if n==1:
        #f(x) = x**2 - 2
        return 2*x
    elif n==2:
        #g(x) = (1/3)*x**3 - x
        return x**2 - 1

def GD(eta, x0, n):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1], n)
        if abs(grad(x_new, n)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)

(x1, it1) = GD(.1, -5, 1)
(x2, it2) = GD(.1, 0, 1)
print('Solution ex1 x1 = %f, obtained after %d iterations'%(x1[-1],  it1))
print('Solution ex1 x2 = %f, obtained after %d iterations'%(x2[-1],  it2))

(x3, it3) = GD(.0001, -5, 2)
(x4, it4) = GD(.0001, 0.5, 2)
print('Solution ex2 x3 = %f, obtained after %d iterations'%(x3[-1],  it3))
print('Solution ex2 x4 = %f, obtained after %d iterations'%(x4[-1],  it4))