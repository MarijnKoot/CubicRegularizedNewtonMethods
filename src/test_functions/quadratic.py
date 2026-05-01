""""
In this folder we give a possible (likely not good) test function the quadratic f(x)=x^T A x - b^T x. 
Since this should converge in a single step (for proper sigma)
"""

import numpy as np



def quadratic(A,b):
    """
    f(x) = 0.5 x^T A x - b^T x
    """        
    # Define f, grad, hess
    def f(x):
        return 0.5 * x.T @ (A @ x) - b.T @ x

    def grad(x):
        return A @ x - b

    def hess(x):
        return A   # constant Hessian
    

    return f, grad, hess, 0.0  # L3 = 0 exactly (constant Hessian)