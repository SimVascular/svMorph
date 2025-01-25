# import numpy as np
from scipy.optimize import root_scalar

def polynomial_inner(r, a, b, eps, s):
    re = (r**2 + eps**2)**(1/2)
    tolerance = 0.005
    sign_s = 1 if s > 0 else -1
    return sign_s * tolerance + ((b*(109*eps**2+34*r**2-4*re**2)/re**7) + ((3*b*(4*re**2-49*eps**2-14*r**2)*r**2 - 52.5*a*eps**4)/re**9)) * s

def get_radius_of_influence(a, b, eps, s):
    # print("polynomial(0) = ", polynomial(0))
    # print("polynomial(2) = ", polynomial(2))
    def polynomial(r):
        # a = 0.0795774715459
        # b = 0.0331572798108
        # eps = 1
        # s = -0.15
        return polynomial_inner(r, a, b, eps, s)
    # Use root_scalar to find a root, with an initial guess or range
    if polynomial(1e-6) * polynomial(5) > 0:
        print("f(a) and f(b) have the same sign. No root found.")
        return 0

    result = root_scalar(polynomial, bracket=[1e-6, 5], method='brentq')

    # Display the root
    if result.converged:
        print(f"Root found: r = {result.root}")
        return result.root
    else:
        print("Root finding did not converge.")
        return 0

