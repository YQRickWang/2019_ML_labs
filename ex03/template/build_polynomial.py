# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    N = len(x)
    phi = np.zeros((N,degree+1))
    for t in range(N):
        for k in range(degree+1):
            phi[t,k] = np.power(x[t],k)
            
    return phi

