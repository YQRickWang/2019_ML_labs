# -*- coding: utf-8 -*-
"""Function used to compute the loss."""
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    #Method: MSE
    
    N = len(y)
    e = y-np.dot(tx,w)
    losses = (1/(2*N))*np.dot(e.transpose(),e)
    
    #Method: MAE
    #losses = 1/N*np.sum(np.absolute(e))
    
    
    
    
    return losses