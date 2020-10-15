# -*- coding: utf-8 -*-
import numpy as np
from utilities import setSelf

class NP(object):
    def __init__(self, t):

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        self.N, self.D_in, self.H, self.D_out = t
    def f_b(self):
        N, D_in, H, D_out=self.N, self.D_in, self.H, self.D_out
        # Create random input and output data
        x = np.random.randn(N, D_in)
        y = np.random.randn(N, D_out)
        # Randomly initialize weights
        w1 = np.random.randn(D_in, H)
        w2 = np.random.randn(H, D_out)
        learning_rate = 1e-6
        _l = []
        for t in range(500):
            # Forward pass: compute predicted y
            h = x.dot(w1)  # (N,H)
            h_relu = np.maximum(h, 0)
            y_pred = h_relu.dot(w2)  # (N,D_out)
            # Compute and print loss
            loss = np.square(y_pred - y).sum()
            _l.append(round(loss,7))
            # Backprop to compute gradients of w1 and w2 with respect to loss
            grad_y_pred = 2.0 * (y_pred - y)  # (N,D_out)
            grad_w2 = h_relu.T.dot(grad_y_pred)
            grad_h_relu = grad_y_pred.dot(w2.T)
            grad_h = grad_h_relu.copy()
            grad_h[h < 0] = 0
            grad_w1 = x.T.dot(grad_h)
            # Update weights
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2
        return setSelf(self,locals(),name_function=True)
