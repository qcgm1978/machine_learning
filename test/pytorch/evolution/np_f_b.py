# -*- coding: utf-8 -*-
import numpy as np
from utilities import setSelf


class NP(object):
    def __init__(self, t):

        # N is batch size; D_in is input dimension;
        # H is hidden dimension; D_out is output dimension.
        self.N, self.D_in, self.H, self.D_out = t

    def f_b(self):
        N, D_in, H, D_out = self.N, self.D_in, self.H, self.D_out
        # Create random input and output data
        x = np.random.randn(N, D_in)
        y = np.random.randn(N, D_out)
        # Randomly initialize weights. H is hidden dimension
        self.w1 = np.random.randn(D_in, H)
        self.w2 = np.random.randn(H, D_out)
        w1=self.w1
        w2=self.w2
        learning_rate = 1e-6
        _l = []
        for t in range(500):
            # Forward pass: compute predicted y(y_pred)
            y_pred, h_relu, h = self.predictY(
                x, w1, w2)  # (N,D_out) mul relu and weights2
            # Compute loss
            loss = self.compute_loss(y_pred, y, _l)
            # Backprop to compute gradients of w1 and w2 with respect to loss
            grad_w1, grad_w2 = self.backprop_compute_gradients(
                loss, h_relu, w2, h, x)
            # Update weights
            self.update_weights(learning_rate, grad_w1, grad_w2)
        return setSelf(self, locals(), name_function=True)

    def update_weights(self, learning_rate, grad_w1, grad_w2):
        self.w1 -= learning_rate * grad_w1
        self.w2 -= learning_rate * grad_w2

    def backprop_compute_gradients(self, loss, h_relu, w2, h, x):
        grad_y_pred = 2.0 * loss  # (N,D_out)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        # The method calculates the gradient of a loss function with respects to all the weights in the network.
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)
        setSelf(self, locals(), name_function=True)
        return grad_w1, grad_w2

    def compute_loss(self, y_pred, y, _l):
        loss = y_pred - y
        f = np.square(loss).sum()
        _l.append(round(f, 7))
        return loss

    def predictY(self, x, w1, w2):
        h = x.dot(w1)  # (N,H) mul input and weights
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)  # (N,D_out) mul relu and weights2
        return y_pred, h_relu, h
