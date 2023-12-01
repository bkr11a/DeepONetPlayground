__author__ = "Brad Rice"
__version__ = 0.1

import numpy as np
import tensorflow as tf

class BiasLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.bias = self.add_weight(shape = input_shape, initializer = tf.keras.initializers.Zeros, trainable = True)

    @tf.function
    def call(self, X):
        return X + self.bias

class TrunkNN(tf.keras.Model):
    def __init__(self, hiddenLayers, **kwargs):
        super().__init__(**kwargs)
        self.hiddenLayers = hiddenLayers
        self.out = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, X):
        for layer in self.hiddenLayers:
            X = layer(X)
        
        return self.out(X)

class BranchNN(tf.keras.Model):
    def __init__(self, hiddenLayers, **kwargs):
        super().__init__(**kwargs)
        self.hiddenLayers = hiddenLayers
        self.out = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, X):
        for layer in self.hiddenLayers:
            X = layer(X)
        
        return self.out(X)

class DeepOPINN(tf.keras.Model):
    def __init__(self, branchNN, trunkNN, **kwargs):
        super().__init__(**kwargs)
        self.branchNN = branchNN
        self.trunkNN = trunkNN

    @tf.function
    def call(self, X):
        branch_input = X["branch_input"]
        trunk_input = X["trunk_input"]
        branch = self.branchNN(branch_input)
        trunk = self.trunkNN(trunk_input)
        dot = tf.reduce_sum(tf.multiply(branch, trunk, axis = 1, keepdims = True))
        out = BiasLayer()(dot)
        return out

    # The below makes this a physics informed PINN
    @tf.function
    def f_theta_residual(self):
        pass

    @tf.function
    def train_step(self):
        pass

class DeepONET(tf.keras.Model):
    def __init__(self, branchNN, trunkNN, **kwargs):
        super().__init__(**kwargs)
        self.branchNN = branchNN
        self.trunkNN = trunkNN

    @tf.function
    def call(self, X):
        branch_input = X["branch_input"]
        trunk_input = X["trunk_input"]
        branch = self.branchNN(branch_input)
        trunk = self.trunkNN(trunk_input)
        dot = tf.reduce_sum(tf.multiply(branch, trunk, axis = 1, keepdims = True))
        out = BiasLayer()(dot)
        return out