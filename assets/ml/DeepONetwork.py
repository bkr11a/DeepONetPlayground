__author__ = "Brad Rice"
__version__ = 0.1

import numpy as np
import tensorflow as tf

class BiasLayer(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.bias = self.add_weight(initializer = tf.keras.initializers.Zeros, trainable = True)

    @tf.function
    def call(self, X):
        return X + self.bias

class TrunkNN(tf.keras.Model):
    def __init__(self, hiddenLayers, input_shape = (1, ), **kwargs):
        super().__init__(**kwargs)
        self.hiddenLayers = hiddenLayers
        self.inp = tf.keras.layers.InputLayer(input_shape = input_shape)
        self.out = tf.keras.layers.Dense(20, activation = 'linear', name = "trunkNET_Output")

    @tf.function
    def call(self, X):
        X = self.inp(X)
        for layer in self.hiddenLayers:
            X = layer(X)
        
        return self.out(X)

class BranchNN(tf.keras.Model):
    def __init__(self, hiddenLayers, input_shape = (1, ), **kwargs):
        super().__init__(**kwargs)
        self.hiddenLayers = hiddenLayers
        self.inp = tf.keras.layers.InputLayer(input_shape = input_shape)
        self.out = tf.keras.layers.Dense(20, activation = 'linear', name = "branchNet_Output")

    @tf.function
    def call(self, X):
        X = self.inp(X)
        for layer in self.hiddenLayers:
            X = layer(X)
        
        return self.out(X)

class DeepOPINN(tf.keras.Model):
    def __init__(self, branchNN, trunkNN, **kwargs):
        super().__init__(**kwargs)
        self.branchNN = branchNN
        self.trunkNN = trunkNN
        self.biasLayer = BiasLayer()

    @tf.function
    def call(self, X):
        branch_input = X["branch_input"]
        trunk_input = X["trunk_input"]
        branch = self.branchNN(branch_input)
        trunk = self.trunkNN(trunk_input)
        dot = tf.reduce_sum(tf.multiply(branch, trunk, axis = 1, keepdims = True))
        out = self.biasLayer(dot)
        return out

    # The below makes this a physics informed PINN
    @tf.function
    def f_theta_residual(self, f_t, g):
        return f_t - g

    @tf.function
    def train_step(self):
        pass

class DeepONET(tf.keras.Model):
    def __init__(self, branchNN, trunkNN, **kwargs):
        super().__init__(**kwargs)
        self.branchNN = branchNN
        self.trunkNN = trunkNN
        self.biasLayer = BiasLayer()

    @tf.function
    def call(self, X):
        branch_input = X[0]
        print(f"Branch Input Shape: {branch_input.shape}")
        trunk_input = X[1]
        print(f"Trunk Input Shape: {trunk_input.shape}")
        branch = self.branchNN(branch_input)
        print(f"Branch Output Shape: {branch.shape}")
        trunk = self.trunkNN(trunk_input)
        print(f"Trunk Output Shape: {trunk.shape}")
        dot = tf.reduce_sum(tf.multiply(branch, trunk), axis = 1, keepdims = True)
        print(f"Dot Product Output Shape: {dot.shape}")
        out = self.biasLayer(dot)
        print(f"Output Shape: {out.shape}")
        return out