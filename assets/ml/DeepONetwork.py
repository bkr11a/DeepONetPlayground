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
    def __init__(self, branchNN, trunkNN, domain, **kwargs):
        super().__init__(**kwargs)
        self.branchNN = branchNN
        self.trunkNN = trunkNN
        self.biasLayer = BiasLayer()
        self.domain = domain

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

    # The below makes this a physics informed PINN
    @tf.function
    def f_theta_residual(self, f_t, g):
        return f_t - g

    @tf.function
    def get_f_residual(self, domain, profile):
        # TODO - test if the persistent flag can be removed for performance
        with tf.GradientTape(persistent = True) as tape:
            # watch how things evole according to the domain
            tape.watch(domain)
            # NOTE: Not sure if this will work. I think it will because it will be using the call method
            # However if this was a neural network object passed in the code should be
            # f = self.nn([profile, domain])
            # Perhaps we need to explicitly call the call function?
            # f = self.call([profile, domain])
            f = self([profile, domain])

            # Compute the gradient
            f_t = tape.gradient(f, domain)

        # delete the tape
        del tape

        g = profile

        return self.f_theta_residual(f_t, g)

    @tf.function
    def boundary_residual(self, boundary, est_boundary):
        return boundary - est_boundary

    @tf.function
    def loss_function(self):
        # Have to combine three things to form a unified loss function
        # 1. Traditional loss function that is explained by the supervised data. i.e. Mean Squared Error
        # 2. Physics Informed Loss, get the loss from the PDE
        # 3. Boundary value loss, get the loss from the boundary conditions (or equivalently initial value)

        # TODO: Log all three types of losses to see what has the biggest impact and for further study.
        # Physics informed loss
        physicsLoss = self.get_f_residual(domain=domain, profile=profile)
        physicsLoss = tf.reduce_mean(tf.square(physicsLoss))

        # Model compiled loss
        traditionalLoss = self.compute_loss(y=y, y_pred=y_pred)

        # Boundary loss - IVP so grab the first value
        # NOTE: Double check to see if this is the correct slicing / way to do this
        boundaryLoss = self.boundary_residual(boundary=y[0], est_boundary=y_pred[0])
        boundaryLoss = tf.reduce_mean(tf.square(boundaryLoss))

        # Return this tuple, should I package this in a neater way?
        return traditionalLoss, physicsLoss, boundaryLoss

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