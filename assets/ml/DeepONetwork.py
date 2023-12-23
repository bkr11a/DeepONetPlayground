__author__ = "Brad Rice"
__version__ = 0.1

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
        self.modelFits = {"Physics_Loss" : [], "Boundary_Loss" : [], "Traditional_Loss" : [], "Combined_Loss" : []}

    @tf.function
    def call(self, X):
        branch_input = X[0]
        trunk_input = X[1]

        branch = self.branchNN(branch_input)
        trunk = self.trunkNN(trunk_input)
        dot = tf.reduce_sum(tf.multiply(branch, trunk), axis = 1, keepdims = True)
        out = self.biasLayer(dot)

        return out

    # The below makes this a physics informed PINN
    @tf.function
    def f_theta_residual(self, f_t, gt):
        return f_t - gt

    @tf.function
    def get_f_residual(self, X): 
        # Grab the relevant information from the inputs.
        g = X[0]
        gt = X[2]
        t = X[1]

        # TODO - test if the persistent flag can be removed for performance
        with tf.GradientTape(persistent = True) as tape:
            # watch how things evole according to the domain
            tape.watch(t)
            f = self([g, t])

            # Compute the gradient
            f_t = tape.gradient(f, t)

        # delete the tape
        del tape

        ODE_residual = self.f_theta_residual(f_t=f_t, gt = gt)
        y_pred = f

        return ODE_residual, y_pred

    @tf.function
    def boundary_residual(self, X):
        g = X[0]
        
        # Make a prediction for each of the boundary conditions
        # In this case that is at t = 0.
        y_pred_IC = self([g, tf.zeros((g.shape[0], 1))])

        return y_pred_IC

    @tf.function
    def loss_function(self, X, y):
        # There are three components to this loss function;
        # 1. Traditional loss function that is explained by the supervised data. i.e. Mean Squared Error
        # 2. Physics Informed Loss, get the loss from the PDE
        # 3. Boundary value loss, get the loss from the boundary conditions (or equivalently initial value)

        physicsLoss, y_pred = self.get_f_residual(X)
        physicsLoss = tf.reduce_mean(tf.square(physicsLoss))

        # Model compiled loss
        # Perhaps this could be removed if we don't have data for y?
        traditionalLoss = self.compute_loss(y=y, y_pred=y_pred)

        # Boundary loss - IVP so grab the first value
        boundaryLoss = self.boundary_residual(X)
        boundaryLoss = tf.reduce_mean(tf.square(boundaryLoss))

        # Could I make this a weighted combination with some tunable variables?
        loss = traditionalLoss + physicsLoss + boundaryLoss

        # Write these out to the tensorboard callback
        tf.summary.scalar('Physics Based Loss', data=physicsLoss, step=self._train_counter)
        tf.summary.scalar('Boundary Loss', data=boundaryLoss, step=self._train_counter)
        tf.summary.scalar('Conventional Loss', data=traditionalLoss, step=self._train_counter)
        tf.summary.scalar('Combined Loss', data=loss, step=self._train_counter)
        tf.summary.scalar('Optimizer Learning Rate', data=self.optimizer.lr, step=self._train_counter)

        # Write the fitting history to a dictionary
        # self.modelFits["Physics_Loss"].append(physicsLoss)
        # self.modelFits["Boundary_Loss"].append(boundaryLoss)
        # self.modelFits["Traditional_Loss"].append(traditionalLoss)
        # self.modelFits["Combined_Loss"].append(loss)

        return loss, y_pred 

    @tf.function
    def get_gradient(self, X, y):
        # Calculate the gradient for the model itself.
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            loss, y_pred = self.loss_function(X=X, y=y)
            grad = tape.gradient(loss, self.trainable_variables)

        del tape
        return loss, grad, y_pred

    @tf.function
    def train_step(self, data):
        # Unpack the data
        X, y = data
        loss, grad_theta, y_pred = self.get_gradient(X=X, y=y)
        self.optimizer.apply_gradients(zip(grad_theta, self.trainable_variables))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        return {m.name : m.result() for m in self.metrics}

class DeepONET(tf.keras.Model):
    def __init__(self, branchNN, trunkNN, **kwargs):
        super().__init__(**kwargs)
        self.branchNN = branchNN
        self.trunkNN = trunkNN
        self.biasLayer = BiasLayer()

    @tf.function
    def call(self, X):
        branch_input = X[0]
        trunk_input = X[1]

        branch = self.branchNN(branch_input)
        trunk = self.trunkNN(trunk_input)
        dot = tf.reduce_sum(tf.multiply(branch, trunk), axis = 1, keepdims = True)
        out = self.biasLayer(dot)

        return out