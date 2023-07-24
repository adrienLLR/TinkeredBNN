import numpy as np
import TinkeredBNN._loss as _loss

class Network:
    def __init__(self, *args, loss=None):
        """
        Parameters
        ----------
            args : Layer
                The first parameters are the layers in the left to right order

            loss : Loss, optionnal
                Loss function for the network
        """
        layers = []
        for arg in args:
            layers.append(arg)
        self.layers = layers
        if loss == None:
            self.loss = _loss.EuclideanLoss()

    def forward(self, x):
        """
        Compute forward pass. 

        Parameters
        ----------
            x : scalar

        Returns
        -------
            list of tuple (z, a) with z the result of Wx + b and a the activated value of it
        """
        L = []
        for layer in self.layers:
            z, a = layer.forward(x)
            L.append((z, a))
            x = a
        return L
    
    def predict(self, x):
        """
        Predict the output given an input

        Parameters
        ----------
            X : scalar, numpy array
        
        Returns
        -------
            prediction of the model for input X
        """
        for layer in self.layers:
            _, a = layer.forward(x)
            x = a
        return np.squeeze(x, axis=0)

    def gradient(self, X, Y):
        """
        Compute the gradient of the neural network regarding weights and biases

        Returns
        ----------
            dWlist, dblist
                Gradients with regards to the weights, biases
        """
        _, n = X.shape

        dWlist = [np.zeros((layer.out_features, layer.in_features)) for layer in self.layers]
        dblist = [np.zeros((layer.out_features, 1)) for layer in self.layers]

        for i in range(n):
            x = X[:, i%n].reshape(-1, 1) 
            y = Y[:, i%n].reshape(-1, 1) 
            current_dWlist, current_dblist = self.backward(x, y)
            dWlist = [dW + current_dW for dW, current_dW in zip(dWlist, current_dWlist)]
            dblist = [db + current_db for db, current_db in zip(dblist, current_dblist)]
        
        dWlist = [1/n * dW for dW in dWlist]
        dblist = [1/n * db for db in dblist]

        return dWlist, dblist

    def backward(self, x, y):
        """
        Compute the gradient of the loss function

        Parameters
        ----------
            x : numpy array of shape (n, 1)
            
            y : numpy array of shape (n, 1)

        Returns
        -------
            gradient of loss function
        """
        forward = self.forward(x)
        n = len(self.layers)
        dWlist = [None] * n
        dblist = [None] * n

        delta, dJdW, dJdb = self.init_dW_db(forward, x, y)

        dWlist[-1] = dJdW
        dblist[-1] = dJdb
        for i in range(n-2, -1, -1):
            delta, dJdW, dJdb = self.dW_db(forward, x, i, delta)
            dWlist[i] = dJdW
            dblist[i] = dJdb
        return dWlist, dblist
    
    def init_dW_db(self, forward, x, y):
        """
        Compute dW, db at the initialisation of the backpropagation

        Parameters
        ----------
            forward : List
                The list of Wx + b evaluation / activation

            x : scalar

            y : scalar
        """
        n = len(self.layers)
        output_layer = self.layers[n-1]
        a_r = self.get_A_r(forward, n-1)
        z_r = self.get_Z_r(forward, n-1)
        df_z_r = output_layer.df(z_r)
        entry_r = self.get_layer_entry(forward, n-1, x)

        delta = self.loss.dJ(a_r, y) * df_z_r

        dJdW = np.dot(delta, entry_r.T)
        dJdb = delta
        return delta, dJdW, dJdb

    def dW_db(self, forward, x, i, delta):
        """
        Compute dW, db during the main iterations of the backpropagation
        
        Parameters
        ----------
            forward :
                List of values retrieved during forward pass

            x : scalar

            i : int
                layer index

            delta : numpy array
                Value passed along the backpropagation to compute dW, db
        """
        W_next = self.get_layer_weight(i+1)
        current_layer = self.layers[i]
        entry = self.get_layer_entry(forward, i, x)
        z_i = self.get_Z_r(forward, i)
        df_z_i = current_layer.df(z_i)
        delta = np.dot(W_next.T, delta)*df_z_i
        dJdW = np.dot(delta, entry.T)
        dJdb = delta
        return delta, dJdW, dJdb

    def get_layer_entry(self, Lforward, i, x):  
        """
        Yield the layer's entry input
        """
        if i == 0:
            return x
        return Lforward[i-1][1]
    
    def get_Z_r(self, Lforward, i):
        """
        Yield the layer's values before activation
        """
        return Lforward[i][0]
    
    def get_A_r(self, Lforward, i):
        """
        Yield the layer's values after activation 
        """
        if i == -1:
            return Lforward[-1][1]
        return Lforward[i][1]

    def get_layer_weight(self, i):
        """
        Yield the layer's weights
        """
        return self.layers[i].W

    def euclidean_Loss(self):
        self.loss = _loss.EuclideanLoss()
    
    def base_loss(self):
        self.loss = _loss.BaseLoss()