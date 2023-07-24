import numpy as np

class Gaussian:
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale
    
    def sample(self, n, m):
        self.noise = np.random.normal(loc=0, scale=1, size=(n, m))
        return self.mean + self.scale * self.noise

    def log_prob(self, X):
        X = (X - self.mean)/self.scale
        X = np.dot(X.T, X)
        X = np.log(2*np.pi*(self.scale**2)) + X
        return -0.5*np.sum(X)
    
    def partial_derivative_w_log_prob(self, X):
        """
        Partial derivative of the variational posterior regarding the weights or biases ∂log(q(w/θ))/∂w

        Parameters
        ----------
        X : numpy array
            The weights or biases parameters of the network
        """
        return -(X - self.mean)/(self.scale**2)

    def partial_derivative_mu_log_prob(self, X):
        """
        Partial derivative of the variational posterior regarding the mean ∂log(q(w/θ))/∂μ

        Parameters
        ----------
        X : numpy array
            The weights or biases parameters of the network
        """
        return -(self.mean - X)/(self.scale**2)

    def partial_derivative_sigma_log_prob(self, X):
        """
        Partial derivative of the variational posterior regarding the scale ∂log(q(w/θ))/∂σ

        Parameters
        ----------
        X : numpy array
            The weights or biases parameters of the network
        """
        return -1/self.scale + ((X - self.mean)**2)/(self.scale**3)