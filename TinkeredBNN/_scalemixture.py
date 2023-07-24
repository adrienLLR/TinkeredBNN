import numpy as np

class ScaleMixturePrior:
    def __init__(self, pi, scale1, scale2):
        self.pi = pi
        self.scale1 = scale1
        self.scale2 = scale2
        self.variance1 = scale1**2
        self.variance2 = scale2**2
    
    def sample(self, n, m):
        """
        Sample a set of values following the Scale Mixture Gaussian distribution which is a peak with a slab

        Parameters
        ----------
        n : scalar
            Number of rows
        
        m : scalar
            Number of columns
        """
        return self.pi*self.scale1*np.random.normal(loc=0, scale=1, size=(n, m)) + (1 - self.pi)*self.scale2*np.random.normal(loc=0, scale=1, size=(n, m))
    
    def normal_density(self, x, mean, scale, maxi=0):
        """
        Normal density function

        Parameters
        ----------
        x : scalar, numpy array
        
        mean : scalar, numpy array

        scale : scalar, numpy array

        maxi : scalar, numpy array
            Value helping to avoid overflow 
        """
        return (1 / (np.abs(scale) * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ( (x - mean) / scale ) ** 2  - maxi)
    
    def log_prob(self, X):
        """
        Log probability of the prior log(p(W))

        Parameters
        ----------
        X : numpy array
            The weights or biases parameters of the network
        """
        return np.sum( np.log(self.pi * self.normal_density(X, 0, self.scale1) + (1 - self.pi) * self.normal_density(X, 0, self.scale2)) )

    def partial_derivative_w_log_prob(self, X):
        """
        Partial derivative of the log prior ∂logp(W)/∂W

        Parameters
        ----------

        X : numpy array
            The weights or biases parameters of the network
        """
        maxi = np.maximum(-0.5 * ( X / self.scale1 ) ** 2, -0.5 * ( X / self.scale2 ) ** 2 )
        a = self.pi * self.normal_density(X, 0, self.scale1, maxi=maxi)
        da_dX = -X/self.scale1**2 * a

        b = (1 - self.pi) * self.normal_density(X, 0, self.scale2, maxi=maxi)
        db_dX = -X/self.scale2**2 * b

        return ( da_dX + db_dX ) / ( a + b)

