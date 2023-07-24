import numpy as np

class Likelihood:
    def __init__(self, mean, sigma):
        self.sigma = sigma
        self.mean = mean
        self.variance = self.sigma**2
    
    def log_likelihood(self, X, Y, functional):
        _, n = X.shape
        log_likelihood_val = 0
        for i in range(n):
            x = X[:, i].reshape(-1, 1) 
            y = Y[:, i].reshape(-1, 1) 
            log_likelihood_val = log_likelihood_val + (-0.5*np.log(np.pi * 2 * self.variance) - 0.5 * ( ( ( functional(x) - y ) / self.variance ) ** 2 ) )
        return log_likelihood_val