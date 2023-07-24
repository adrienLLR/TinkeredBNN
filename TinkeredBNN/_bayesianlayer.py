import numpy as np
from TinkeredBNN._scalemixture import ScaleMixturePrior
from TinkeredBNN._gaussian import Gaussian

class BayesianLayer:
    def __init__(self, in_features, out_features, f, df):
        """
        Parameters
        ----------
            mean : int, numpy array
                The mean of the distribution of which weights are sampled

            rho : int, numpy array
                Parameter used to compute standard deviation ( log(1 + exp(rho))  = scale )

            in_features : int
                Number of inputs

            out_features : int
                Number of neurons

            f : adress
                Activation function

            df : adress
                Derivative of the activation function

        """
        self.training = False
        self.debug = False

        self.f = f
        self.df = df
        self.in_features= in_features
        self.out_features = out_features

        self.w_mean = np.random.uniform(-0.2, 0.2, size=(out_features, in_features))
        self.w_scale = np.random.uniform(0.01, 0.02, size=(out_features, in_features))
        self.weight = Gaussian(self.w_mean, self.w_scale)

        self.b_mean = np.random.uniform(-0.2, 0.2, size=(out_features, 1))
        self.b_scale = np.random.uniform(0.01, 0.02, size=(out_features, 1))
        self.bias =  Gaussian(self.b_mean, self.b_scale)

        pi = 0.5
        scale1 = 1
        scale2 = 0.002
        self.weight_prior = ScaleMixturePrior(pi, scale1, scale2)
        self.bias_prior = ScaleMixturePrior(pi, scale1, scale2)

    def __str__(self):
        return 'w_mean={}, w_scale={}, b_mean={}, b_scale={}'.format(self.w_mean, self.w_scale, self.b_mean, self.b_scale)
    
    def debug_mode(self):
        """
        Switch debug mode
        """
        self.debug = not self.debug

    def generate_param(self):
        """
        Generate set of weights, biases for the layer.
        """
        self.W = self.weight.sample(self.out_features, self.in_features)
        self.b = self.bias.sample(self.out_features, 1)

    def log_variational_posterior(self):
        """
        Compute the log of the variational posterior q(w/θ)
        """
        return self.weight.log_prob(self.W) + self.bias.log_prob(self.b)

    def log_prior(self):
        """
        Compute the log of the prior distribution p(w)
        """
        return self.weight_prior.log_prob(self.W) + self.bias_prior.log_prob(self.b)

    def forward(self, x):
        """
        Compute the forward pass.

        Parameters
        ----------
        x : scalar, numpy array
            The input you want to make the forward pass to
            Rows are features, columns are individuals

        Returns
        -------
        z, a - The result of the Wx + b operation, the result of the activated Wx + b
        """
        z = np.dot(self.W, x) + self.b
        a = self.f(z)
        return z, a
    
    def update(self, dW_mean, db_mean, dW_scale, db_scale, learning_rate):
        """
        Updates the mean and scale of the weights and biases of the distribution of which they are sampled
        """
        if self.debug:
            print("---------Before parameter's update ---------")
            print(self)

        self.w_mean = self.w_mean - learning_rate * dW_mean
        self.b_mean = self.b_mean - learning_rate * db_mean
        self.w_scale = self.w_scale - learning_rate * dW_scale
        self.b_scale = self.b_scale - learning_rate * db_scale

        if self.debug:
            print("---------After parameter's update ---------")
            print(self)

        self.weight = Gaussian(self.w_mean, self.w_scale)
        self.bias = Gaussian(self.b_mean, self.b_scale)
