import numpy as np
import matplotlib.pyplot as plt
from TinkeredBNN._network import Network
from TinkeredBNN._learning_rate import StepDecay
import TinkeredBNN._loss as _loss

class BayesianNetwork(Network):
    def __init__(self, *args, likelihood, lr, kl_weight):
        """
        Parameters
        ----------
            args : BayesianLayer 
                The first parameters are the layers in the left to right order, minimum 2 layers

            likelihood : Likelihood
                The likelihood of the data

            lr : StepDecay
                Class yielding the length of the step ( learning_rate ) that will be taken to update the parameters, depending on the epoch.
        """
        super().__init__(*args)
        self.likelihood = likelihood
        self.lr = lr
        self.kl_weight = kl_weight
        self.debug = False

    def generate_param(self):
        """
        Generate set of weights, biases for each layer of the model.
        """
        for layer in self.layers:
            layer.generate_param()
    
    def debug_mode(self):
        """
        Switch debug mode : True -> False or False -> True
        """
        self.debug = not self.debug
        for layer in self.layers:
            layer.debug_mode()

    def elbo(self, X, Y):
        """
        Compute the ELBO value given input X, output Y
        """
        elbo_value = 0
        for layer in self.layers:
            elbo_value = layer.log_variational_posterior() - layer.log_prior()
        elbo_value = elbo_value - self.likelihood.log_likelihood(X, Y, self.predict)
        return elbo_value[0, 0]

    def partial_derivative_w_log_likelihood(self, X, Y):
        """
        Compute log(p(D/w)) by using the backward function. 
        The backward function computes the total derivatives of the functional model regarding the weights and biases (dϕ/dW, dϕ/db).
        It is then linearly transformed to compute the log likelihood partial derivative regarding the weights and biases (∂log(p(D/w))/∂w, ∂log(p(D/w))/∂b)

        Parameters
        ----------
            X - numpy array
                Rows are features, columns are individuals
            Y - numpy array

        Returns
        -------
            partial_derivative_w_dW, partial_derivative_w_db - list, list
                Partial derivatives of the log likelihood regarding weights and biases
        """
        _, n = X.shape

        dWlist = [np.zeros((layer.out_features, layer.in_features)) for layer in self.layers]
        dblist = [np.zeros((layer.out_features, 1)) for layer in self.layers]

        for i in range(n):
            x = X[:, i%n].reshape(-1, 1) 
            y = Y[:, i%n].reshape(-1, 1) 
            current_dWlist, current_dblist = self.backward(x, y)
            pred_diff = (self.predict(x) - y)[0, 0]
            dWlist = [dW + pred_diff * current_dW for dW, current_dW in zip(dWlist, current_dWlist)]
            dblist = [db + pred_diff * current_db for db, current_db in zip(dblist, current_dblist)]

        partial_derivative_w_dW = [(-1/self.likelihood.variance)*dW for dW in dWlist]
        partial_derivative_w_db = [(-1/self.likelihood.variance)*db for db in dblist]

        return partial_derivative_w_dW, partial_derivative_w_db
    
    def bayesgradient(self, dlikelihood_w, dlikelihood_b , index, layer):
        """
        Compute the gradient of the loss function defined in "Weight Uncertainty in Neural Networks" paper, of a layer regarding the mean and scale

        Returns
        ----------
            dW_mean, db_mean, dW_scale, db_scale
                Gradients with regard to the mean, scale of the weights and biases
        """
        weights = layer.W
        biases = layer.b

        dfdW = layer.weight.partial_derivative_w_log_prob(weights) - layer.weight_prior.partial_derivative_w_log_prob(weights) - dlikelihood_w[index]
        dfdb = layer.bias.partial_derivative_w_log_prob(biases) - layer.bias_prior.partial_derivative_w_log_prob(biases) - dlikelihood_b[index]

        dW_mean = dfdW + layer.weight.partial_derivative_mu_log_prob(weights)
        db_mean = dfdb + layer.bias.partial_derivative_mu_log_prob(biases)

        dW_scale = dfdW * layer.weight.noise + layer.weight.partial_derivative_sigma_log_prob(weights)
        db_scale = dfdb * layer.bias.noise + layer.bias.partial_derivative_sigma_log_prob(biases)

        if self.debug:
            self.debug_info(index, layer, weights, biases, dlikelihood_w, dlikelihood_b)

        return dW_mean, db_mean, dW_scale, db_scale

    def nngradient(self, gradient, index, layer):
        """
        Compute the gradient of the neural network functional model regarding the mean and scale

        Returns
        ----------
            dW_mean, db_mean, dW_scale, db_scale
                Gradients with regard to the mean, scale of the weights and biases
        """
        gradient_W_list, gradient_b_list = gradient

        dW_mean = gradient_W_list[index]
        db_mean = gradient_b_list[index]

        dW_scale = gradient_W_list[index] * layer.weight.noise
        db_scale = gradient_b_list[index] * layer.bias.noise

        return dW_mean, db_mean, dW_scale, db_scale


    def bayesbybackprop_noadam(self, X, Y):
        """
        Performs the bayesbybackprop algorithm stated by the "Weight Uncertainty in Neural Networks, 21 May 2015" paper and update the parameters accordingly.
        Adam optimizer is not used to optimize parameters
        
        Parameters
        ----------
            X : scalar, numpy array
                Rows are features, columns are individuals

            Y : scalar, numpy array

        """
        weight_partial_derivative_w_log_likelihood, bias_partial_derivative_w_log_likelihood = self.partial_derivative_w_log_likelihood(X, Y)
        self.euclidean_Loss()
        net_gradient = self.gradient(X, Y)
        self.base_loss()

        for index, layer in enumerate(self.layers):

            bayes_gradients = self.bayesgradient(weight_partial_derivative_w_log_likelihood, bias_partial_derivative_w_log_likelihood, index, layer)
            nn_gradient = self.nngradient(net_gradient, index, layer)  
            dW_mean, db_mean, dW_scale, db_scale = self.combine_gradients(nn_gradient, bayes_gradients, self.kl_weight)

            layer.update(dW_mean, db_mean, dW_scale, db_scale, self.learning_rate)

    
    def bayesbybackprop_withadam(self, X, Y):
        """
        Performs the bayesbybackprop algorithm stated by the "Weight Uncertainty in Neural Networks, 21 May 2015" paper and update the parameters accordingly.
        Adam optimizer is used to optimize the parameters.

        Parameters
        ----------
            X : scalar, numpy array
                Rows are features, columns are individuals

            Y : scalar, numpy array

        """
        weight_partial_derivative_w_log_likelihood, bias_partial_derivative_w_log_likelihood = self.partial_derivative_w_log_likelihood(X, Y)
        self.euclidean_Loss()
        net_gradient = self.gradient(X, Y)
        self.base_loss()

        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8

        # Initialize first and second moment vectors
        m_w_mean = [np.zeros_like(layer.w_mean) for layer in self.layers]
        v_w_mean = [np.zeros_like(layer.w_mean) for layer in self.layers]
        m_b_mean = [np.zeros_like(layer.b_mean) for layer in self.layers]
        v_b_mean = [np.zeros_like(layer.b_mean) for layer in self.layers]
        
        m_w_scale = [np.zeros_like(layer.w_scale) for layer in self.layers]
        v_w_scale = [np.zeros_like(layer.w_scale) for layer in self.layers]
        m_b_scale = [np.zeros_like(layer.b_scale) for layer in self.layers]
        v_b_scale = [np.zeros_like(layer.b_scale) for layer in self.layers]

        t = 0

        for index, layer in enumerate(self.layers):
            t += 1

            bayes_gradients = self.bayesgradient(weight_partial_derivative_w_log_likelihood, bias_partial_derivative_w_log_likelihood, index, layer)
            nn_gradient = self.nngradient(net_gradient, index, layer)  
            dW_mean, db_mean, dW_scale, db_scale = self.combine_gradients(nn_gradient, bayes_gradients, self.kl_weight)

            # Update biased first moment estimate for weights and biases
            m_w_mean[index] = beta1 * m_w_mean[index] + (1-beta1) * dW_mean
            m_b_mean[index] = beta1 * m_b_mean[index] + (1-beta1) * db_mean
            m_w_scale[index] = beta1 * m_w_scale[index] + (1-beta1) * dW_scale
            m_b_scale[index] = beta1 * m_b_scale[index] + (1-beta1) * db_scale

            # Update biased second raw moment estimate for weights and biases
            v_w_mean[index] = beta2 * v_w_mean[index] + (1-beta2) * (dW_mean**2)
            v_b_mean[index] = beta2 * v_b_mean[index] + (1-beta2) * (db_mean**2)
            v_w_scale[index] = beta2 * v_w_scale[index] + (1-beta2) * (dW_scale**2)
            v_b_scale[index] = beta2 * v_b_scale[index] + (1-beta2) * (db_scale**2)

            # Compute bias-corrected first moment estimate for weights and biases
            m_w_mean_hat = m_w_mean[index] / (1 - beta1**t)
            m_b_mean_hat = m_b_mean[index] / (1 - beta1**t)
            m_w_scale_hat = m_w_scale[index] / (1 - beta1**t)
            m_b_scale_hat = m_b_scale[index] / (1 - beta1**t)

            # Compute bias-corrected second raw moment estimate for weights and biases
            v_w_mean_hat = v_w_mean[index] / (1 - beta2**t)
            v_b_mean_hat = v_b_mean[index] / (1 - beta2**t)
            v_w_scale_hat = v_w_scale[index] / (1 - beta2**t)
            v_b_scale_hat = v_b_scale[index] / (1 - beta2**t)

            # Update parameters
            layer.w_mean -= self.learning_rate * m_w_mean_hat / (np.sqrt(v_w_mean_hat) + epsilon)
            layer.b_mean -= self.learning_rate * m_b_mean_hat / (np.sqrt(v_b_mean_hat) + epsilon)
            layer.w_scale -= self.learning_rate * m_w_scale_hat / (np.sqrt(v_w_scale_hat) + epsilon)
            layer.b_scale -= self.learning_rate * m_b_scale_hat / (np.sqrt(v_b_scale_hat) + epsilon)
    
    def train(self, X, Y, n_epoch=1, verbose=False, graph=False, adam=True):
        """
        Train the model using the input X and the output Y, on a chosen number of epoch
        
        Parameters
        ----------

        X : numpy array
            Rows are features, columns are individuals
        
        Y : scalar, numpy array
            Value yielded by the true function when given X as input
        
        n_epoch : scalar, optional
            Number of epoch to train the model, 1 by default
        
        verbose : boolean, optional
            Display informations relative to training
        
        graph : boolean, optional
            Display the graph of the elbo regarding the epoch
        """
        elbo_values = []
        mse_values = []
        for epoch in range(n_epoch):
            self.learning_rate = self.lr(epoch)
            self.generate_param()
            
            if graph or verbose:
                elbo_val = self.elbo(X, Y)
                y_pred = self.predict(X).reshape(1, -1)
                mse = (y_pred - Y).dot((y_pred - Y).T)**0.5

            if graph:
                elbo_values.append(elbo_val)
                mse_values.append(mse[0])

            if verbose:
                print("Epoch", epoch+1, "is starting :")
                print('- MSE : %2.2f, ELBO : %2.2f' % (mse, elbo_val))
                self.loading_bar(epoch, n_epoch)

            if adam:
                self.bayesbybackprop_withadam(X, Y)
            else:
                self.bayesbybackprop_noadam(X, Y)

        if graph :
            _, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].plot(list(range(n_epoch)), elbo_values, color='r')
            axs[0].set_xlabel("Epoch")
            axs[0].set_ylabel("ELBO")

            axs[1].plot(list(range(n_epoch)), mse_values, color='g')
            axs[1].set_xlabel("Epoch")
            axs[1].set_ylabel("MSE")

            plt.subplots_adjust(wspace=0.3)
            plt.show()

    def combine_gradients(self, nn_grad, bayes_grad, k):
        """
        Add the weighted gradient of the functional model and the elbo regarding the mean and scale

        Return
        -------
        (1-k)* nn_grad + k * bayes_grad
        """
        return tuple((1-k)*nn + k * bayes for nn, bayes in zip(nn_grad, bayes_grad))

    def loading_bar(self, epoch, n_epoch):
        """
        Print a loading bar
        """
        percent = (epoch+1) / n_epoch
        num_bars = int(percent * 20)
        print('\r' + '[' + '='*num_bars + ' '*(20-num_bars) + ']', end='')
        print(f' {percent*100:.1f}% complete', end='\n')

    
    def debug_info(self, index, layer, weights, biases, weight_partial_derivative_w_log_likelihood, bias_partial_derivative_w_log_likelihood):
        """
        Print the values of the partial derivatives for debug purpose
        """
        print("Layer :", index, "\n")
        print("--------- Values of the partial derivatives for the weights ---------\n")
        print("Partial derivative of : \n log(q(w/phi) : \n", layer.weight.partial_derivative_w_log_prob(weights), "\nlog(p(w)) : \n", layer.weight_prior.partial_derivative_w_log_prob(weights), "\n log(p(D/w)) : \n", weight_partial_derivative_w_log_likelihood[index], "\n f(w, phi) selon mu : \n", layer.weight.partial_derivative_mu_log_prob(weights), "\n f(w, phi) selon sigma : \n", layer.weight.partial_derivative_sigma_log_prob(weights), "\n")
        print("--------- Values of the partial derivatives for the biases ---------\n")
        print("Partial derivative of : \n log(q(w/phi) : \n", layer.bias.partial_derivative_w_log_prob(biases), "\nlog(p(w)) : \n", layer.bias_prior.partial_derivative_w_log_prob(biases), "\n log(p(D/w)) : \n", bias_partial_derivative_w_log_likelihood[index], "\n f(w, phi) selon mu : \n", layer.bias.partial_derivative_mu_log_prob(biases), "\n f(w, phi) selon sigma : \n", layer.bias.partial_derivative_sigma_log_prob(biases), "\n")
    
