## Abstract

Implementation of a Bayesian Neural network, using only numpy library. This project was made when I didn't know PyTorch's powerful framework, that is why I only used numpy. So I implemented everything by hand, at least I tried to : the gradients, the optimizers, the neural network ... As you can expect, this project is not as optimized as it could be if I used PyTorch but the objective behind this project was just education, to learn more things. The papers I used for the theory are linked in the [Reference](#reference) section.

### Prerequisites

The following libraries have to be installed :
  * numpy
  * matplotlib

## Theory

If you want some explanation on what is going on, feel free to look at the [notebook](theory.ipynb/).

## Usage

An [example](example.ipynb/) has been made to understand how to use the project and see what kind of results you can get.

## Reference

  * [Weight Uncertainty in Neural Network](https://arxiv.org/pdf/1505.05424.pdf)
  * [Hands-on Bayesian Neural Networks](https://arxiv.org/pdf/2007.06823.pdf)
  * [AdamB: Decoupled Bayes by Backprop with Gaussian Scale Mixture Prior](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9874837)
