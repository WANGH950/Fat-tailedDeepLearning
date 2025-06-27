# Deep Learning for High-dimensional PDEs with Fat-tailed L\'evy measure

Authors: Kamran Arif, Guojiang Xi, Heng Wang and Weihua Deng (dengwh@lzu.edu.cn)

Address: School of Mathematics and Statistics, State Key Laboratory of Natural Product Chemistry, Lanzhou University, Lanzhou 730000, China.

# Environment configuration

python==3.9 torch==2.2.1

# Abstract

The partial differential equations (PDEs) for jump process with L\'evy measure have wide applications.
When the measure has fat tails, it will bring big challenges for both computational cost and accuracy.
In this work, we develop a deep learning method for high-dimensional PDEs related to fat-tailed L\'evy measure, which can be naturally extended to the general case.
Building on the theory of backward stochastic differential equations (BSDEs) for L\'evy processes, our deep learning method avoids the need for neural network differentiation and introduces a novel technique to address the singularity of fat-tailed L\'evy measures.
The developed method is used to solve four kinds of high-dimensional PDEs: the diffusion equation with fractional Laplacian; the convective diffusion equation with fractional Laplacian; the convective diffusion reaction equation with fractional Laplacian; the nonlinear diffusion reaction equation with fractional Laplacian.
The parameter $\beta$ in fractional Laplacian is an indicator of the strength of the singularity of L\'evy measure.
Specifically, for $\beta\in (0,1)$, the model describes super-ballistic diffusion, whereas for $\beta\in (1,2)$, it characterizes super-diffusion.
Our method achieves a relative error of $\mathcal{O}(10^{-3})$ for low-dimensional problems and $\mathcal{O}(10^{-2})$ for high-dimensional problems.
We also investigate three factors that influence the algorithm's performance: (1) the number of hidden layers, (2) the number of Monte Carlo samples, and (3) the choice of activation functions.
Our numerical results demonstrate that the algorithm achieves optimal stability with deeper hidden layers, a larger number of Monte Carlo samples, and the Softsign activation function.

# How to use?
We provide an invocation example in the notebook file "./example.ipynb", where you can define and modify any of the components in the sample for testing.