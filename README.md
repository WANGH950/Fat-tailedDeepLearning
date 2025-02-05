# Deep Learning for High-dimensional PDEs with Fat-tailed L\'evy measure

Authors: Kamran Arif, Heng Wang and Weihua Deng (dengwh@lzu.edu.cn)

Address: School of Mathematics and Statistics, State Key Laboratory of Natural Product Chemistry, Lanzhou University, Lanzhou 730000, China.

# Environment configuration

python==3.9 torch==2.2.1

# Abstract

The partial differential equations (PDEs) for jump process with L\'evy measure has wide applications.
When the measure has fat tails, it will bring big challenges for both computational cost and accuracy.
Here we develop a deep learning method for high-dimensional PDEs related to fat-tailed L\'evy measure, which can be naturally extended to the general case.
Based on the theory of backward stochastic differential equation (BSDE) for L\'evy process, in developing the deep learning method, the differentiation of neural network is circumvented, and the technique is introduced to treat the singularity of the fat-tailed L\'evy measure.
The developed method is used to solve four kinds of high-dimensional PDEs: the diffusion equation with fractional Laplacian; the convective diffusion equation with fractional Laplacian; the convective diffusion reaction equation with fractional Laplacian; the nonlinear diffusion reaction equation with fractional Laplacian.
The order of $\beta$ of fractional Laplacian is an indicator of the strength of singularity of L\'evy measure.
In fact, when $\beta\in (0,1)$, it describes the super ballistic diffusion; while $\beta\in (1,2)$, it characterizes super diffusion.
Our method reachs the relative error of $\mathcal{O}(10^{-3})$ in low-dimensional problems, and $\mathcal{O}(10^{-2})$ in high-dimensional problems.
We also test three factors that could affect the algorithm: the number of hidden layers; the number of Monte Carlo sampling samples; activation functions. 
The test results show that the algorithm is the most stable when deeper hidden layers, larger Monte Carlo sampling numbers and Softsign activation function are selected.

# How to use?
We provide an invocation example in the notebook file "./example.ipynb", where you can define and modify any of the components in the sample for testing.