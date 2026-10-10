# Supervised Learning Foundations: Linear and Logistic Regression

Linear and logistic regression serve as fundamental parametric models in supervised statistical learning. These algorithms establish the mathematical baseline for mapping independent feature vectors to continuous targets and categorical likelihoods.

## Linear Regression and Ordinary Least Squares

Linear regression models the relationship between an input vector x and a continuous target y through a linear combination of weights w and bias b: y = w^T x + b. In Ordinary Least Squares (OLS), the objective is minimizing the Residual Sum of Squares (RSS), measuring the discrepancy between observed and predicted values. Under the Gauss-Markov assumptions—linearity, homoscedasticity, no multicollinearity, and zero error expectation—OLS produces the Best Linear Unbiased Estimator (BLUE). The closed-form analytical solution is obtained via the normal equation w = (X^T X)^(-1) X^T y. However, when features outnumber samples or feature correlation is high, the matrix X^T X becomes singular or ill-conditioned, necessitating iterative optimization via gradient descent.

## Logistic Regression and Cross-Entropy Loss

Logistic regression extends linear modeling to classification by mapping real-valued affine transformations through the sigmoid activation function: sigma(z) = 1 / (1 + exp(-z)). This squashes outputs into the open interval (0, 1), representing the posterior probability P(Y=1|x). Because squared error produces a non-convex surface when composed with the sigmoid, parameter estimation relies on Maximum Likelihood Estimation (MLE). Minimizing the negative log-likelihood yields the binary cross-entropy loss: L = -[y log(p) + (1-y) log(1-p)]. Unlike OLS, logistic regression lacks a closed-form solution and requires convex optimization routines such as Newton-Raphson, L-BFGS, or stochastic gradient descent.

## Regularization Strategies: Ridge, Lasso, and ElasticNet

High-dimensional data frequently leads to overfitting, where models memorize training noise and suffer from high generalization variance. Regularization mitigates this pathology by penalizing model complexity in the loss objective.

L2 Regularization (Ridge) adds the squared Euclidean norm of weights (lambda * ||w||_2^2). It shrinks coefficient magnitudes uniformly toward zero without setting them strictly to zero, effectively stabilizing matrix inversion in multicollinear settings.

L1 Regularization (Lasso) penalizes absolute weight magnitudes (lambda * ||w||_1). Because the L1 ball contains sharp corners on coordinate axes, gradient updates drive non-informative feature weights strictly to zero, performing intrinsic feature selection and yielding sparse parameter representations.

ElasticNet combines both penalties using a convex mixture parameter alpha, preserving Lasso's sparsity while retaining Ridge's grouping effect across correlated predictors.
