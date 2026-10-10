# Deep Learning Principles: Backpropagation and Optimization

Deep neural networks compose layered parametric functions to approximate complex nonlinear mappings. Efficient training requires calculating accurate gradient vectors with respect to millions of network parameters across deep computational graphs.

## Backpropagation and the Computational Graph

Backpropagation applies the calculus chain rule recursively backward from scalar loss outputs to intermediate weight tensors. During the forward pass, activations are evaluated and stored in memory. In the backward pass, upstream gradient signals multiply local Jacobian matrices to determine weight sensitivities: dL/dw = (dL/dy) * (dy/dw). Reverse-mode automatic differentiation reduces computational complexity from exponential to linear with respect to node count. Efficient tensor memory caching is critical because high batch sizes rapidly saturate GPU high-bandwidth memory.

## Activation Functions and the Dying ReLU Problem

Nonlinear activation functions enable multi-layer perceptrons to transcend linear composition. Classical activations like Sigmoid and Hyperbolic Tangent (Tanh) saturate for large absolute inputs, driving derivatives toward zero and triggering vanishing gradient collapse in deep architectures.

The Rectified Linear Unit, ReLU(z) = max(0, z), resolves saturation for positive activations by maintaining a constant derivative of 1. However, ReLU introduces the "dying ReLU" problem: if large negative gradient updates push a neuron's pre-activation into strictly negative territory for all dataset inputs, the local gradient becomes permanently zero. The neuron stops updating and becomes functionally inactive. Variants such as Leaky ReLU, Parametric ReLU (PReLU), and Gaussian Error Linear Units (GELU) alleviate dead zones by providing non-zero slopes for negative domains.

## Vanishing Gradients and Xavier/Glorot Initialization

When training deep networks, repeated matrix multiplications during backward passes cause gradient norms to either diminish exponentially (vanishing gradients) or escalate unboundedly (exploding gradients). 

Proper weight initialization stabilizes signal variances across forward activations and backward gradients. Xavier (Glorot) initialization draws initial weights from a distribution with variance Var(W) = 2 / (n_in + n_out), matching input and output variance under linear or symmetric sigmoid/tanh activations. For asymmetric rectifiers such as ReLU and GELU, He (Kaiming) initialization doubles this variance to Var(W) = 2 / n_in, preventing activation variances from collapsing toward zero across dozens of sequential layers.
