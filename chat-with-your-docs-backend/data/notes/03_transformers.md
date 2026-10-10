# Sequence Modeling: Transformer Architectures and Attention

The Transformer architecture replaced recurrent architectures by eliminating sequential recurrence in favor of parallelized attention mechanisms. This design achieved superior computational throughput on hardware accelerators and effectively resolved long-range context degradation.

## Scaled Dot-Product Self-Attention

Self-attention computes pairwise semantic interactions among all tokens in a sequence simultaneously. Given input embedding matrix X, linear projection matrices W_Q, W_K, and W_V project tokens into Query (Q), Key (K), and Value (V) subspaces. The attention compatibility matrix is calculated via scaled dot-products:

Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) ) V

Dividing by the square root of the key projection dimension d_k counteracts dot products growing excessively large for high dimensions. Without scaling, large dot products push the softmax operator into regions with infinitesimal gradient slopes, stifling training dynamics. The resulting attention matrix assigns soft alignment weights to each token representation.

## Multi-Head Attention Mechanisms

Single-head attention averages semantic relationships across a single subspace, potentially blurring disparate linguistic dependencies. Multi-Head Attention (MHA) resolves this by running h parallel attention heads, each projecting queries, keys, and values through independent linear weights W_i^Q, W_i^K, and W_i^V.

Head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

The individual head output matrices are concatenated along the feature axis and passed through an output projection matrix W_O:

MultiHead(Q, K, V) = Concat(Head_1, ..., Head_h) W_O

This allows models to simultaneously track syntactic dependencies, semantic affinities, and distant coreference markers across distinct representation subspaces.

## Positional Encoding Formulations

Because self-attention is permutation-equivariant, the model possesses no intrinsic sense of sequence order. Positional encodings must be injected into token representations to preserve temporal sequence information.

Sinusoidal positional encodings interweave deterministic sine and cosine functions of varying frequencies across dimension channels: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) and PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)). This enables linear transformations to extrapolate relative distance offsets. Modern architectures commonly employ learnable absolute embeddings or relative formulations like Rotary Position Embeddings (RoPE) that apply complex rotation matrices directly to query and key vectors.
