import torch
import torch.nn as nn
from einops import rearrange, einsum
from math import sqrt


class Linear(torch.nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        matrix = torch.empty(out_features, in_features, device=device, dtype=dtype)

        # Linear weights initialization : N(µ = 0, σ2 =2/(din+dout )) truncated at [−3σ, 3σ]
        self.weight = torch.nn.Parameter(matrix)
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=sqrt(2/(in_features + out_features)), a = -3*sqrt(2/(in_features + out_features)), b=3*sqrt(2/(in_features + out_features)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        # Size of the vocabulary
        self.num_embeddings = num_embeddings

        # Dimension of the embedding vectors
        self.embedding_dim = embedding_dim
        matrix = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(matrix)
        torch.nn.init.trunc_normal_(self.weight, mean=0, std=1, a = -3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Simple indexing
        return self.weight[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, d_model)
        # original dtype
        in_dtype = x.dtype

        # upscale
        x = x.to(torch.float32)

        # dim = -1: only normalize on a, the last dimension
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        normalized = x / rms
        result = normalized * self.weight

        return result.to(in_dtype)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int,  device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        matrix = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        self.w1_weight = torch.nn.Parameter(matrix)
        torch.nn.init.trunc_normal_(self.w1_weight, mean=0, std=1, a = -3, b=3)

        matrix = torch.empty(d_model, d_ff, device=device, dtype=dtype)
        self.w2_weight = torch.nn.Parameter(matrix)
        torch.nn.init.trunc_normal_(self.w2_weight, mean=0, std=1, a = -3, b=3)

        matrix = torch.empty(d_ff, d_model, device=device, dtype=dtype)
        self.w3_weight = torch.nn.Parameter(matrix)
        torch.nn.init.trunc_normal_(self.w3_weight, mean=0, std=1, a = -3, b=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        def silu(a: torch.Tensor) -> torch.Tensor:
            return a * torch.sigmoid(a)

        right = silu(einsum(self.w1_weight, x, "dff d_model, ... d_model -> ... dff")) * einsum(self.w3_weight, x, "dff d_model, ... d_model -> ... dff")
        return einsum(self.w2_weight, right, "d_model dff, ... dff -> ... d_model")

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        rotation_matrices = torch.empty(self.max_seq_len, self.d_k, self.d_k)

        # Create arange for positions from 0 to max_seq_len, for each i
        positions = torch.arange(0, self.max_seq_len, device=device)

        # for pair i, frequency is 2*i/d_k
        freqs = torch.arange(0, self.d_k, 2, device=device) / self.d_k
        
        # Compute angles: position / (theta ** freq)
        # Shape: (max_seq_len, d_k // 2)
        # squeeze and unsqueeze makes broadcasting work
        angles = positions.unsqueeze(-1) / (self.theta ** freqs.unsqueeze(0))
        
        cos_theta = torch.cos(angles)  # Shape: (max_seq_len, d_k // 2)
        sin_theta = torch.sin(angles)  # Shape: (max_seq_len, d_k // 2)

        # Apply RoPE rotation: for each pair (2i, 2i+1), apply 2D rotation matrix
        # [cos  -sin]
        # [sin   cos]
        # We need to set diagonal elements and off-diagonal elements separately
        
        idx = torch.arange(self.d_k // 2, device=device)
        even_idx = 2 * idx
        odd_idx = even_idx + 1

        rotation_matrices[:, even_idx, even_idx] = cos_theta[:, idx]
        rotation_matrices[:, odd_idx, odd_idx] = cos_theta[:, idx]
        rotation_matrices[:, even_idx, odd_idx] = -sin_theta[:, idx]
        rotation_matrices[:, odd_idx, even_idx] = sin_theta[:, idx]

        # constant so does not need to be a torch.nn parameter
        self.rotation_matrices = rotation_matrices

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # input tensor of shape (..., seq_len, d_k)
        # token positions of shape (..., seq_len)
        # output tensor of shape (..., seq_len, d_k)

        def apply_rope_rotations(x, token_positions, rotation_matrices):

            # Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
            # Tolerate x with an arbitrary number of batch dimensions.
            # Assume that the token positions are a tensor of shape (..., seq_len) specifying the token
            # positions of x along the sequence dimension.
            # Use the token positions to slice your (possibly precomputed) cos and sin tensors
            # along the sequence dimension.

            # x: shape (..., seq_len, d_k)
            # token_positions: shape (..., seq_len)
            # rotation_matrices: shape (max_seq_len, d_k, d_k)

            selected_rotations = rotation_matrices[token_positions]  # Shape: (..., seq_len, d_k, d_k)
            
            # Apply rotations using einops.einsum
            # selected_rotations: (..., seq_len, d_k, d_k)
            # x: (..., seq_len, d_k)
            # Output: (..., seq_len, d_k)
            return einsum(selected_rotations, x, "... s i j, ... s j -> ... s i")

        return apply_rope_rotations(x, token_positions, self.rotation_matrices)

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,  mask: torch.Tensor | None = None) -> torch.Tensor:
    # q (batch_size, ..., n, d_k) 
    # k (batch_size, ..., m, d_k) 
    # v (batch_size, ..., m, d_v)
    # mask (batch_size, ..., n, m)

    d_k = q.shape[-1]

    #Q^T @ K 
    e_matrix = einsum(q, k, "... n d_k, ... m d_k -> ... n m") / sqrt(d_k)

    if mask is not None:
  
        e_matrix = e_matrix.masked_fill(~mask, -float('inf'))

    a_matrix = softmax(e_matrix, dim=-1)

    return einsum(a_matrix, v, "... n m, ... m d_v -> ... n d_v")


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False, theta=None, max_seq_len: int | None = None, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.Q = nn.Parameter(torch.empty(num_heads*self.d_k, d_model, device=device, dtype=dtype))
        self.K = nn.Parameter(torch.empty(num_heads*self.d_k, d_model, device=device, dtype=dtype))
        self.V = nn.Parameter(torch.empty(num_heads*self.d_v, d_model, device=device, dtype=dtype))
        self.O = nn.Parameter(torch.empty(d_model, num_heads*self.d_v, device=device, dtype=dtype))
        self.use_rope = use_rope
        self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device) if (use_rope and theta is not None and max_seq_len is not None) else None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q_batch = rearrange(self.Q, "(head d_k) d_model -> head d_k d_model", head=self.num_heads, d_k=self.d_k)
        K_batch = rearrange(self.K, "(head d_k) d_model -> head d_k d_model", head=self.num_heads, d_k=self.d_k)
        V_batch = rearrange(self.V, "(head d_v) d_model -> head d_v d_model", head=self.num_heads, d_v=self.d_v)

        # get query, key, value vectors per head per token
        WQx = einsum(Q_batch, x, "head d_k d_model, ... seq d_model -> ... head seq d_k")
        WKx = einsum(K_batch, x, "head d_k d_model, ... seq d_model -> ... head seq d_k")
        WVx = einsum(V_batch, x, "head d_v d_model, ... seq d_model -> ... head seq d_v")

        # Apply RoPE to queries and keys if enabled (after projection)
        if self.rope is not None and token_positions is not None:
            WQx = self.rope(WQx, token_positions)
            WKx = self.rope(WKx, token_positions)

        seq_len = x.shape[-2]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        multihead = scaled_dot_product_attention(WQx, WKx, WVx, causal_mask)
        multihead = rearrange(multihead, "... head seq d_v -> ... seq (head d_v)", head=self.num_heads, d_v=self.d_v)
        return einsum(self.O, multihead, "d_model d_out, ... seq d_out -> ... seq d_model")




