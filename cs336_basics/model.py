import torch
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
