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
        weight = torch.nn.init.trunc_normal_(matrix, mean=0, std=sqrt(2/(in_features + out_features)), a = -3*sqrt(2/(in_features + out_features)), b=3*sqrt(2/(in_features + out_features)))
        
        self.weight = torch.nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")