
import torch
from torch import Tensor, nn
import math


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        l: int,
    ) -> None:
        super(PositionalEncoding, self).__init__()
        self.l = l

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        bs = x.shape[0]
        freq = torch.tensor([(2.0 ** t) for t in range(self.l)]).reshape(-1, 1).to(x.device)
        p = torch.matmul(freq, x.reshape(bs,1,2)).reshape(bs, -1)
        return torch.cat([torch.sin(p), torch.cos(p)], 1)

class SIREN(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        first_layer: bool = False,
    ):
        super().__init__(in_features, out_features, bias)
        if first_layer:
            nn.init.uniform_(self.weight, -30.0 / in_features, 30.0 / in_features)
        else:
            nn.init.uniform_(
                self.weight,
                -math.sqrt(6.0 / in_features),
                math.sqrt(6.0 / in_features),
            )

    def forward(self, input):
        return torch.sin(super().forward(input))

class GaussianMLP(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias: bool = True,
        sigma2: float = 1.0,
    ):
        super().__init__(in_features, out_features, bias)
        self.sigma2 = sigma2

    def forward(self, input):
        return torch.exp(-super().forward(input).square()) * (0.5/self.sigma2)

class ReLUNeuralField(nn.Module):
    def __init__(
        self,
        use_pe:bool = True,
        pe_dim: int = 10,
        layer_count: int = 4,
        layer_width: int = 64
    ) -> None:
        super(ReLUNeuralField, self).__init__()
        layers = []
        if use_pe:
            layers.append(PositionalEncoding(pe_dim))
            layers.append(nn.Linear(pe_dim*4, layer_width))
        else:
            layers.append(nn.Linear(2, layer_width))
        layers.append(nn.ReLU())

        for i in range(layer_count-1):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_width, 3))
        #layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*tuple(layers))

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        return self.network(x)


class SIRENNeuralField(nn.Module):
    def __init__(
        self,
        use_pe:bool = False,
        pe_dim: int = 0,
        layer_count: int = 4,
        layer_width: int = 64
    ) -> None:
        super(SIRENNeuralField, self).__init__()
        layers = []
        layers.append(SIREN(2, layer_width, bias=True, first_layer=True))
        for i in range(layer_count-1):
            layers.append(SIREN(layer_width, layer_width, bias=True, first_layer=False))
        layers.append(nn.Linear(layer_width, 3))
        #layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*tuple(layers))

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        return self.network(x)

class GARF(nn.Module):
    def __init__(
        self,
        use_pe:bool = False,
        pe_dim: int = 0,
        layer_count: int = 4,
        layer_width: int = 64
    ) -> None:
        super(GARF, self).__init__()
        layers = []
        layers.append(GaussianMLP(2, layer_width, bias=True))
        for i in range(layer_count-1):
            layers.append(GaussianMLP(layer_width, layer_width, bias=True))
        layers.append(nn.Linear(layer_width, 3))
        #layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*tuple(layers))

    def forward(
        self,
        x: Tensor
    ) -> Tensor:
        return self.network(x)