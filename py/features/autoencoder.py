from torch import nn, Tensor
from typing import Iterable

class AutoEncoder(nn.Module):
    def __init__(self, *sizes: list[int]) -> None:
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            *[layer for (in_, out) in zip(sizes[:-2], sizes[1:-1]) 
                    for layer in (nn.Linear(in_, out), nn.LeakyReLU())],
            nn.Linear(*sizes[-2:]),
        )
        self.decoder = nn.Sequential(
            *[layer for (in_, out) in zip(sizes[2:], sizes[1:-1])
                    for layer in (nn.LeakyReLU(), nn.Linear(in_, out))][::-1],
            nn.Linear(*sizes[1::-1]),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        hidden_representation = self.encoder(x)
        reconstructed         = self.decoder(hidden_representation)
        return reconstructed
    
    @property
    def trainable_parameters(self) -> Iterable[nn.parameter.Parameter]:
        return [*self.encoder.parameters(), *self.decoder.parameters()]