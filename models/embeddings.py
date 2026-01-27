import torch
import torch.nn as nn



class StateEmbedding(nn.Module):

    #Encodes structured world state into a latent space.



    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:

        #state: [batch, num_entities]


        x = self.linear(state)
        return self.norm(x)
        
        
        
        
