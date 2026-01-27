import torch
import torch.nn as nn
from typing import Optional




class TransformerPolicy(nn.Module):

    #Transformer-based reasoning model.
    #Learns intervention policies conditioned on system state.



    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        action_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
            
        )


        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.policy_head = nn.Linear(embed_dim, action_dim)
        
        

    def forward(
        self,
        embedded_state: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        #embedded_state: [batch, seq_len, embed_dim]

        encoded = self.encoder(embedded_state, src_key_padding_mask=mask)
        #pooled = encoded.mean()
        pooled = encoded.mean(dim=1)
        
        return self.policy_head(pooled)
        
        
        
        
        
