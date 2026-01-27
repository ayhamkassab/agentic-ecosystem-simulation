import torch
import torch.nn as nn




class ValueModel(nn.Module):

    #Scores outcomes of simulated trajectories.
    #Analogous to pathway importance or target value.



    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)

        )



    def forward(self, state: torch.Tensor) -> torch.Tensor:

        #state: [batch, num_entities]


        return self.net(state).squeeze(-1)



