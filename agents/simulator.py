import torch
from world_model.dynamics import EcosystemDynamics
from world_model.constraints import EcologicalConstraints


class Simulator:

    #Runs constrained rollouts under proposed interventions.


    def __init__(self, dynamics: EcosystemDynamics, constraints: EcologicalConstraints):
        self.dynamics = dynamics
        self.constraints = constraints



    def apply_intervention(self, intervention):
        for name, delta in intervention.items():
            self.dynamics.graph.species[name].population += delta



    def rollout(self, steps: int = 10) -> torch.Tensor:
        return self.dynamics.rollout(steps)



