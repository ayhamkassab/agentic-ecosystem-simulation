import torch
from .ecosystem_graph import EcosystemGraph


class EcosystemDynamics:

    #Continuous-time dynamics approximated in discrete steps


    def __init__(self, graph: EcosystemGraph):
        self.graph = graph

    def step(self, dt: float = 1.0):
        delta = {}

        for src, species in self.graph.species.items():
            growth = species.growth_rate * species.population
            #interaction_effect = 0
            interaction_effect = 0.0

            for dst, strength in self.graph.interactions[src].items():
                interaction_effect += (
                    strength * self.graph.species[dst].population
                )

            delta[src] = dt * (growth + interaction_effect)


        for name, change in delta.items():
            self.graph.species[name].population = max(
                0.0,
                self.graph.species[name].population + change
            )


    def rollout(self, steps: int, dt: float = 1.0) -> torch.Tensor:
        states = []
        for _ in range(steps):
            self.step(dt)
            states.append(self.graph.state_vector())
        return torch.stack(states)

