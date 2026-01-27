from dataclasses import dataclass
from typing import Dict, List

import torch
#from typing import List


@dataclass
class Species:
    name: str
    population: float
    growth_rate: float


class EcosystemGraph:
    #Graph-based world model.
    #Nodes = species
    #Edges = interactions (predation, competition, symbiosis)

    def __init__(self):
        self.species: Dict[str, Species] = {}
        #self.interactions: Dict[str, Dict[]] = {}
        self.interactions: Dict[str, Dict[str, float]] = {}

    def add_species(self, name: str, population: float, growth_rate: float):
        self.species[name] = Species(name, population, growth_rate)
        self.interactions[name] = {}

    def add_interaction(self, src: str, dst: str, strength: float):

        #strength > 0 : positive influence
        #strength < 0 : negative influence

        self.interactions[src][dst] = strength

    def state_vector(self) -> torch.Tensor:

        #Returns ordered population vector

        return torch.tensor(
            [s.population for s in self.species.values()],
            dtype=torch.float32
        )

    def species_names(self) -> List[str]:
        #return list(self.species())
        return list(self.species.keys())
