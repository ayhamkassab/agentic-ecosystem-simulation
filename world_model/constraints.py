from typing import Dict




class EcologicalConstraints:

    #Knowledge-based inference layer.
    #Hard constraints that override model suggestions.


    def __init__(self, min_population: float = 1e-3):
        self.min_population = min_population


    def validate(self, populations: Dict[str, float]) -> Dict[str, float]:

        #Enforce biologically plausible bounds.

        validated = {}
        for k, v in populations.items():
            validated[k] = max(self.min_population, v)

        return validated

    def check_feasibility(self, intervention: Dict[str, float]) -> bool:


        #Reject clearly invalid interventions.


        for delta in intervention.values():
            if abs(delta) > 1.0:
                return False
        return True
