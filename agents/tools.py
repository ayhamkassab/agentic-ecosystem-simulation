from typing import Dict




def intervention_to_dict(action_vector, species_names):

    #Maps model output to structured intervention.

    return {
        name: float(delta)
        for name, delta in zip(species_names, action_vector)
    }


