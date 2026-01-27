import torch
from agents.tools import intervention_to_dict



class AgentPlanner:

    #Agentic loop:
    #propose → simulate → evaluate → refine




    def __init__(self, policy, value_model, simulator, species_names):
        self.policy = policy
        self.value_model = value_model
        self.simulator = simulator
        self.species_names = species_names
        


    def propose(self, embedded_state):
        with torch.no_grad():
            return self.policy(embedded_state)



    def evaluate(self, final_state):
        return self.value_model(final_state)



    def plan(self, embedded_state):
        action = self.propose(embedded_state)
        intervention = intervention_to_dict(
            action.squeeze(0),
            self.species_names

        )


        if not self.simulator.constraints.check_feasibility(intervention):
            return None, None

        self.simulator.apply_intervention(intervention)
        trajectory = self.simulator.rollout()

        score = self.evaluate(trajectory[-1])
        return intervention, score



