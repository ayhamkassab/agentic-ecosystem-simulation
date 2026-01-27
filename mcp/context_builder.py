from mcp.protocol import MCPContext


class ContextBuilder:

    #Constructs structured context for agent / LLM reasoning.


    def build(self, graph, constraints, history, feedback=None):
        state = {
            name: species.population
            for name, species in graph.species.items()
        }

        return MCPContext(
            world_state=state,
            constraints={"min_population": constraints.min_population},
            history=history,
            human_feedback=feedback or {}
        )



