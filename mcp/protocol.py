from dataclasses import dataclass
from typing import Dict, Any




@dataclass
class MCPContext:

    #Minimal Model Context Protocol abstraction.


    world_state: Dict[str, float]
    constraints: Dict[str, Any]
    history: list
    human_feedback: Dict[str, Any]



