import torch


def stability_score(trajectories: torch.Tensor) -> float:

    #Measures variance across rollouts.


    return trajectories.var(dim=0).mean().item()


