import torch


def robustness_test(model, embedded_state, noise_std=0.01):
    noisy = embedded_state + noise_std * torch.randn_like(embedded_state)
    out1 = model(embedded_state)
    out2 = model(noisy)
    return torch.norm(out1 - out2).item()


