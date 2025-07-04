import torch

data = torch.load("model/bellman_gamma01/bellman_checkpoint_4000.pth", map_location="cpu", weights_only=False)

print(data.keys())
print(data["model_state_dict"].keys())
