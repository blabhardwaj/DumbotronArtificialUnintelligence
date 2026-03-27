import torch
import numpy as np


def countParameter(MODEL_SAVE_PATH):
    state_dict = torch.load(MODEL_SAVE_PATH, map_location="cpu")
    total_params = sum(p.numel() for p in state_dict.values())
    token_ids = np.load("data/input.npy")
    total_tokens = len(token_ids)

    print(f"Total number of tokens: {total_tokens}")
    print("Total parameters (millions):", round(total_params / 1e6, 2), "m")
