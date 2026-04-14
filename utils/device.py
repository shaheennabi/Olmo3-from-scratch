import torch


class Device:
    """
    Device selection utility.

    Chooses the best available compute backend:
    CUDA (NVIDIA GPU) > MPS (Apple GPU) > CPU
    """

    @staticmethod
    def get():
        if torch.cuda.is_available():
            # NVIDIA GPU
            return torch.device("cuda")

        elif torch.backends.mps.is_available():
            # Apple Silicon GPU
            return torch.device("mps")

        else:
            # CPU fallback
            return torch.device("cpu")