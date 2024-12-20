"""
    This module is for metric functions to evaluate models.

"""
import torch
from torch import Tensor



if __name__ == "__main__":
    # Run some tests

    # Test Frobenius distance

    # Create some tensors
    input_tensor = torch.randn(3, 36, 36, 1)
    target_tensor = torch.randn(10, 36, 36, 7)