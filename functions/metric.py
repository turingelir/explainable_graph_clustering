"""
    This module is for metric functions to evaluate models.

"""
import torch
from torch import Tensor

from sklearn.metrics import v_measure_score as v_score

def v_measure_score(true_labels:Tensor, predicted_labels:Tensor)->float:
    """
    Calculate the V-measure score between two clusterings.
    
    This function is a wrapper around the sklearn.metrics.v_measure_score function, 
    for compatibility with PyTorch tensors that have batch dimensions.

    The V-measure is the harmonic mean between homogeneity and completeness:
    v = (1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness)

    Args:
        true_labels (Tensor): The ground truth labels. 
            Shape is [B, N] where B is the batch size and N is the number of samples/nodes.
        predicted_labels (Tensor): The predicted labels. 
            Shape is [B, N] where B is the batch size and N is the number of samples/nodes.

    Returns:
        float: The V-measure score.
    """
    metric = 0.
    for true, pred in zip(true_labels, predicted_labels):
        metric += v_score(true.cpu().numpy(), pred.cpu().numpy())
    return metric / len(true_labels)
     

if __name__ == "__main__":
    # Run some tests

    # Test v_measure_score 
    true_labels = torch.tensor([0, 0, 1, 1, 2, 2]).unsqueeze(0)
    predicted_labels = torch.tensor([0, 0, 2, 2, 1, 1]).unsqueeze(0)
    print(v_measure_score(true_labels, predicted_labels))

    # Create some tensors
    input_tensor = torch.randn(3, 36, 36, 1)
    target_tensor = torch.randn(10, 36, 36, 7)