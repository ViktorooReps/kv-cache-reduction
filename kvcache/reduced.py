"""
Implementation is based on OffloadedCache from transformers.
"""
from typing import List, Tuple, Optional, Dict, Any

import torch
from transformers import StaticCache


@torch.compile()
@torch.no_grad()
def kmeans_torch_reduction(X, num_clusters, num_iters=10):
    """
    This implementation is very memory-intensive and is not suitable for long context.

    X: Tensor of shape (B, H, L, D)
    num_clusters: Number of clusters C
    Returns: centroids of shape (B, H, C, D)
    """
    B, H, L, D = X.shape
    C = num_clusters
    device = X.device

    X = X.contiguous()

    # Initialize centroids by selecting random points
    shuffled_indices = torch.rand(B, H, L, device=device).argsort(dim=-1)
    indices = shuffled_indices[:, :, :C]
    centroids = torch.gather(X, 2, indices.unsqueeze(-1).expand(-1, -1, -1, D)).clone()

    assignments = torch.zeros((B, H, L), dtype=torch.int64, device=device)
    new_centroids = torch.zeros((B, H, C, D), device=device)
    cluster_counts = torch.zeros((B, H, C), device=device)

    for _ in range(num_iters):
        # Compute distances & assign clusters
        distances = torch.cdist(X, centroids)  # Shape: (B, H, L, C)
        assignments = torch.argmin(distances, dim=-1)  # Shape: (B, H, L)

        # Compute new centroids using vectorized operations
        one_hot_assignments = torch.nn.functional.one_hot(assignments, num_clusters).to(X.dtype)  # (B, H, L, C)
        new_centroids = torch.einsum('bhld,bhlc->bhcd', X, one_hot_assignments) / one_hot_assignments.sum(dim=2).clamp(min=1).unsqueeze(-1)

        # Reset empty clusters
        empty_clusters = one_hot_assignments.sum(dim=2) == 0
        if empty_clusters.any():
            new_samples = torch.randint(0, L, (B, H, num_clusters), device=device)
            new_centroids[empty_clusters] = X[torch.arange(B)[:, None, None], 
                                              torch.arange(H)[None, :, None], 
                                              new_samples][empty_clusters]

        centroids = new_centroids.clone()  # Update centroids

    return centroids, assignments



REDUCTION_IMPL = {
    'torch': kmeans_torch_reduction
}


class KVBiasCache(StaticCache):
    pass
