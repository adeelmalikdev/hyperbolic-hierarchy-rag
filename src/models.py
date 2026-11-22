"""
PyTorch models for Euclidean and Hyperbolic (Lorentz model) hierarchy-aware retrieval.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def exp_map_origin(v, max_norm=5.0):
    """
    Exponential map from tangent space at origin (1, 0, ..., 0)
    into the Lorentz hyperboloid model H^d.
    v: tensor of shape (..., d)
    Returns: tensor of shape (..., d + 1) where -x0^2 + sum(xi^2) = -1.
    """
    norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp_min(1e-7)
    scaled_norm = torch.clamp(norm, max=max_norm)
    v_scaled = v * (scaled_norm / norm)
    
    x0 = torch.cosh(scaled_norm)
    x_rest = (torch.sinh(scaled_norm) / scaled_norm) * v_scaled
    return torch.cat([x0, x_rest], dim=-1)

def lorentz_distance(u, v):
    """
    Computes hyperbolic distance between points u and v on the Lorentz hyperboloid.
    u: shape (..., d+1)
    v: shape (..., d+1)
    Returns: shape (...)
    """
    minkowski_prod = -u[..., 0] * v[..., 0] + torch.sum(u[..., 1:] * v[..., 1:], dim=-1)
    z = (-minkowski_prod).clamp_min(1.0 + 1e-6)
    return torch.log(z + torch.sqrt((z**2 - 1.0).clamp_min(1e-12)))

def lorentz_distance_matrix(u, v):
    """
    Pairwise Lorentz distance between u (N, d+1) and v (M, d+1).
    Returns (N, M) matrix of distances.
    """
    # minkowski = -u0*v0 + sum(ui*vi)
    u0 = u[:, 0:1] # (N, 1)
    v0 = v[:, 0:1].t() # (1, M)
    u_rest = u[:, 1:] # (N, d)
    v_rest = v[:, 1:].t() # (d, M)
    
    minkowski = -torch.matmul(u0, v0) + torch.matmul(u_rest, v_rest)
    z = (-minkowski).clamp_min(1.0 + 1e-6)
    return torch.log(z + torch.sqrt((z**2 - 1.0).clamp_min(1e-12)))

DEPTH_NORMS = {
    0: 0.5, # product (root, near centre)
    1: 1.2, # version
    2: 2.0, # category
    3: 2.8, # article
    4: 3.6  # passage (leaves, near boundary)
}

class HyperbolicRetriever(nn.Module):
    """
    Lorentz hyperbolic retrieval model with learned linear projection
    and taxonomy embeddings.
    """
    def __init__(self, in_dim=384, proj_dim=384, num_taxonomy_nodes=1000):
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        
        # Tangent space projection at origin
        self.proj = nn.Linear(in_dim, proj_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        
        # Taxonomy nodes in tangent space
        self.taxonomy_emb = nn.Embedding(num_taxonomy_nodes, proj_dim)
        nn.init.normal_(self.taxonomy_emb.weight, mean=0.0, std=0.02)
        
    def project_tangent(self, x):
        return self.proj(x)
        
    def encode(self, x):
        v = self.project_tangent(x)
        return exp_map_origin(v)
        
    def encode_taxonomy(self, node_indices):
        v = self.taxonomy_emb(node_indices)
        return exp_map_origin(v)
        
    def compute_similarity(self, q_hyper, p_hyper):
        """Higher is better (negative hyperbolic distance)."""
        dist = lorentz_distance_matrix(q_hyper, p_hyper)
        return -dist

class EuclideanRetriever(nn.Module):
    """
    Euclidean control with matching projection and taxonomy embeddings.
    """
    def __init__(self, in_dim=384, proj_dim=384, num_taxonomy_nodes=1000):
        super().__init__()
        self.in_dim = in_dim
        self.proj_dim = proj_dim
        
        self.proj = nn.Linear(in_dim, proj_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        
        self.taxonomy_emb = nn.Embedding(num_taxonomy_nodes, proj_dim)
        nn.init.normal_(self.taxonomy_emb.weight, mean=0.0, std=0.02)
        
    def encode(self, x):
        v = self.proj(x)
        return v
        
    def encode_taxonomy(self, node_indices):
        return self.taxonomy_emb(node_indices)
        
    def compute_similarity(self, q_vec, p_vec):
        """Higher is better (negative Euclidean distance)."""
        # Pairwise euclidean distance
        dist = torch.cdist(q_vec, p_vec, p=2.0)
        return -dist
