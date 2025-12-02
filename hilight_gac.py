#GraphAttentionConcat module
#Input: x of shape (B,N,F) where F = 56 from our LocalEncoderMLP
#Neighbor Index tensor: neighbor_index of shape (N,K) with K = 4 (the 4 neighbors per node/intersection)
#See the paper's description for details on how the attention is computed and concatenated on in section 4.2.2
    #eij = LeakyReLU(a^T[hi || hj]) where hi and hj are the feature vectors of nodes i and j
    # aij = softmax_j(eij) over all neighbors j of node i
    # zi = hi || ai1*hj1 || aij2*hj2 || ... || aijK*hjK
#Output: z of shape (B,N,(K+1)*F) where K=4 and F=56 -> output dim = 280
#F can be thought of as the feature dimension per node, K is the number of neighbors per node

import torch
import torch.nn as nn
import torch.nn.functional as F

#Graph Attention Concat (GAC) module for HiLight-style node features.
#Inputs:
    #  x: (B, N, F)   - encoded node features (from LocalEncoderMLP)
    #  neighbor_index: (N, K) - indices of K neighbors for each node
    #Output:
    #  z: (B, N, (K+1)*F) - concatenation of self + each attention-weighted neighbor
class GraphAttentionConcat(nn.Module):

    def __init__(self, in_dim=56, num_neighbors=4, negative_slope=0.2):
        #Negative slope for LeakyReLU comes from the GAT paper Velickovic et al. 2018 where they used LeakyReLU with slope 0.2
        #This was cited in the paper so I just used the same value here
        super().__init__()
        self.in_dim = in_dim
        self.num_neighbors = num_neighbors
        self.leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

        # a^T [h_i || h_j]  -> scalar
        # a has shape (2*F,)
        self.attn_vec = nn.Parameter(torch.empty(2 * in_dim))
        nn.init.xavier_uniform_(self.attn_vec.view(1, -1))

    
    def forward(self, x, neighbor_index):

        #x: (B, N, F) float tensor
        #neighbor_index: (N, K) long tensor (same N as in x)
        B, N, F = x.shape
        K = self.num_neighbors
        assert F == self.in_dim, f"Expected in_dim={self.in_dim}, got {F}"
        assert neighbor_index.shape == (N, K), \
            f"neighbor_index should be (N,{K}), got {neighbor_index.shape}"

        #In this section we are building all the components needed for attention computation
        #These are outline in sectio 4.2.2 of the HiLight paper (2025)
        #All formulas are from that paper

        #1- neighbor embeddings: (B, N, K, F)
        # neighbor_index is independent of batch, so we use it on the node dim
        neighbors = x[:, neighbor_index, :]            # (B, N, K, F)

        #2- build [h_i || h_j] for all neighbors 
        h_self = x.unsqueeze(2).expand(B, N, K, F)     # (B, N, K, F)
        pair_feat = torch.cat([h_self, neighbors], dim=-1)  # (B, N, K, 2F)

        #3- compute attention scores e_{ij}
        #   e_ij = LeakyReLU(a^T [h_i || h_j])
        attn_vec = self.attn_vec.view(1, 1, 1, 2 * F)  # broadcast
        e_ij = (pair_feat * attn_vec).sum(dim=-1)      # (B, N, K)
        e_ij = self.leaky_relu(e_ij)

        #4- normalize over neighbors with softmax -> alpha_{ij}
        alpha = F.softmax(e_ij, dim=-1)                # (B, N, K)

        #5- attention-weighted neighbor features
        alpha_expanded = alpha.unsqueeze(-1)           # (B, N, K, 1)
        neigh_weighted = alpha_expanded * neighbors    # (B, N, K, F)

        #6- concat self + each weighted neighbor separately
        #    z_i = h_i || alpha_i1 h_j1 || ... || alpha_iK h_jK
        parts = [x]  # (B, N, F)
        for k in range(K):
            parts.append(neigh_weighted[:, :, k, :])   # (B, N, F) each

        z = torch.cat(parts, dim=-1)                   # (B, N, (K+1)*F)
        return z