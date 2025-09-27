from typing import Sequence, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- MLP builder ---------
def mlp_stack(dims: Sequence[int], p_drop: float = 0.1) -> nn.Sequential:
    """
    Builds: [Linear -> BatchNorm -> GELU -> Dropout] * (L-1) + final Linear (no BN/act/drop)
    Example dims: [Du, 512, 256, 128] -> 3 layers, output dim 128.
    """
    layers = []
    for i in range(len(dims) - 1):
        inp, out = dims[i], dims[i + 1]
        is_last = (i == len(dims) - 2)
        layers.append(nn.Linear(inp, out))
        if not is_last:
            layers += [nn.LayerNorm(out), nn.GELU(), nn.Dropout(p_drop)]
    return nn.Sequential(*layers)

# --------- Two-tower model (BPR only) ---------
class TwoTowerBPR(nn.Module):
    def __init__(
        self,
        du: int,  # user feature dim
        di: int,  # item feature dim
        d_embed: int = 128,
        user_hidden: Optional[Sequence[int]] = (512, 256),
        item_hidden: Optional[Sequence[int]] = (512, 256),
        p_drop: float = 0.1,
    ):
        super().__init__()
        self.user_mlp = mlp_stack([du, *user_hidden, d_embed], p_drop)
        self.item_mlp = mlp_stack([di, *item_hidden, d_embed], p_drop)

    def encode_user(self, u_feat: torch.Tensor) -> torch.Tensor:
        # u_feat: [B, Du] -> [B, d]
        z = self.user_mlp(u_feat)
        return F.normalize(z, dim=-1)

    def encode_item(self, m_feat: torch.Tensor) -> torch.Tensor:
        # m_feat: [*, Di] -> [*, d]  (B or B*K works)
        z = self.item_mlp(m_feat)
        return F.normalize(z, dim=-1)

    @staticmethod
    def bpr_loss(u_emb: torch.Tensor, v_pos: torch.Tensor, v_neg: torch.Tensor) -> torch.Tensor:
        """
        u_emb:  [B, d]
        v_pos:  [B, d]
        v_neg:  [B, K, d]
        Loss:  -mean(log σ( (u·v⁺) - (u·v⁻) ))
        """
        pos = (u_emb * v_pos).sum(dim=-1)                  # [B]
        neg = (u_emb.unsqueeze(1) * v_neg).sum(dim=-1)     # [B, K]
        diff = pos.unsqueeze(1) - neg                      # [B, K]
        return -F.logsigmoid(diff).mean()

    def forward(self, u_feat: torch.Tensor, m_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode_user(u_feat), self.encode_item(m_feat)

# --------- Negative sampling helpers (training-time only) ---------
@torch.no_grad()
def sample_neg_indices_uniform(
    num_items: int,
    pos_indices: torch.LongTensor,
    K: int,
    device: Optional[torch.device] = None,
) -> torch.LongTensor:
    """
    Uniformly sample K negatives per positive index, avoiding the positive itself.
    Returns: [B, K] long tensor of item indices in [0, num_items).
    Notes:
      - Duplicates among K are possible (fine for a first pass).
      - Runs entirely on the given device if provided.
    """
    B = pos_indices.shape[0]
    device = device or pos_indices.device
    # initial draw
    neg_idx = torch.randint(low=0, high=num_items, size=(B, K), device=device)
    # ensure no neg equals its positive; if equal, n = (pos+1) % N (cheap correction)
    equals_pos = (neg_idx == pos_indices.view(-1, 1))
    if equals_pos.any():
        neg_idx[equals_pos] = (pos_indices.view(-1, 1).expand_as(neg_idx)[equals_pos] + 1) % num_items
    return neg_idx

@torch.no_grad()
def gather_item_features(
    all_item_features: torch.Tensor,  # [Ni, Di] on same device as model
    neg_indices: torch.LongTensor,    # [B, K]
) -> torch.Tensor:
    """
    Gathers negative item feature rows.
    Returns: [B, K, Di]
    """
    B, K = neg_indices.shape
    Di = all_item_features.shape[1]
    v_neg = all_item_features.index_select(0, neg_indices.view(-1)).view(B, K, Di)
    return v_neg
