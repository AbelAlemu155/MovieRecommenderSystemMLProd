# load data
from .utils.data_loading import (
    load_user_features_npz, load_movie_features_npz, load_engagements_npz, quick_summary, peek_ids
)
from .model import TwoTowerBPR, sample_neg_indices_uniform, gather_item_features
from .dataset import MovieDataset, create_dataloaders
import torch
import os
from tqdm.auto import tqdm
from time import time

def train_one_epoch(
    model,
    loader,
    optimizer,
    all_item_feats: torch.Tensor,   # [Ni, Di] on device
    device: torch.device,
    K: int = 5,                     # negatives per positive
) -> float:
    model.train()
    running_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        # unpack (expecting your current dataset return)
        u_feat, m_feat, y, pos_item_idx = batch
        u_feat = u_feat.to(device, non_blocking=True)
        m_feat = m_feat.to(device, non_blocking=True)
        pos_item_idx = pos_item_idx.to(device, non_blocking=True)

        # sample negatives -> gather features
        neg_idx = sample_neg_indices_uniform(
            num_items=all_item_feats.shape[0],
            pos_indices=pos_item_idx,
            K=K,
            device=device,
        )  # [B, K]
        v_neg_feat = gather_item_features(all_item_feats, neg_idx)  # [B, K, Di]

        # encode
        u_emb = model.encode_user(u_feat)                                  # [B, d]
        v_pos = model.encode_item(m_feat)                                  # [B, d]
        B, Kneg, Di = v_neg_feat.shape
        v_neg = model.encode_item(v_neg_feat.view(B * Kneg, Di)).view(B, Kneg, -1)  # [B, K, d]

        # loss
        loss = TwoTowerBPR.bpr_loss(u_emb, v_pos, v_neg)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(1, n_batches)

@torch.no_grad()
def eval_one_epoch(
    model,
    loader,
    all_item_feats: torch.Tensor,
    device: torch.device,
    K: int = 5,
    cached_negs=None
) -> float:
    """
    Same BPR objective on the val set (no optimizer step) just to track loss.
    """
    model.eval()
    running_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(tqdm(loader, desc="Val  ", leave=False)):
        u_feat, m_feat, y, pos_item_idx = batch
        u_feat = u_feat.to(device, non_blocking=True)
        m_feat = m_feat.to(device, non_blocking=True)
        pos_item_idx = pos_item_idx.to(device, non_blocking=True)

        neg_idx = cached_negs[i].to(device) if cached_negs is not None else sample_neg_indices_uniform(
            all_item_feats.shape[0], pos_item_idx.to(device), K, device
        )
        v_neg_feat = gather_item_features(all_item_feats, neg_idx)

        u_emb = model.encode_user(u_feat)
        v_pos = model.encode_item(m_feat)
        B, Kneg, Di = v_neg_feat.shape
        v_neg = model.encode_item(v_neg_feat.view(B * Kneg, Di)).view(B, Kneg, -1)

        loss = TwoTowerBPR.bpr_loss(u_emb, v_pos, v_neg)
        running_loss += loss.item()
        n_batches += 1

    return running_loss / max(1, n_batches)


if __name__ == "__main__":
    # load data
    users = load_user_features_npz("src/models/deep_rec_sys/features/user_features.npz")
    movies = load_movie_features_npz("src/models/deep_rec_sys/features/movie_features.npz")
    eng    = load_engagements_npz("src/models/deep_rec_sys/features/engagements.npz")

    print(quick_summary(users, movies, eng))
    peek_ids(users, movies, eng)

    # create dataset class instance
    dataset = MovieDataset(users, movies, eng)
    print(f"Dataset length: {len(dataset)}")

    # create dataloader for train, val and test
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=256,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # create model, optimizer, training loop
    users_dim = users["X"].shape[1]
    items_dim = movies["X"].shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TwoTowerBPR(du=users_dim, di=items_dim, d_embed=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-6)

    # Preload all item features once
    all_item_feats = torch.from_numpy(movies["X"]).to(device)  # [Ni, Di]

    K = 40  # negatives per positive

    def precompute_negatives(loader, num_items, K, device):
        neg_cache = []
        for batch in loader:
            *rest, pos_item_idx = batch  # works with your dataset returns
            pos_item_idx = pos_item_idx.to(device)
            neg_idx = sample_neg_indices_uniform(num_items, pos_item_idx, K, device)
            neg_cache.append(neg_idx)  # store [B,K] for this batch order
        return neg_cache

    VAL_NEG = precompute_negatives(val_loader, all_item_feats.shape[0], K=K, device=device)
    TEST_NEG = precompute_negatives(test_loader, all_item_feats.shape[0], K=K, device=device)

    EPOCHS = 30
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        t0 = time()
        train_loss = train_one_epoch(model, train_loader, opt, all_item_feats, device, K=K)
        val_loss = eval_one_epoch(model, val_loader, all_item_feats, device, K=K, cached_negs=VAL_NEG)
        dt = time() - t0

        print(f"[Epoch {epoch}/{EPOCHS}] "
            f"train_bpr={train_loss:.4f}  val_bpr={val_loss:.4f}  ({dt:.1f}s)")

        sch.step()

        if val_loss < best_val:
            best_val = val_loss
            # save weights
            # make dir
            os.makedirs("src/models/deep_rec_sys/checkpoints", exist_ok=True)
            ckpt_path = "src/models/deep_rec_sys/checkpoints/two_tower_bpr_best.pt"
            print(f"  New best val bpr, saving model to {ckpt_path}")
            torch.save(model.state_dict(), ckpt_path)

    # test loss
    test_loss = eval_one_epoch(model, test_loader, all_item_feats, device, K=K)
    print(f"Test BPR loss: {test_loss:.4f}")


