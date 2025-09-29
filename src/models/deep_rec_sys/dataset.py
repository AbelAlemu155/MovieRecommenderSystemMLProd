import torch
import os
import numpy as np 
from typing import Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader, random_split

def create_dataloaders(
    dataset,
    *,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int | None = None,
    pin_memory: bool = True,
):
    """
    Split a dataset into train/val/test and return DataLoaders.
    Ratios are applied to the full dataset length (remaining goes to train).
    """
    N = len(dataset)
    n_val  = int(N * val_ratio)
    n_test = int(N * test_ratio)
    n_train = N - n_val - n_test
    assert n_train > 0, "Train split is empty—reduce val/test ratios."

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=g)

    if num_workers is None:
        try:
            num_workers = max(0, (os.cpu_count() or 0) // 2)
        except Exception:
            num_workers = 0

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False
    )
    return train_loader, val_loader, test_loader

def _norm_id_scalar(x):
    """Normalize an ID so 123 == '123' == 123.0; else trimmed string."""
    # handle NaN/None
    if x is None:
        return None
    # numpy scalars -> python scalars
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass
    # try integer coercion (but avoid coercing '00123' mistakenly)
    try:
        # floats like 123.0 -> 123
        if isinstance(x, float) and x.is_integer():
            return int(x)
        # ints pass through
        if isinstance(x, (int, np.integer)):
            return int(x)
        # strings that look like ints
        s = str(x).strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        return s
    except Exception:
        return str(x).strip()

def _norm_id_array(arr):
    return np.array([_norm_id_scalar(v) for v in arr], dtype=object)

class MovieDataset(Dataset):
    def __init__(
        self,
        users: Dict[str, Any],
        movies: Dict[str, Any],
        eng: Dict[str, Any],
        *,
        return_ids: bool = False,
        sample_weight_col: Optional[str] = None,
    ):
        """
        Returns per sample:
          - user_feat:  FloatTensor [Du]
          - movie_feat: FloatTensor [Di]   (positive item features)
          - y:          FloatTensor [1]
          - pos_item_idx: LongTensor []    (row index into movies["X"])
          - (optional) weight, and/or ids if requested
        """
        # normalize ids
        user_ids_all = _norm_id_array(users["ids"])
        movie_ids_all = _norm_id_array(movies["ids"])
        e_uids = _norm_id_array(eng["user_ids"])
        e_mids = _norm_id_array(eng["movie_ids"])

        self.user_X: np.ndarray = users["X"]        # (Nu, Du) float32
        self.movie_X: np.ndarray = movies["X"]      # (Ni, Di) float32
        self.user_ids_all = users["ids"]
        self.movie_ids_all = movies["ids"]

        self.user_id_to_index = {uid: i for i, uid in enumerate(self.user_ids_all.tolist())}
        self.movie_id_to_index = {mid: i for i, mid in enumerate(self.movie_ids_all.tolist())}

        e_uids = np.asarray(eng["user_ids"], dtype=object)
        e_mids = np.asarray(eng["movie_ids"], dtype=object)
        e_y    = np.asarray(eng["engagement"], dtype=np.float32)

        u_idx = np.fromiter((self.user_id_to_index.get(u, -1) for u in e_uids), count=len(e_uids), dtype=np.int64)
        m_idx = np.fromiter((self.movie_id_to_index.get(i, -1) for i in e_mids), count=len(e_mids), dtype=np.int64)
        mask = (u_idx >= 0) & (m_idx >= 0)

        if not np.any(mask):
            raise ValueError("After alignment, no rows remain—ID mismatch between features and engagements?")

        self.u_idx = u_idx[mask]
        self.m_idx = m_idx[mask]
        self.y = e_y[mask].astype(np.float32)


        self.return_ids = return_ids
        if return_ids:
            self.u_ids = e_uids[mask]
            self.m_ids = e_mids[mask]
        else:
            self.u_ids = None
            self.m_ids = None

    
        if sample_weight_col and sample_weight_col in eng:
            w = np.asarray(eng[sample_weight_col], dtype=np.float32)
            self.w = w[mask]
        else:
            self.w = None

        # Cache dims
        self.Du = self.user_X.shape[1]
        self.Di = self.movie_X.shape[1]

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        u_feat = torch.from_numpy(self.user_X[self.u_idx[idx]])  # (Du,)
        m_feat = torch.from_numpy(self.movie_X[self.m_idx[idx]])  # (Di,)
        y = torch.from_numpy(np.asarray(self.y[idx:idx+1], dtype=np.float32))  # shape (1,)
        pos_item_idx = torch.tensor(self.m_idx[idx], dtype=torch.long) 

        if self.w is not None:
            w = torch.from_numpy(np.asarray(self.w[idx:idx+1], dtype=np.float32))
            if self.return_ids:
                return u_feat, m_feat, y, w, self.u_ids[idx], self.m_ids[idx]
            return u_feat, m_feat, y, w, pos_item_idx

        if self.return_ids:
            return u_feat, m_feat, y, self.u_ids[idx], self.m_ids[idx]
        return u_feat, m_feat, y, pos_item_idx

    def __repr__(self) -> str:
        n = len(self)
        return f"MovieDataset(n={n}, Du={self.Du}, Di={self.Di}, return_ids={self.return_ids}, weighted={self.w is not None})"
