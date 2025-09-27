# inference_server.py
from __future__ import annotations
from typing import Optional, Dict, Any, Tuple, Callable, List
import os
import numpy as np
import torch
import torch.nn.functional as F
from time import time
from .model import TwoTowerBPR

import pandas as pd

def load_movie_id_to_title(csv_path: str) -> dict:
    """
    Build a dict {movie_id -> title} from movie_data.csv.
    """
    df = pd.read_csv(csv_path)
    if "movie_id" not in df.columns or "title" not in df.columns:
        raise ValueError("movie_data.csv must have 'movie_id' and 'title' columns")
    return dict(zip(df["movie_id"], df["title"]))


def recommendations_with_titles(
    recs: dict,
    id_to_title: dict,
) -> dict:
    """
    Replace 'item_ids' in recommendations dict with movie titles when available.
    Keeps scores and user_source intact.
    """
    ids = recs["item_ids"]
    titles = [id_to_title.get(mid, str(mid)) for mid in ids]
    return {
        "titles": titles,
        "scores": recs.get("scores"),
        "user_source": recs.get("user_source", None),
    }


class StatelessModelServer:
    """
    Stateless server for two-tower retrieval inference.

    Typical flow:
      1) server = StatelessModelServer(...).load_model(...).load_item_features(...)
      2) server.precompute_item_embeddings(..., save_npz=...)    # one-time job
      3) server.load_item_embeddings(saved_npz)
      4) server.get_recommendations(user_id=..., top_k=10)       # known user
         or
         server.get_recommendations(user_features=..., top_k=10) # ad-hoc/new user

    Cold-start policy (new users):
      - If user_features not provided & user_id not found:
          use global mean of user feature matrix (simple, explainable fallback).
      - You can override by passing a custom `user_encoder_fn` to build features from a raw profile dict.
    """

    def __init__(
        self,
        *,
        device: Optional[torch.device] = None,
        user_features_npz: Optional[str] = None,
        movie_features_npz: Optional[str] = None,
        user_encoder_fn: Optional[Callable[[Dict[str, Any]], np.ndarray]] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TwoTowerBPR] = None
        self.user_feats: Optional[Dict[str, Any]] = None
        self.item_feats: Optional[Dict[str, Any]] = None

        self.user_encoder_fn = user_encoder_fn  # optional callable: raw_profile -> np.ndarray (Du,)

        # Preload feature stores (for known-user lookup / item precompute)
        if user_features_npz:
            self.user_feats = self._load_npz(user_features_npz, required=("X","ids"))
        if movie_features_npz:
            self.item_feats = self._load_npz(movie_features_npz, required=("X","ids"))

        # Runtime item embedding store
        self.item_ids: Optional[np.ndarray] = None        # (Ni,)
        self.item_embeds: Optional[torch.Tensor] = None   # (Ni, d) on device

        # Cached means (for cold-start fallback)
        self._user_feat_mean: Optional[np.ndarray] = None
        if self.user_feats is not None:
            self._user_feat_mean = self.user_feats["X"].mean(axis=0).astype(np.float32)
            noise = np.random.normal(
                loc=0.0,
                scale=0.1, 
                size=self._user_feat_mean.shape
            ).astype(np.float32)
            self._user_feat_mean += noise

    # ---------- IO helpers ----------
    @staticmethod
    def _load_npz(path: str, required: Tuple[str, ...]) -> Dict[str, Any]:
        data = np.load(path, allow_pickle=True)
        for k in required:
            if k not in data.files:
                raise KeyError(f"{path} missing key '{k}'")
        out = {k: data[k] for k in data.files}
        return out

    # ---------- Model loading ----------
    def load_model(
        self,
        state_dict_path: str,
        *,
        du: Optional[int] = None,
        di: Optional[int] = None,
        d_embed: int = 128,
        user_hidden: Tuple[int, ...] = (512, 256),
        item_hidden: Tuple[int, ...] = (512, 256),
        p_drop: float = 0.1,
    ) -> "StatelessModelServer":
        """
        Instantiate model with dims inferred from loaded feature matrices unless du/di are provided.
        """
        if du is None:
            if self.user_feats is None:
                raise ValueError("du not provided and user feature NPZ not loaded.")
            du = int(self.user_feats["X"].shape[1])
        if di is None:
            if self.item_feats is None:
                raise ValueError("di not provided and movie feature NPZ not loaded.")
            di = int(self.item_feats["X"].shape[1])

        model = TwoTowerBPR(du=du, di=di, d_embed=d_embed,
                            user_hidden=user_hidden, item_hidden=item_hidden,
                            p_drop=p_drop).to(self.device)
        sd = torch.load(state_dict_path, map_location=self.device)
        model.load_state_dict(sd)
        model.eval()
        self.model = model
        return self

    # ---------- Item embeddings ----------
    @torch.no_grad()
    def precompute_item_embeddings(
        self,
        *,
        batch_size: int = 2048,
        save_npz: Optional[str] = None,
        force: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode all item feature rows -> item embedding matrix (Ni, d).
        Optionally save to NPZ {item_ids, item_embeds}.

        Will reuse cached embeddings unless force=True.
        """
        if self.item_embeds is not None and not force:
            # already computed, just return cached
            return self.item_ids, self.item_embeds.cpu().numpy()

        if self.model is None:
            raise RuntimeError("Model not loaded.")
        if self.item_feats is None:
            raise RuntimeError("Movie features NPZ not loaded.")

        X_items = self.item_feats["X"].astype(np.float32)   # (Ni, Di)
        item_ids = self.item_feats["ids"]                   # (Ni,)
        Ni, Di = X_items.shape

        embeds = []
        for start in range(0, Ni, batch_size):
            end = min(start + batch_size, Ni)
            x = torch.from_numpy(X_items[start:end]).to(self.device)
            e = self.model.encode_item(x)
            e = F.normalize(e, dim=-1)  # ensure normalized
            embeds.append(e.cpu().numpy())

        item_embeds = np.vstack(embeds).astype(np.float32)
        if save_npz:
            os.makedirs(os.path.dirname(save_npz) or ".", exist_ok=True)
            np.savez_compressed(save_npz, item_ids=item_ids, item_embeds=item_embeds)

        # cache in memory
        self.item_ids = item_ids
        self.item_embeds = torch.from_numpy(item_embeds).to(self.device)

        return self.item_ids, item_embeds

    def load_item_embeddings(self, npz_path: str) -> "StatelessModelServer":
        data = self._load_npz(npz_path, required=("item_ids", "item_embeds"))
        self.item_ids = data["item_ids"]
        self.item_embeds = torch.from_numpy(data["item_embeds"].astype(np.float32)).to(self.device)
        return self

    # ---------- User feature helpers ----------
    def _lookup_user_features(self, user_id: Any) -> Optional[np.ndarray]:
        if self.user_feats is None:
            return None
        # Build id -> row map lazily
        if not hasattr(self, "_user_id_to_idx"):
            ids_list = self.user_feats["ids"].tolist()
            self._user_id_to_idx = {uid: i for i, uid in enumerate(ids_list)}
        idx = self._user_id_to_idx.get(user_id, None)
        if idx is None:
            return None
        return self.user_feats["X"][idx].astype(np.float32)

    def _cold_start_user_features(self, profile: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Default cold-start: global mean of user feature matrix (very simple).
        If `user_encoder_fn` is provided and `profile` is given, use it to compute features.
        """
        if profile and self.user_encoder_fn is not None:
            feats = self.user_encoder_fn(profile)
            return feats.astype(np.float32)
        if self._user_feat_mean is not None:
            return self._user_feat_mean.copy()
        raise RuntimeError("No way to build user features: provide user_features.npz or a user_encoder_fn.")

    # ---------- Scoring ----------
    @torch.no_grad()
    def _encode_user(self, user_features: np.ndarray) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        x = torch.from_numpy(user_features.astype(np.float32)).unsqueeze(0).to(self.device)  # [1, Du]
        u = self.model.encode_user(x)  # [1, d]
        return F.normalize(u, dim=-1)  # safe normalization (idempotent if already normalized)

    @torch.no_grad()
    def get_recommendations(
        self,
        *,
        top_k: int = 10,
        user_id: Optional[Any] = None,
        user_features: Optional[np.ndarray] = None,
        cold_start_profile: Optional[Dict[str, Any]] = None,
        return_scores: bool = True,
    ) -> Dict[str, Any]:
        """
        Recommend top-K items.

        Priority for user representation:
          1) user_features (explicit array)
          2) lookup by user_id in user_features.npz
          3) cold-start fallback (global mean or user_encoder_fn(profile))

        Returns:
          { "item_ids": np.ndarray (K,), "scores": np.ndarray (K,), "user_source": str }
        """
        if self.item_embeds is None or self.item_ids is None:
            raise RuntimeError("Item embeddings not loaded. Run precompute_item_embeddings(...) or load_item_embeddings(...).")
        # Build user vector
        source = "unknown"
        if user_features is not None:
            uf = user_features.astype(np.float32)
            source = "provided_features"
        elif user_id is not None:
            print(f"Looking up user_id={user_id}")
            uf = self._lookup_user_features(user_id)
            print(f"Lookup result: {uf}")
            if uf is not None:
                source = "lookup_user_id"
            else:
                uf = self._cold_start_user_features(cold_start_profile)
                source = "cold_start"
        else:
            uf = self._cold_start_user_features(cold_start_profile)
            source = "cold_start"

        u = self._encode_user(uf)            # [1, d]
        # Scores = dot with all items
        scores = (u @ self.item_embeds.T).squeeze(0)   # [Ni]
        vals, idx = torch.topk(scores, k=min(top_k, scores.numel()), largest=True, sorted=True)
        top_ids = self.item_ids[idx.cpu().numpy()]
        out = {"item_ids": top_ids}
        if return_scores:
            out["scores"] = vals.detach().cpu().numpy()
        out["user_source"] = source
        return out

if __name__ == "__main__":
    server = StatelessModelServer(
        user_features_npz="src/models/deep_rec_sys/features/user_features.npz",
        movie_features_npz="src/models/deep_rec_sys/features/movie_features.npz"
    ).load_model("src/models/deep_rec_sys/checkpoints/two_tower_bpr_best.pt")

    server.precompute_item_embeddings(save_npz="features/item_embeds.npz")
    server.load_item_embeddings("features/item_embeds.npz")

    id_to_title = load_movie_id_to_title("src/models/deep_rec_sys/fake_data/movie_data.csv")
    
    t0 = time()
    recs = server.get_recommendations(user_id=21343, top_k=20)
    # print(recs)
    dt = time() - t0
    print(f"Time taken: {dt*1000:.1f} ms")

    pretty = recommendations_with_titles(recs, id_to_title)
    print(pretty["user_source"])
    for title, score in zip(pretty["titles"], pretty["scores"]):
        print(f"{title:40s}  score={score:.3f}")

    # try unknown user
    t0 = time()
    recs2 = server.get_recommendations(user_id="unknown_user", top_k=20)
    # print(recs2)
    dt = time() - t0
    print(f"Time taken: {dt*1000:.1f} ms")

    pretty = recommendations_with_titles(recs2, id_to_title)
    print(pretty["user_source"])
    for title, score in zip(pretty["titles"], pretty["scores"]):
        print(f"{title:40s}  score={score:.3f}")