# part 2/recommend.py
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Set

import numpy as np


@dataclass(frozen=True)
class Config:
    embeddings_npz: str = "movie_embeddings.npz"
    index_json: str = "movie_index.json"
    user_profile_json: str = "user_profile.json"  # you create this
    top_k: int = 20

    # Popularity smoothing weight (0 = none, 1 = only popularity)
    pop_alpha: float = 0.10

    # Optional: downweight “weak likes”
    min_abs_rating: float = 0.0


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    ids = data["ids"]
    emb = data["emb"]
    meta = json.loads(str(data["meta"]))
    return ids, emb, meta


def load_index(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_user(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v
    return v / n


def build_user_embedding(
    emb: np.ndarray,
    id_to_row: Dict[str, int],
    ratings: List[Dict[str, Any]],
    min_abs_rating: float,
) -> Tuple[np.ndarray, Set[int]]:
    """
    ratings: list of { "id": <tmdb_id>, "rating": <float> }
      - rating can be any scale; typical: 1..5 or -1..+1
      - sign controls like/dislike if you want that behavior
    """
    seen: Set[int] = set()
    acc = np.zeros((emb.shape[1],), dtype=np.float32)
    wsum = 0.0

    for r in ratings:
        mid = r.get("id")
        if mid is None:
            continue
        seen.add(int(mid))

        rating = float(r.get("rating", 1.0))
        if abs(rating) < min_abs_rating:
            continue

        row = id_to_row.get(str(mid))
        if row is None:
            continue

        w = rating
        acc += emb[row] * w
        wsum += abs(w)

    if wsum == 0.0:
        # fallback: average of seen (unweighted) if any
        if not seen:
            return acc, seen
        for mid in seen:
            row = id_to_row.get(str(mid))
            if row is not None:
                acc += emb[row]
                wsum += 1.0

    if wsum > 0.0:
        acc = acc / float(wsum)

    acc = l2_normalize(acc)
    return acc.astype(np.float32), seen


def popularity_scores(items: List[Dict[str, Any]]) -> np.ndarray:
    # log1p(popularity) to compress heavy tail
    pops = np.array([float(it.get("popularity") or 0.0) for it in items], dtype=np.float32)
    pops = np.log1p(np.maximum(pops, 0.0))
    mx = float(pops.max()) if pops.size else 1.0
    if mx <= 0.0:
        return pops
    return pops / mx  # [0,1]


def recommend():
    cfg = Config()

    ids, emb, meta = load_npz(cfg.embeddings_npz)
    index = load_index(cfg.index_json)
    items = index["items"]
    id_to_row = index["id_to_row"]

    user = load_user(cfg.user_profile_json)
    ratings = user.get("ratings", [])
    user_vec, seen = build_user_embedding(emb, id_to_row, ratings, cfg.min_abs_rating)

    if float(np.linalg.norm(user_vec)) == 0.0:
        raise SystemExit("User embedding is zero. Provide at least 1 valid rated movie id present in the index.")

    # Cosine similarity since both are normalized => dot product
    sims = emb @ user_vec  # shape (n_movies,)

    # Popularity smoothing
    pop = popularity_scores(items)
    final = sims * (1.0 - cfg.pop_alpha) + pop * cfg.pop_alpha

    # Exclude seen ids
    seen_rows = set()
    for mid in seen:
        row = id_to_row.get(str(mid))
        if row is not None:
            seen_rows.add(row)

    candidate_idx = [i for i in range(len(items)) if i not in seen_rows]
    candidate_idx = np.array(candidate_idx, dtype=np.int64)

    cand_scores = final[candidate_idx]
    top = min(cfg.top_k, int(candidate_idx.size))
    if top <= 0:
        return []

    # partial sort then order
    top_part = np.argpartition(-cand_scores, kth=top - 1)[:top]
    top_idx = candidate_idx[top_part]
    top_idx = top_idx[np.argsort(-final[top_idx])]

    out = []
    for i in top_idx:
        it = items[int(i)]
        out.append(
            {
                "id": it["id"],
                "title": it.get("title"),
                "year": it.get("year"),
                "score": float(final[int(i)]),
                "sim": float(sims[int(i)]),
                "pop_norm": float(pop[int(i)]),
                "popularity": it.get("popularity"),
                "vote_average": it.get("vote_average"),
                "vote_count": it.get("vote_count"),
            }
        )

    return out


def main():
    recs = recommend()
    print(json.dumps(recs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
