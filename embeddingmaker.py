# build_embeddings.py
import json
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


IN_JSON = "tmdb_movies_with_meshes.json"
OUT_NPZ = "movie_embeddings.npz"


@dataclass(frozen=True)
class EmbedConfig:
    max_points: int = 100_000
    n_components: int = 128  # compact semantic embedding
    random_state: int = 42


def load_movies(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fit_embed(texts: List[str], cfg: EmbedConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
    vec = TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+")
    X = vec.fit_transform(texts)

    svd = TruncatedSVD(n_components=cfg.n_components, random_state=cfg.random_state)
    E = svd.fit_transform(X).astype(np.float32)

    meta = {
        "vocab_size": len(vec.vocabulary_),
        "explained_variance_sum": float(getattr(svd, "explained_variance_ratio_", np.array([])).sum()),
        "n_components": cfg.n_components,
    }
    return E, meta


def main():
    cfg = EmbedConfig()
    movies = load_movies(IN_JSON)[: cfg.max_points]

    # Combine the three meshes into one representation for general similarity.
    texts = [
        " ".join(
            [
                m.get("genre_mesh") or "",
                m.get("people_mesh") or "",
                m.get("theme_mesh") or "",
            ]
        ).strip()
        for m in movies
    ]

    E, meta = fit_embed(texts, cfg)
    ids = np.array([m.get("id") for m in movies], dtype=np.int64)

    np.savez_compressed(OUT_NPZ, ids=ids, emb=E, meta=json.dumps(meta))
    print("Saved:", OUT_NPZ, "shape:", E.shape, "meta:", meta)


if __name__ == "__main__":
    main()
