# part2/build_movie_embeddings.py
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass(frozen=True)
class Config:
    in_json: str = "../tmdb_movies_with_meshes.json"
    out_npz: str = "movie_embeddings.npz"
    out_index: str = "movie_index.json"

    # TF-IDF tuning
    min_df: int = 2
    max_df: float = 0.60
    max_features: int = 120_000

    # Embedding size
    n_components: int = 128
    random_state: int = 42


def load_movies(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array.")
    return data


def movie_text(m: Dict[str, Any]) -> str:
    return " ".join(
        [
            (m.get("genre_mesh") or "").strip(),
            (m.get("people_mesh") or "").strip(),
            (m.get("theme_mesh") or "").strip(),
        ]
    ).strip()


def build_index(movies: List[Dict[str, Any]]) -> Dict[str, Any]:
    items = []
    id_to_row = {}
    for i, m in enumerate(movies):
        mid = m.get("id")
        if mid is None:
            continue
        id_to_row[str(mid)] = i
        items.append(
            {
                "id": mid,
                "title": m.get("title"),
                "year": m.get("year"),
                "popularity": m.get("popularity"),
                "vote_average": m.get("vote_average"),
                "vote_count": m.get("vote_count"),
                "poster_path": m.get("poster_path"),
                "backdrop_path": m.get("backdrop_path"),
            }
        )
    return {"items": items, "id_to_row": id_to_row}


def main():
    cfg = Config()

    if not os.path.exists(cfg.in_json):
        raise SystemExit(f"Missing input: {cfg.in_json}")

    movies = load_movies(cfg.in_json)
    texts = [movie_text(m) for m in movies]

    vec = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"[^ ]+",
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        max_features=cfg.max_features,
    )
    X = vec.fit_transform(texts)

    svd = TruncatedSVD(n_components=cfg.n_components, random_state=cfg.random_state)
    E = svd.fit_transform(X).astype(np.float32)

    # L2 normalize rows -> cosine = dot
    E = normalize(E, norm="l2", axis=1).astype(np.float32)

    ids = np.array([m.get("id", -1) for m in movies], dtype=np.int64)

    meta = {
        "n_movies": int(len(movies)),
        "embedding_dim": int(cfg.n_components),
        "tfidf_vocab_size": int(len(vec.vocabulary_)),
        "svd_explained_variance_sum": float(getattr(svd, "explained_variance_ratio_", np.array([])).sum()),
        "min_df": cfg.min_df,
        "max_df": cfg.max_df,
        "max_features": cfg.max_features,
    }

    np.savez_compressed(cfg.out_npz, ids=ids, emb=E, meta=json.dumps(meta))
    with open(cfg.out_index, "w", encoding="utf-8") as f:
        json.dump(build_index(movies), f, ensure_ascii=False, indent=2)

    print("Wrote:", cfg.out_npz)
    print("Wrote:", cfg.out_index)
    print("Meta:", meta)


if __name__ == "__main__":
    main()
