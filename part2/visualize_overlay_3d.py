# part2/visualize_overlay_3d.py
import json
from pathlib import Path
from typing import Dict, Any, List, Set

import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# Import your recommender function
# recommend.py must have recommend() that returns list of dicts with "id"
from recommend import recommend as run_recommend


EMB_NPZ = "movie_embeddings.npz"
INDEX_JSON = "movie_index.json"
USER_JSON = "user_profile.json"
OUT_HTML = "recs_overlay_3d.html"


def load_npz(path: str):
    data = np.load(path, allow_pickle=False)
    ids = data["ids"].astype(np.int64)
    emb = data["emb"].astype(np.float32)
    meta = json.loads(str(data["meta"]))
    return ids, emb, meta


def load_index(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_user_seen(path: str) -> Set[int]:
    with open(path, "r", encoding="utf-8") as f:
        u = json.load(f)
    seen = set()
    for r in u.get("ratings", []):
        mid = r.get("id")
        if mid is not None:
            seen.add(int(mid))
    return seen


def make_hover(item: Dict[str, Any]) -> str:
    title = item.get("title", "")
    year = item.get("year", "")
    pop = item.get("popularity", None)
    va = item.get("vote_average", None)
    vc = item.get("vote_count", None)
    return (
        f"<b>{title}</b> ({year})<br>"
        f"id: {item.get('id')}<br>"
        f"popularity: {pop}<br>"
        f"vote_average: {va} (n={vc})"
    )


def main():
    ids, emb, _meta = load_npz(EMB_NPZ)
    index = load_index(INDEX_JSON)
    items: List[Dict[str, Any]] = index["items"]

    # 3D projection for visualization (not used for scoring)
    pca = PCA(n_components=3, random_state=42)
    X3 = pca.fit_transform(emb)

    # Sets
    seen_ids = load_user_seen(USER_JSON)
    recs = run_recommend()
    rec_ids = {int(r["id"]) for r in recs}

    # Build masks
    all_mask = np.ones(len(ids), dtype=bool)
    seen_mask = np.array([int(mid) in seen_ids for mid in ids], dtype=bool)
    rec_mask = np.array([int(mid) in rec_ids for mid in ids], dtype=bool)

    # Hover text aligned with rows
    hover = [make_hover(items[i]) if i < len(items) else f"id: {ids[i]}" for i in range(len(ids))]

    fig = go.Figure()

    # Background: all movies (faint)
    fig.add_trace(
        go.Scatter3d(
            x=X3[all_mask, 0],
            y=X3[all_mask, 1],
            z=X3[all_mask, 2],
            mode="markers",
            name="All movies",
            marker=dict(size=2, color="rgba(255, 0, 0, 0.25)"),
            text=np.array(hover, dtype=object)[all_mask],
            hoverinfo="text",
        )
    )

    # Watched / Seen (blue)
    if seen_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=X3[seen_mask, 0],
                y=X3[seen_mask, 1],
                z=X3[seen_mask, 2],
                mode="markers",
                name="Watched",
                marker=dict(size=5, color="rgba(30,120,255,0.95)"),
                text=np.array(hover, dtype=object)[seen_mask],
                hoverinfo="text",
            )
        )

    # Recommended (green)
    if rec_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=X3[rec_mask, 0],
                y=X3[rec_mask, 1],
                z=X3[rec_mask, 2],
                mode="markers",
                name="Recommended",
                marker=dict(size=6, color="rgba(0,200,80,0.95)"),
                text=np.array(hover, dtype=object)[rec_mask],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Movie space with recommendation overlay (PCA 3D from embeddings)",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    fig.write_html(OUT_HTML, include_plotlyjs="cdn")
    print("Wrote:", OUT_HTML)
    print("Seen:", int(seen_mask.sum()), "Recommended:", int(rec_mask.sum()), "Total:", len(ids))


if __name__ == "__main__":
    main()
