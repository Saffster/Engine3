# build_graph_edges.py
import json
from collections import defaultdict
from typing import Dict, Any, List, Tuple


IN_JSON = "tmdb_movies_with_meshes.json"
OUT_GRAPH_JSON = "movie_mesh_graph.json"


def split_tokens(mesh: str) -> List[str]:
    mesh = mesh or ""
    return [t for t in mesh.split(" ") if t]


def main():
    with open(IN_JSON, "r", encoding="utf-8") as f:
        movies: List[Dict[str, Any]] = json.load(f)

    # Build a bipartite graph: movie nodes and token nodes.
    # node ids:
    #   movie:<tmdb_id>
    #   token:<token>
    nodes = {}
    edges: List[Tuple[str, str]] = []

    token_df = defaultdict(int)

    for m in movies:
        mid = m.get("id")
        if not mid:
            continue
        mnode = f"movie:{mid}"
        nodes[mnode] = {"type": "movie", "id": mid, "title": m.get("title"), "year": m.get("year")}

        all_tokens = []
        all_tokens += split_tokens(m.get("genre_mesh"))
        all_tokens += split_tokens(m.get("people_mesh"))
        all_tokens += split_tokens(m.get("theme_mesh"))

        # Deduplicate per-movie to avoid huge repeated edges;
        # keep “weighting” in the mesh text for embeddings instead.
        seen = set()
        for tok in all_tokens:
            if tok in seen:
                continue
            seen.add(tok)

            tnode = f"token:{tok}"
            if tnode not in nodes:
                nodes[tnode] = {"type": "token", "token": tok}
            edges.append((mnode, tnode))
            token_df[tok] += 1

    out = {
        "nodes": nodes,
        "edges": [{"source": s, "target": t} for (s, t) in edges],
        "token_document_frequency": dict(token_df),
        "counts": {"nodes": len(nodes), "edges": len(edges), "movies": sum(1 for k in nodes if k.startswith("movie:"))},
    }

    with open(OUT_GRAPH_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print("Wrote:", OUT_GRAPH_JSON, "nodes:", out["counts"]["nodes"], "edges:", out["counts"]["edges"])


if __name__ == "__main__":
    main()
