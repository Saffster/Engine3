# build_meshes.py
import json
import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Any


IN_JSON = "tmdb_movies_full.json"
OUT_JSON = "tmdb_movies_with_meshes.json"


@dataclass(frozen=True)
class MeshConfig:
    director_weight: int = 2
    actor_prefix: str = "actor"
    director_prefix: str = "dir"
    genre_prefix: str = "genre"
    keyword_prefix: str = "kw"


_ws_re = re.compile(r"\s+")


def clean_token_value(x: str) -> str:
    # Ensure the TF-IDF tokenization (split by spaces) remains stable.
    # - collapse whitespace
    # - strip
    # Keep punctuation (like Zoë, apostrophes) intact.
    s = _ws_re.sub(" ", str(x)).strip()
    return s


def tokens(prefix: str, items: Iterable[str]) -> List[str]:
    out = []
    for x in items or []:
        if x is None:
            continue
        v = clean_token_value(str(x))
        if not v:
            continue
        out.append(f"{prefix}:{v}")
    return out


def main():
    with open(IN_JSON, "r", encoding="utf-8") as f:
        movies: List[Dict[str, Any]] = json.load(f)

    cfg = MeshConfig()

    for m in movies:
        genres = m.get("genres", []) or []
        keywords = m.get("keywords", []) or []
        actors = m.get("actors", []) or []
        directors = m.get("directors", []) or []

        m["genre_mesh"] = " ".join(tokens(cfg.genre_prefix, genres))

        people_tokens: List[str] = []
        people_tokens += tokens(cfg.actor_prefix, actors)
        people_tokens += tokens(cfg.director_prefix, directors) * max(1, cfg.director_weight)
        m["people_mesh"] = " ".join(people_tokens)

        m["theme_mesh"] = " ".join(tokens(cfg.keyword_prefix, keywords))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)

    print("Wrote:", OUT_JSON)
    if movies:
        print(
            "Sample meshes:",
            {k: movies[0].get(k) for k in ["genre_mesh", "people_mesh", "theme_mesh"]},
        )


if __name__ == "__main__":
    main()
