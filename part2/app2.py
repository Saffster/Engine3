# part2/app.py

import base64
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from sklearn.decomposition import PCA

# --- TMDB poster settings ---
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/"
POSTER_SIZE = "w342"


@dataclass(frozen=True)
class Config:
    embeddings_npz: str = "movie_embeddings.npz"
    index_json: str = "movie_index.json"
    meshes_json: str = "../tmdb_movies_with_meshes.json"
    hemo_logo_path: str = "hemo_logo.png"
    top_k_default: int = 20
    profiles_dir: str = "user_profiles"
    legacy_profile_json: str = "user_profile.json"


# ---------------------------
# Styling
# ---------------------------
def inject_css(watermark_logo_b64: Optional[str]):
    watermark_logo_html = ""
    if watermark_logo_b64:
        watermark_logo_html = f"""
          <img src="data:image/png;base64,{watermark_logo_b64}" style="height:28px;width:auto;border-radius:6px;opacity:0.95;" />
        """

    watermark_html = f"""
    <div class="hemo-watermark">
      {watermark_logo_html}
      <div class="hemo-watermark-text">
        <div class="hemo-watermark-small">POWERED BY</div>
        <div class="hemo-watermark-big">HEMOdata</div>
      </div>
    </div>
    """

    st.markdown(
        f"""
        <style>
        .stApp {{ background: #0b0f19; color: #e8eaf0; }}
        .block-container {{ padding-top: 1.1rem; max-width: 1450px; }}

        section[data-testid="stSidebar"] {{
            background: #0f1526;
            border-right: 1px solid rgba(255,255,255,0.06);
        }}

        .card {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 16px 16px 12px 16px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.25);
        }}
        .card-title {{
            font-size: 1.05rem;
            font-weight: 800;
            margin: 0 0 0.55rem 0;
            color: #ffffff;
        }}
        .muted {{ color: rgba(255,255,255,0.65); font-size: 0.92rem; }}

        .stButton>button {{
            border-radius: 12px;
            background: linear-gradient(90deg, #7c3aed, #2563eb);
            border: 0px;
            color: white;
            font-weight: 750;
            padding: 0.6rem 1rem;
        }}

        [data-testid="stMainMenu"] {{ visibility: hidden; }}
        [data-testid="stHeader"] {{ visibility: hidden; }}
        [data-testid="stFooter"] {{ visibility: hidden; }}

        .hemo-watermark {{
            position: fixed;
            right: 18px;
            bottom: 14px;
            z-index: 9999;
            display: flex;
            align-items: center;
            gap: 10px;
            background: rgba(15, 21, 38, 0.75);
            border: 1px solid rgba(255,255,255,0.10);
            backdrop-filter: blur(10px);
            padding: 10px 12px;
            border-radius: 16px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.35);
        }}
        .hemo-watermark-text {{
            display: flex;
            flex-direction: column;
            line-height: 1.05;
        }}
        .hemo-watermark-small {{
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            color: rgba(255,255,255,0.7);
        }}
        .hemo-watermark-big {{
            font-size: 0.98rem;
            font-weight: 900;
            color: rgba(255,255,255,0.95);
        }}
        </style>

        {watermark_html}
        """,
        unsafe_allow_html=True,
    )


def file_to_b64(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


# ---------------------------
# Data loading
# ---------------------------
@st.cache_data(show_spinner=False)
def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    data = np.load(path, allow_pickle=False)
    ids = data["ids"].astype(np.int64)

    if "emb" in data:
        emb = data["emb"].astype(np.float32)
    elif "emb_all" in data:
        emb = data["emb_all"].astype(np.float32)
    else:
        raise KeyError("NPZ missing embedding array: expected 'emb' or 'emb_all'.")

    # robust meta parsing
    raw = data["meta"]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    meta = json.loads(raw)

    return ids, emb, meta


@st.cache_data(show_spinner=False)
def load_index(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_meshes(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected meshes JSON to be an array.")
    return data


# ---------------------------
# Profile persistence
# ---------------------------
def sanitize_user_id(user_id: str) -> str:
    keep = []
    for ch in (user_id or "").strip():
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)[:80]
    return out or "local_test_user"


def get_profile_path(cfg: Config, user_id: str) -> Path:
    uid = sanitize_user_id(user_id)
    profiles_dir = Path(cfg.profiles_dir)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    return profiles_dir / f"{uid}.json"


def default_profile(user_id: str) -> Dict[str, Any]:
    return {"user_id": user_id, "watched": [], "unfinished": []}


def load_user_profile(cfg: Config, user_id: str) -> Tuple[Dict[str, Any], Path]:
    preferred = get_profile_path(cfg, user_id)

    if preferred.exists():
        with open(preferred, "r", encoding="utf-8") as f:
            prof = json.load(f)
        if not isinstance(prof, dict):
            prof = default_profile(user_id)
        prof.setdefault("user_id", user_id)
        prof.setdefault("watched", [])
        prof.setdefault("unfinished", [])
        return prof, preferred

    legacy = Path(cfg.legacy_profile_json)
    if legacy.exists():
        with open(legacy, "r", encoding="utf-8") as f:
            prof = json.load(f)
        if not isinstance(prof, dict):
            prof = default_profile(user_id)
        prof.setdefault("user_id", user_id)
        prof.setdefault("watched", [])
        prof.setdefault("unfinished", [])
        prof["user_id"] = user_id
        return prof, preferred

    return default_profile(user_id), preferred


def save_user_profile(path: Path, profile: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)


def profile_to_internal(profile: Dict[str, Any]) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    watched_ratings: List[Tuple[int, float]] = []
    unfinished_list: List[Tuple[int, float]] = []

    for x in profile.get("watched", []) or []:
        try:
            watched_ratings.append((int(x["id"]), float(x.get("rating", 4.0))))
        except Exception:
            continue

    for x in profile.get("unfinished", []) or []:
        try:
            unfinished_list.append((int(x["id"]), float(x.get("progress", 0.35))))
        except Exception:
            continue

    return watched_ratings, unfinished_list


def list_friend_user_ids(cfg: Config, current_user_id: str) -> List[str]:
    pdir = Path(cfg.profiles_dir)
    if not pdir.exists():
        return []
    cur = sanitize_user_id(current_user_id)
    out = []
    for p in sorted(pdir.glob("*.json")):
        if p.stem and p.stem != cur:
            out.append(p.stem)
    return out


# ---------------------------
# Utility / scoring
# ---------------------------
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_gamification_fields(profile: Dict[str, Any]) -> None:
    profile.setdefault("points", 0)
    profile.setdefault("trophies", [])
    profile.setdefault("points_log", [])
    for w in profile.get("watched", []) or []:
        if isinstance(w, dict):
            w.setdefault("completed_at", None)


def watched_ids_set(profile: Dict[str, Any]) -> Set[int]:
    out = set()
    for w in profile.get("watched", []) or []:
        if isinstance(w, dict) and "id" in w:
            out.add(int(w["id"]))
    return out


def promoted_movie_ids(items: List[Dict[str, Any]], top_pct: float = 0.10) -> Set[int]:
    years = [int(it.get("year") or 0) for it in items if str(it.get("year") or "").isdigit()]
    max_year = max(years) if years else 0

    pops = np.array([float(it.get("popularity") or 0.0) for it in items], dtype=np.float32)
    thresh = float(np.quantile(pops, 1.0 - top_pct)) if pops.size else 0.0

    out = set()
    for it in items:
        mid = int(it["id"])
        try:
            y = int(it.get("year") or 0)
        except Exception:
            y = 0
        p = float(it.get("popularity") or 0.0)
        if (max_year and y >= max_year - 1) or (p >= thresh):
            out.add(mid)
    return out


def award_points(profile: Dict[str, Any], movie_id: int, reason: str, pts: int) -> None:
    ensure_gamification_fields(profile)
    profile["points"] = int(profile.get("points", 0)) + int(pts)
    profile["points_log"].append(
        {"ts": iso_now(), "reason": reason, "points": int(pts), "movie_id": int(movie_id)}
    )


def compute_trophies(points: int, flags: Dict[str, bool]) -> List[str]:
    trophies: List[str] = []
    if points >= 100:
        trophies.append("Bronze Explorer")
    if points >= 300:
        trophies.append("Silver Explorer")
    if points >= 600:
        trophies.append("Gold Explorer")
    if points >= 1000:
        trophies.append("Platinum Explorer")

    if flags.get("promoted_5"):
        trophies.append("Sponsor Supporter")
    return trophies


def update_trophies(profile: Dict[str, Any], flags: Dict[str, bool]) -> None:
    ensure_gamification_fields(profile)
    new_list = compute_trophies(int(profile.get("points", 0)), flags)
    profile["trophies"] = sorted(set(new_list))


def poster_url(poster_path: Optional[str], size: str = POSTER_SIZE) -> Optional[str]:
    if not poster_path:
        return None
    return f"{TMDB_IMG_BASE}{size}{poster_path}"


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def popularity_scores(items: List[Dict[str, Any]]) -> np.ndarray:
    pops = np.array([float(it.get("popularity") or 0.0) for it in items], dtype=np.float32)
    pops = np.log1p(np.maximum(pops, 0.0))
    mx = float(pops.max()) if pops.size else 1.0
    return pops / mx if mx > 0 else pops


def build_user_embedding(
    emb: np.ndarray,
    id_to_row: Dict[str, int],
    ratings: List[Tuple[int, float]],
    mode: str,
    unfinished: List[Tuple[int, float]],
) -> Tuple[np.ndarray, Set[int], Set[int]]:
    seen_completed: Set[int] = {int(mid) for mid, _ in ratings}
    seen_unfinished: Set[int] = {int(mid) for mid, _ in unfinished}

    if ratings:
        weights = np.array([float(r) for _, r in ratings], dtype=np.float32)
        if mode == "centered":
            weights = weights - float(weights.mean())

        acc = np.zeros((emb.shape[1],), dtype=np.float32)
        wsum = 0.0
        for (mid, _), w in zip(ratings, weights):
            row = id_to_row.get(str(int(mid)))
            if row is None:
                continue
            acc += emb[row] * float(w)
            wsum += abs(float(w))

        if wsum > 0.0:
            acc = acc / float(wsum)
        user_vec = acc
    else:
        user_vec = np.zeros((emb.shape[1],), dtype=np.float32)

    for mid, prog in unfinished:
        row = id_to_row.get(str(int(mid)))
        if row is None:
            continue
        user_vec += emb[row] * float(0.25 * max(0.0, min(1.0, prog)))

    user_vec = l2_normalize(user_vec)
    return user_vec, seen_completed, seen_unfinished


def recommend_df(
    emb: np.ndarray,
    items: List[Dict[str, Any]],
    id_to_row: Dict[str, int],
    user_vec: np.ndarray,
    seen_completed: Set[int],
    seen_unfinished: Set[int],
    top_k: int,
    pop_alpha: float,
    allow_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    sims = emb @ user_vec
    pop = popularity_scores(items)
    score = sims * (1.0 - pop_alpha) + pop * pop_alpha

    exclude_ids = set(seen_completed) | set(seen_unfinished)
    exclude_rows = {id_to_row.get(str(mid)) for mid in exclude_ids}
    exclude_rows = {r for r in exclude_rows if r is not None}
    idx = np.array([i for i in range(len(items)) if i not in exclude_rows], dtype=np.int64)

    if allow_mask is not None:
        idx = idx[allow_mask[idx]]
    if idx.size == 0:
        return pd.DataFrame()

    cand = score[idx]
    k = min(int(top_k), int(idx.size))
    top_part = np.argpartition(-cand, kth=k - 1)[:k]
    top_idx = idx[top_part]
    top_idx = top_idx[np.argsort(-score[top_idx])]

    rows = []
    for i in top_idx:
        it = items[int(i)]
        rows.append(
            {
                "id": int(it["id"]),
                "title": it.get("title"),
                "year": it.get("year"),
                "score": float(score[int(i)]),
                "sim": float(sims[int(i)]),
                "pop_norm": float(pop[int(i)]),
                "vote_average": it.get("vote_average"),
                "vote_count": it.get("vote_count"),
                "poster_path": it.get("poster_path"),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------
# Plot
# ---------------------------
@st.cache_data(show_spinner=False)
def pca_3d(emb: np.ndarray) -> np.ndarray:
    return PCA(n_components=3, random_state=42).fit_transform(emb)


def plot_overlay_3d(
    emb: np.ndarray,
    items: List[Dict[str, Any]],
    completed_ids: Set[int],
    unfinished_ids: Set[int],
    rec_ids: Set[int],
) -> go.Figure:
    X3 = pca_3d(emb)

    ids = [int(it["id"]) for it in items]
    completed_mask = np.array([mid in completed_ids for mid in ids], dtype=bool)
    unfinished_mask = np.array([mid in unfinished_ids for mid in ids], dtype=bool)
    rec_mask = np.array([mid in rec_ids for mid in ids], dtype=bool)

    hover = []
    for it in items:
        hover.append(
            f"<b>{it.get('title','')}</b> ({it.get('year','')})<br>"
            f"id: {it.get('id')}<br>"
            f"popularity: {it.get('popularity')}<br>"
            f"vote_average: {it.get('vote_average')} (n={it.get('vote_count')})"
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=X3[:, 0],
            y=X3[:, 1],
            z=X3[:, 2],
            mode="markers",
            name="All movies",
            marker=dict(size=2, color="rgba(220,220,220,0.16)"),
            text=hover,
            hoverinfo="text",
        )
    )

    if completed_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=X3[completed_mask, 0],
                y=X3[completed_mask, 1],
                z=X3[completed_mask, 2],
                mode="markers",
                name="Watched",
                marker=dict(size=6, color="rgba(30,120,255,0.95)"),
                text=np.array(hover, dtype=object)[completed_mask],
                hoverinfo="text",
            )
        )

    if unfinished_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=X3[unfinished_mask, 0],
                y=X3[unfinished_mask, 1],
                z=X3[unfinished_mask, 2],
                mode="markers",
                name="Unfinished",
                marker=dict(size=6, color="rgba(255,180,0,0.95)"),
                text=np.array(hover, dtype=object)[unfinished_mask],
                hoverinfo="text",
            )
        )

    if rec_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=X3[rec_mask, 0],
                y=X3[rec_mask, 1],
                z=X3[rec_mask, 2],
                mode="markers",
                name="Recommended",
                marker=dict(size=7, color="rgba(0,200,80,0.95)"),
                text=np.array(hover, dtype=object)[rec_mask],
                hoverinfo="text",
            )
        )

    fig.update_layout(
        title="Spacial Map",
        height=620,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(itemsizing="constant"),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, backgroundcolor="#0b0f19"),
            yaxis=dict(showgrid=False, zeroline=False, backgroundcolor="#0b0f19"),
            zaxis=dict(showgrid=False, zeroline=False, backgroundcolor="#0b0f19"),
        ),
    )
    return fig


# ---------------------------
# Header
# ---------------------------
def render_header():
    st.markdown(
        """
        <div class="card" style="margin-bottom:16px;">
          <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
              <div style="font-size:0.9rem; color:rgba(255,255,255,0.65);">ENGINE3</div>
              <div style="font-size:2.1rem; font-weight:900;">The recommendation engine</div>
              <div class="muted">Content-based embeddings • cosine similarity • popularity smoothing • mesh search</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# Carousel (auto-scroll)
# ---------------------------
def carousel_html(
    items: List[Dict[str, Any]],
    subtitle_key: str = "sub",
    variant: str = "normal",
    auto_scroll: bool = True,
    interval_ms: int = 30,
) -> str:
    cid = f"car_{random.randint(10**8, 10**9)}"

    if variant == "big":
        card_w, poster_h, title_fs, sub_fs, gap, pad_y = 220, 300, 14, 12, 18, 10
    else:
        card_w, poster_h, title_fs, sub_fs, gap, pad_y = 170, 250, 13, 12, 14, 8

    def render_item(it: Dict[str, Any]) -> str:
        poster = it.get("poster_url")
        title = (it.get("title") or "")[:60]
        year = it.get("year") or ""
        sub = it.get(subtitle_key) or ""
        poster_html = (
            f'<img class="poster-img" src="{poster}" loading="lazy" />'
            if poster
            else '<div class="poster-img poster-fallback">No poster</div>'
        )
        return f"""
        <div class="carousel-item">
          <div class="poster-card">
            {poster_html}
            <div class="poster-title">{title}</div>
            <div class="poster-sub">{year} • {sub}</div>
          </div>
        </div>
        """

    items_html = "\n".join(render_item(it) for it in items)
    doubled_html = items_html + items_html
    snap_css = "none" if auto_scroll else "x mandatory"

    css = f"""
    <style>
      :root {{
        --card: rgba(255,255,255,0.03);
        --border: rgba(255,255,255,0.10);
        --text: rgba(255,255,255,0.92);
        --muted: rgba(255,255,255,0.65);
      }}
      html, body {{
        margin: 0; padding: 0; background: transparent; color: var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
      }}
      #{cid} {{
        display: flex;
        gap: {gap}px;
        overflow-x: auto;
        overflow-y: hidden;
        padding: {pad_y}px 2px 10px 2px;
        -webkit-overflow-scrolling: touch;
        scroll-behavior: auto;
        scroll-snap-type: {snap_css};
        scrollbar-width: none;
      }}
      #{cid}::-webkit-scrollbar {{ display: none; }}

      #{cid} .carousel-item {{
        scroll-snap-align: start;
        flex: 0 0 {card_w}px;
        width: {card_w}px;
      }}
      #{cid} .poster-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 10px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.22);
        box-sizing: border-box;
      }}
      #{cid} .poster-img {{
        width: 100%;
        height: {poster_h}px;
        object-fit: cover;
        border-radius: 14px;
        display: block;
        background: rgba(255,255,255,0.06);
      }}
      #{cid} .poster-fallback {{
        display:flex;align-items:center;justify-content:center;
        color: rgba(255,255,255,0.6);
        font-size: 12px;
      }}
      #{cid} .poster-title {{
        font-size: {title_fs}px;
        font-weight: 780;
        line-height: 1.25em;
        margin-top: 8px;
        max-height: 2.6em;
        overflow: hidden;
      }}
      #{cid} .poster-sub {{
        font-size: {sub_fs}px;
        color: var(--muted);
        margin-top: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}
    </style>
    """

    js = f"""
    <script>
      (function() {{
        const rail = document.getElementById("{cid}");
        if (!rail) return;

        const running = {str(bool(auto_scroll)).lower()};
        if (!running) return;

        let paused = false;
        rail.addEventListener('mouseenter', () => paused = true);
        rail.addEventListener('mouseleave', () => paused = false);

        let userInteracting = false;
        let resumeTimer = null;
        function markInteract() {{
          userInteracting = true;
          if (resumeTimer) clearTimeout(resumeTimer);
          resumeTimer = setTimeout(() => userInteracting = false, 900);
        }}
        rail.addEventListener('wheel', markInteract, {{passive:true}});
        rail.addEventListener('touchstart', markInteract, {{passive:true}});
        rail.addEventListener('touchmove', markInteract, {{passive:true}});
        rail.addEventListener('mousedown', markInteract);

        function loopResetIfNeeded() {{
          const half = rail.scrollWidth / 2;
          if (rail.scrollLeft >= half) rail.scrollLeft = rail.scrollLeft - half;
        }}

        const pxPerFrame = Math.max(0.6, Math.min(3.0, 60 / {int(interval_ms)}));

        function frame() {{
          if (!paused && !userInteracting) {{
            rail.scrollLeft += pxPerFrame;
            loopResetIfNeeded();
          }}
          requestAnimationFrame(frame);
        }}

        rail.scrollLeft = 0;
        requestAnimationFrame(frame);
      }})();
    </script>
    """

    return "\n".join([css, f'<div id="{cid}">', doubled_html, "</div>", js])


def render_carousel(
    title: str,
    items: List[Dict[str, Any]],
    height: int = 360,
    variant: str = "normal",
    auto_scroll: bool = True,
    interval_ms: int = 30,
):
    st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    if not items:
        st.write("None.")
        return
    components.html(
        carousel_html(items, subtitle_key="sub", variant=variant, auto_scroll=auto_scroll, interval_ms=interval_ms),
        height=height,
        scrolling=False,
    )


# ---------------------------
# Search + mesh masking
# ---------------------------
def search_by_mesh_feature(meshes: List[Dict[str, Any]], mesh_field: str, query: str, limit: int = 30) -> List[int]:
    q = (query or "").strip().lower()
    if not q:
        return []
    out: List[int] = []
    for m in meshes:
        mid = m.get("id")
        if mid is None:
            continue
        if mesh_field == "all":
            text = " ".join([(m.get("genre_mesh") or ""), (m.get("people_mesh") or ""), (m.get("theme_mesh") or "")])
        else:
            text = m.get(mesh_field) or ""
        if q in text.lower():
            out.append(int(mid))
            if len(out) >= limit:
                break
    return out


def split_mesh_tokens(mesh_str: str) -> Set[str]:
    if not mesh_str:
        return set()
    return set(t.strip() for t in mesh_str.split() if t.strip())


def build_user_token_set(meshes: List[Dict[str, Any]], watched_ids: Set[int], field: str) -> Set[str]:
    by_id = {int(m["id"]): m for m in meshes if m.get("id") is not None}
    toks = set()
    for mid in watched_ids:
        m = by_id.get(int(mid))
        if not m:
            continue
        toks |= split_mesh_tokens(m.get(field, "") or "")
    return toks


def mesh_mask_for_tokens(meshes: List[Dict[str, Any]], items: List[Dict[str, Any]], field: str, tokens: Set[str]) -> np.ndarray:
    by_id = {int(m["id"]): m for m in meshes if m.get("id") is not None}
    mask = np.zeros((len(items),), dtype=bool)
    if not tokens:
        return mask
    for i, it in enumerate(items):
        mid = int(it["id"])
        m = by_id.get(mid)
        if not m:
            continue
        toks = split_mesh_tokens(m.get(field, "") or "")
        if toks & tokens:
            mask[i] = True
    return mask


# ---------------------------
# Discovery stats (watched only)
# ---------------------------
def movie_mesh_by_id(meshes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for m in meshes:
        mid = m.get("id")
        if mid is None:
            continue
        out[int(mid)] = m
    return out


def discovery_stats_from_watched(
    meshes_by_id: Dict[int, Dict[str, Any]],
    watched_ids: Set[int],
    field: str = "genre_mesh",
    top_n: int = 10,
) -> List[Tuple[str, float, int]]:
    counts: Dict[str, int] = {}
    total = 0

    for mid in watched_ids:
        m = meshes_by_id.get(int(mid))
        if not m:
            continue
        s = (m.get(field) or "").strip()
        if not s:
            continue
        toks = [t.strip() for t in s.split() if t.strip()]
        if not toks:
            continue
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
            total += 1

    if total == 0:
        return []

    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [(tok, (cnt / total) * 100.0, cnt) for tok, cnt in top]


# ---------------------------
# Synthetic profile
# ---------------------------
def synth_profile(user_seed: str, df_items: pd.DataFrame, n_watched: int = 8, n_unfinished: int = 4) -> Tuple[List[int], Dict[int, float], Dict[int, float]]:
    seed = abs(hash(user_seed)) % (2**32)
    rng = random.Random(seed)

    tmp = df_items.copy()
    tmp["pop"] = tmp.get("popularity", 0.0)
    tmp["pop"] = tmp["pop"].fillna(0.0).astype(float)
    tmp = tmp.sort_values("pop", ascending=False).reset_index(drop=True)

    pool = tmp.head(min(600, len(tmp)))
    watched = rng.sample(pool["id"].tolist(), k=min(n_watched, len(pool)))
    remaining = [x for x in pool["id"].tolist() if x not in watched]
    unfinished = rng.sample(remaining, k=min(n_unfinished, len(remaining)))

    ratings = {}
    for mid in watched:
        r = max(0.0, min(5.0, rng.gauss(4.0, 0.6)))
        r = round(r * 2) / 2.0
        ratings[int(mid)] = float(r)

    unfinished_prog = {}
    for mid in unfinished:
        p = max(0.05, min(0.95, rng.random()))
        unfinished_prog[int(mid)] = float(round(p, 2))

    return watched, ratings, unfinished_prog


# ---------------------------
# App
# ---------------------------
def main():
    st.set_page_config(page_title="ENGINE3", layout="wide")
    cfg = Config()

    logo_b64 = file_to_b64(Path(cfg.hemo_logo_path))
    inject_css(logo_b64)
    render_header()

    # ---------- Load assets ----------
    _, emb, meta = load_npz(cfg.embeddings_npz)
    index = load_index(cfg.index_json)
    items: List[Dict[str, Any]] = index["items"]
    id_to_row: Dict[str, int] = index["id_to_row"]
    meshes = load_meshes(cfg.meshes_json)
    meshes_by_id = movie_mesh_by_id(meshes)

    df_items = pd.DataFrame(items)
    df_items["title"] = df_items.get("title", "").fillna("").astype(str)
    df_items["year"] = df_items.get("year", "").fillna("").astype(str)
    df_items["label"] = df_items.apply(lambda r: f'{r["title"]} ({r["year"]}) — {r["id"]}', axis=1)
    item_by_id = {int(it["id"]): it for it in items}

    # ---------- Sidebar UI (NO profile reads here yet) ----------
    with st.sidebar:
        st.markdown("## Profile")
        user_id = st.text_input("User ID", value="local_test_user")
        st.caption("Profile storage: per-user JSON in part2/user_profiles/")

        use_synth = st.toggle("Use synthetic watch history", value=True)
        regen_synth = st.button("Regenerate synthetic profile")

        mode = st.selectbox("Weighting", ["raw", "centered"], index=0)
        pop_alpha = st.slider("Popularity smoothing", 0.0, 0.30, 0.10, 0.01)
        top_k = st.slider("Top K", 5, 50, cfg.top_k_default, 1)

        st.markdown("---")
        st.markdown("## Carousel")
        auto_scroll = st.toggle("Auto-scroll carousels", value=True)
        scroll_speed = st.slider("Auto-scroll speed", 15, 60, 30, 1, help="Lower is faster.")

        st.markdown("---")
        st.markdown("## Recommendation view")
        rec_view = st.selectbox("Show top recommendations", ["general", "by genre", "by actor", "by theme"], index=0)

        st.markdown("---")
        show_discovery = st.toggle("Show discovery stats", value=True)

        st.markdown("---")
        st.markdown("## Search by feature")
        mesh_choice = st.selectbox("Search mesh", ["all", "genre_mesh", "people_mesh", "theme_mesh"], index=0)
        feature_query = st.text_input("Feature query", placeholder="e.g. Sci-Fi or Tom Cruise or space")

        st.markdown("---")
        auto_run = st.toggle("Auto-run recommendations", value=True)
        run_btn = st.button("Run recommendations", disabled=auto_run)

        st.markdown("---")
        st.markdown("## Rewards")
        complete_pick = st.selectbox("Mark a movie as completed", options=["—"] + df_items["label"].tolist(), index=0)
        give_complete = st.button("Complete + award points")

        st.markdown("---")
        st.markdown("## Friends")
        friend_ids = list_friend_user_ids(cfg, user_id)
        map_target = st.selectbox("View spacial map for", options=["me"] + friend_ids, index=0)

        st.markdown("---")
        st.caption(
            f"Embeddings: {meta.get('embedding_dim')}D • "
            f"TF-IDF vocab: {meta.get('tfidf_vocab_size')} • "
            f"Explained variance (SVD sum): {meta.get('svd_explained_variance_sum'):.3f}"
        )

    # ---------- Load profile (NOW safe to read/write it) ----------
    profile, profile_path = load_user_profile(cfg, user_id)
    ensure_gamification_fields(profile)

    promoted_ids = promoted_movie_ids(items, top_pct=0.10)

    # ---------- Rewards action ----------
    if give_complete and complete_pick != "—":
        row = df_items[df_items["label"] == complete_pick].iloc[0]
        mid = int(row["id"])

        found = False
        for w in profile.get("watched", []) or []:
            if isinstance(w, dict) and int(w.get("id", -1)) == mid:
                w["rating"] = float(w.get("rating", 4.0))
                w["completed_at"] = w.get("completed_at") or iso_now()
                found = True
                break
        if not found:
            profile["watched"].append({"id": mid, "rating": 4.0, "completed_at": iso_now()})

        profile["unfinished"] = [u for u in (profile.get("unfinished") or []) if int(u.get("id", -1)) != mid]

        if mid in promoted_ids:
            award_points(profile, mid, "Completed promoted content (platform paid)", 100)
        else:
            award_points(profile, mid, "Completed movie", 25)

        wids = watched_ids_set(profile)
        promoted_count = sum(1 for x in wids if x in promoted_ids)
        update_trophies(profile, {"promoted_5": promoted_count >= 5})

        save_user_profile(profile_path, profile)
        st.sidebar.success("Completed + points awarded")

    # ---------- Points & Trophies display ----------
    with st.sidebar:
        st.markdown("---")
        st.markdown("## Points & Trophies")
        st.metric("Points", int(profile.get("points", 0)))
        trophies = profile.get("trophies", []) or []
        if trophies:
            for t in trophies:
                st.write(f"🏆 {t}")
        else:
            st.write("No trophies yet.")

    # ---------- Synthetic/manual watch history ----------
    if use_synth:
        if regen_synth or (not profile_path.exists()) or (not profile.get("watched")):
            watched_ids, ratings_by_id, unfinished_by_id = synth_profile(user_seed=user_id, df_items=df_items)
            profile = {
                "user_id": user_id,
                "watched": [{"id": int(mid), "rating": float(ratings_by_id[int(mid)]), "completed_at": iso_now()} for mid in watched_ids],
                "unfinished": [{"id": int(mid), "progress": float(unfinished_by_id[int(mid)])} for mid in unfinished_by_id],
                "points": int(profile.get("points", 0)),
                "trophies": profile.get("trophies", []),
                "points_log": profile.get("points_log", []),
            }
            save_user_profile(profile_path, profile)

        watched_ratings, unfinished_list = profile_to_internal(profile)
    else:
        watched_ratings, unfinished_list = profile_to_internal(profile)

    # ---------- Search results ----------
    if feature_query.strip():
        hit_ids = search_by_mesh_feature(meshes, mesh_choice, feature_query.strip(), limit=40)
        hit_items = []
        for mid in hit_ids:
            it = item_by_id.get(int(mid))
            if not it:
                continue
            hit_items.append(
                {
                    "title": it.get("title"),
                    "year": it.get("year"),
                    "poster_url": poster_url(it.get("poster_path")),
                    "sub": mesh_choice.replace("_mesh", "") if mesh_choice != "all" else "all meshes",
                }
            )

        st.markdown('<div class="card" style="margin-bottom:14px;">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Search results</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="muted">Query: <b>{feature_query}</b> • Scope: <b>{mesh_choice}</b> • Matches: <b>{len(hit_items)}</b></div>',
            unsafe_allow_html=True,
        )
        components.html(
            carousel_html(hit_items, variant="normal", auto_scroll=auto_scroll, interval_ms=int(scroll_speed)),
            height=360,
            scrolling=False,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Auto-run gate ----------
    should_run = auto_run or run_btn
    if not should_run:
        st.info("Enable Auto-run, or click Run recommendations.")
        st.caption(f"Profile file: {profile_path.as_posix()}")
        return

    # ---------- Compute recommendations ----------
    with st.spinner("Computing recommendations..."):
        user_vec, completed_ids, unfinished_ids_set = build_user_embedding(
            emb=emb,
            id_to_row=id_to_row,
            ratings=watched_ratings,
            mode=mode,
            unfinished=unfinished_list,
        )
        if float(np.linalg.norm(user_vec)) == 0.0:
            st.error("User embedding is zero. Add at least 1 watched title (or enable synthetic history).")
            return

        watched_ids_set_local = {mid for mid, _ in watched_ratings}
        allow_mask = None
        if rec_view == "by genre":
            toks = build_user_token_set(meshes, watched_ids_set_local, "genre_mesh")
            allow_mask = mesh_mask_for_tokens(meshes, items, "genre_mesh", toks)
        elif rec_view == "by actor":
            toks = build_user_token_set(meshes, watched_ids_set_local, "people_mesh")
            allow_mask = mesh_mask_for_tokens(meshes, items, "people_mesh", toks)
        elif rec_view == "by theme":
            toks = build_user_token_set(meshes, watched_ids_set_local, "theme_mesh")
            allow_mask = mesh_mask_for_tokens(meshes, items, "theme_mesh", toks)

        recs = recommend_df(
            emb=emb,
            items=items,
            id_to_row=id_to_row,
            user_vec=user_vec,
            seen_completed=completed_ids,
            seen_unfinished=unfinished_ids_set,
            top_k=int(top_k),
            pop_alpha=float(pop_alpha),
            allow_mask=allow_mask,
        )

    # ---------- Build carousels ----------
    watched_sorted = sorted(watched_ratings, key=lambda x: x[1], reverse=True)
    rewatch_ids = [mid for mid, r in watched_sorted if r >= 4.5] or [mid for mid, _ in watched_sorted[:3]]

    def to_carousel_item(mid: int, sub: str) -> Optional[Dict[str, Any]]:
        it = item_by_id.get(int(mid))
        if not it:
            return None
        return {
            "title": it.get("title"),
            "year": it.get("year"),
            "poster_url": poster_url(it.get("poster_path")),
            "sub": sub,
        }

    rec_car = []
    if not recs.empty:
        for _, row in recs.head(20).iterrows():
            rec_car.append(
                {
                    "title": row.get("title"),
                    "year": row.get("year"),
                    "poster_url": poster_url(row.get("poster_path")),
                    "sub": f"score {row.get('score'):.3f}",
                }
            )

    rewatch_car = []
    wr_map = dict(watched_ratings)
    for mid in rewatch_ids[:12]:
        rr = wr_map.get(mid)
        ci = to_carousel_item(mid, f"re-watch • {rr:.1f}" if rr is not None else "re-watch")
        if ci:
            rewatch_car.append(ci)

    unfinished_car = []
    for mid, prog in unfinished_list[:12]:
        ci = to_carousel_item(mid, f"{int(prog * 100)}% watched")
        if ci:
            unfinished_car.append(ci)

    watched_car = []
    for mid, r in watched_sorted[:12]:
        ci = to_carousel_item(mid, f"rating {r:.1f}")
        if ci:
            watched_car.append(ci)

    # ---------- Full-width hero carousel ----------
    render_carousel(
        "Top recommendations",
        rec_car,
        height=500,
        variant="big",
        auto_scroll=auto_scroll,
        interval_ms=int(scroll_speed),
    )

    # ---------- Layout ----------
    colA, colB = st.columns([0.46, 0.54], gap="large")

    with colA:
        render_carousel("Re-watch", rewatch_car, height=360, variant="normal", auto_scroll=auto_scroll, interval_ms=int(scroll_speed))
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        render_carousel("Unfinished (continue watching)", unfinished_car, height=360, variant="normal", auto_scroll=auto_scroll, interval_ms=int(scroll_speed))
        st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
        render_carousel("Watched history", watched_car, height=360, variant="normal", auto_scroll=auto_scroll, interval_ms=int(scroll_speed))
        st.caption(f"Profile file: {profile_path.as_posix()}")

    with colB:
        # choose map target (me or friend)
        if map_target == "me":
            map_user_label = "Me"
            map_completed_ids = completed_ids
            map_unfinished_ids = unfinished_ids_set
            map_rec_ids = set(recs["id"].astype(int).tolist()) if not recs.empty else set()
        else:
            friend_user_id = map_target
            friend_profile, _ = load_user_profile(cfg, friend_user_id)
            ensure_gamification_fields(friend_profile)
            friend_watched_ratings, friend_unfinished_list = profile_to_internal(friend_profile)

            friend_vec, friend_completed_ids, friend_unfinished_ids = build_user_embedding(
                emb=emb,
                id_to_row=id_to_row,
                ratings=friend_watched_ratings,
                mode=mode,
                unfinished=friend_unfinished_list,
            )

            friend_allow_mask = None
            friend_watched_ids_set = {mid for mid, _ in friend_watched_ratings}
            if rec_view == "by genre":
                toks = build_user_token_set(meshes, friend_watched_ids_set, "genre_mesh")
                friend_allow_mask = mesh_mask_for_tokens(meshes, items, "genre_mesh", toks)
            elif rec_view == "by actor":
                toks = build_user_token_set(meshes, friend_watched_ids_set, "people_mesh")
                friend_allow_mask = mesh_mask_for_tokens(meshes, items, "people_mesh", toks)
            elif rec_view == "by theme":
                toks = build_user_token_set(meshes, friend_watched_ids_set, "theme_mesh")
                friend_allow_mask = mesh_mask_for_tokens(meshes, items, "theme_mesh", toks)

            friend_recs = recommend_df(
                emb=emb,
                items=items,
                id_to_row=id_to_row,
                user_vec=friend_vec,
                seen_completed=friend_completed_ids,
                seen_unfinished=friend_unfinished_ids,
                top_k=int(top_k),
                pop_alpha=float(pop_alpha),
                allow_mask=friend_allow_mask,
            )

            map_user_label = friend_user_id
            map_completed_ids = friend_completed_ids
            map_unfinished_ids = friend_unfinished_ids
            map_rec_ids = set(friend_recs["id"].astype(int).tolist()) if not friend_recs.empty else set()

        st.markdown(f'<div class="card-title">Movieverse ({map_user_label})</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="muted">Watched (blue) • Unfinished (orange) • Recommended (green) • Background (gray)</div>',
            unsafe_allow_html=True,
        )

        fig = plot_overlay_3d(
            emb=emb,
            items=items,
            completed_ids=set(map_completed_ids),
            unfinished_ids=set(map_unfinished_ids),
            rec_ids=set(map_rec_ids),
        )
        st.plotly_chart(fig, use_container_width=True)

        # CSV under map
        if not recs.empty and map_target == "me":
            st.download_button(
                "Download recommendations (CSV)",
                recs.to_csv(index=False).encode("utf-8"),
                file_name="recommendations.csv",
                mime="text/csv",
            )

        # Discovery stats UNDER MAP
        if show_discovery:
            st.markdown('<div class="card-title" style="margin-top:14px;">Discovery stats</div>', unsafe_allow_html=True)
            stats = discovery_stats_from_watched(
                meshes_by_id=meshes_by_id,
                watched_ids=set(map_completed_ids),
                field="genre_mesh",
                top_n=10,
            )
            if not stats:
                st.write("No watched genre data yet.")
            else:
                for tok, pct, _cnt in stats:
                    st.markdown(
                        f"""
                        <div style="display:flex; justify-content:space-between; gap:10px; margin: 6px 0;">
                          <div style="font-weight:700;">{tok}</div>
                          <div style="color:rgba(255,255,255,0.75);">{pct:.0f}%</div>
                        </div>
                        <div style="height:6px; background:rgba(255,255,255,0.08); border-radius:999px; overflow:hidden; margin-bottom:10px;">
                          <div style="height:6px; width:{min(100.0, pct):.2f}%; background:rgba(124,58,237,0.9);"></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Details UNDER DISCOVERY
        if not recs.empty and map_target == "me":
            st.markdown('<div class="card-title" style="margin-top:14px;">Details</div>', unsafe_allow_html=True)
            show = recs[["title", "year", "score", "sim", "pop_norm", "vote_average", "vote_count"]].copy()
            st.dataframe(
                show,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "title": st.column_config.TextColumn("title", width="large"),
                    "year": st.column_config.TextColumn("year", width="small"),
                    "score": st.column_config.NumberColumn("score", format="%.4f", width="small"),
                    "sim": st.column_config.NumberColumn("sim", format="%.4f", width="small"),
                    "pop_norm": st.column_config.NumberColumn("pop_norm", format="%.4f", width="small"),
                    "vote_average": st.column_config.NumberColumn("vote_average", format="%.2f", width="small"),
                    "vote_count": st.column_config.NumberColumn("vote_count", format="%d", width="small"),
                },
            )


if __name__ == "__main__":
    main()
