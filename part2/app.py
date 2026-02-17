# part2/app.py
# ENGINE3 — Streamlit app (Supabase-backed, NO synthetic users/history)
#
# Data lives in Supabase:
# - profiles_public (id uuid PK, username unique not null, display_name, user_id text unique nullable)
# - watched (id?, user_id uuid FK, movie_id bigint, rating, completed_at, inserted_at)
# - unfinished (id?, user_id uuid FK, movie_id bigint, progress, inserted_at)
# - points_ledger (id?, user_id uuid FK, points, reason, movie_id, created_at)
# - user_points_totals (user_id uuid PK, total_points, updated_at)
# - follows (follower_id uuid, following_id uuid, PK(follower_id,following_id))

import base64
import json
import time
import httpx
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
from supabase import create_client

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


# ---------------------------
# Supabase client + robust retry wrappers
# ---------------------------

@st.cache_resource(show_spinner=False)
def get_supabase():
    return create_client(
        st.secrets["SUPABASE_URL"],
        st.secrets["SUPABASE_SERVICE_ROLE_KEY"],
    )


def sb_first(rows):
    return rows[0] if isinstance(rows, list) and rows else None


def _sb_is_transient(e: Exception) -> bool:
    msg = str(e).lower()
    transient = [
        "resource temporarily unavailable",  # Errno 35
        "timed out",
        "timeout",
        "connection reset",
        "connection aborted",
        "broken pipe",
        "readerror",
        "writeerror",
        "server disconnected",
        "temporarily unavailable",
        "try again",
        "tls",
    ]
    return any(t in msg for t in transient) or isinstance(
        e,
        (
            httpx.ReadError,
            httpx.ConnectError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.RemoteProtocolError,
        ),
    )


def sb_exec(q, tries: int = 6, base_sleep: float = 0.25):
    last = None
    for i in range(int(tries)):
        try:
            return q.execute()
        except Exception as e:
            last = e
            if (not _sb_is_transient(e)) or i == tries - 1:
                raise
            time.sleep(min(3.0, base_sleep * (2**i) + (i * 0.03)))
    raise last


def sb_data(q, tries: int = 6, base_sleep: float = 0.25):
    resp = sb_exec(q, tries=tries, base_sleep=base_sleep)
    return getattr(resp, "data", None) or []


def sb_select_one(table: str, select: str = "*", **filters):
    sb = get_supabase()
    q = sb.table(table).select(select)
    for k, v in filters.items():
        q = q.eq(k, v)
    rows = sb_data(q.limit(1))
    return sb_first(rows)


# ---------------------------
# Profiles + DB profile reads
# ---------------------------

def sanitize_user_id(user_id: str) -> str:
    keep = []
    for ch in (user_id or "").strip():
        keep.append(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_")
    out = "".join(keep)[:80]
    return out or "local_test_user"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_profile_for_handle(handle: str) -> Dict[str, Any]:
    """
    Ensure row exists in profiles_public for this username (handle).
    Returns: {"profile_uuid": <uuid>, "handle": <username>, "username": <username>}
    """
    sb = get_supabase()
    h = (handle or "").strip() or "local_test_user"

    prof = sb_select_one("profiles_public", "id,username,display_name,user_id", username=h)
    if prof:
        return {"profile_uuid": prof["id"], "handle": prof["username"], "username": prof["username"]}

    sb_exec(
        sb.table("profiles_public").insert(
            {
                "username": h,
                "display_name": h,
                "user_id": h,  # optional text unique; remove if you don't want it
            }
        )
    )
    prof = sb_select_one("profiles_public", "id,username,display_name,user_id", username=h)
    return {"profile_uuid": prof["id"], "handle": prof["username"], "username": prof["username"]}


def sb_get_or_create_profile(handle: str) -> Dict[str, Any]:
    sb = get_supabase()
    p = ensure_profile_for_handle(handle)
    profile_uuid = p["profile_uuid"]

    watched_rows = sb_data(
        sb.table("watched")
        .select("movie_id,rating,completed_at")
        .eq("user_id", profile_uuid)
        .order("completed_at", desc=True)
    )
    watched = [
        {
            "id": int(r["movie_id"]),
            "rating": float(r.get("rating") or 4.0),
            "completed_at": r.get("completed_at"),
        }
        for r in watched_rows
    ]

    unfinished_rows = sb_data(
        sb.table("unfinished")
        .select("movie_id,progress,inserted_at")
        .eq("user_id", profile_uuid)
        .order("inserted_at", desc=True)
    )
    unfinished = [{"id": int(r["movie_id"]), "progress": float(r.get("progress") or 0.35)} for r in unfinished_rows]

    totals = sb_select_one("user_points_totals", "total_points", user_id=profile_uuid)
    if totals and totals.get("total_points") is not None:
        points_total = int(totals["total_points"])
    else:
        ledger = sb_data(sb.table("points_ledger").select("points").eq("user_id", profile_uuid))
        points_total = int(sum(int(x.get("points") or 0) for x in ledger))

    return {
        "profile_uuid": profile_uuid,
        "user_id": p["handle"],
        "username": p.get("username"),
        "watched": watched,
        "unfinished": unfinished,
        "points": points_total,
        "trophies": [],
        "points_log": [],
    }


def sb_list_friend_user_ids(current_handle: str) -> List[str]:
    sb = get_supabase()
    rows = sb_data(sb.table("profiles_public").select("username").limit(80))
    cur = (current_handle or "").strip()
    return [r["username"] for r in rows if r.get("username") and r["username"] != cur]


# ---------------------------
# DB write helpers (real “platform” actions)
# ---------------------------

def sb_add_or_update_unfinished(handle: str, movie_id: int, progress: float) -> None:
    sb = get_supabase()
    p = ensure_profile_for_handle(handle)
    uid = p["profile_uuid"]

    progress = float(max(0.0, min(1.0, progress)))
    existing = sb_data(
        sb.table("unfinished")
        .select("id")
        .eq("user_id", uid)
        .eq("movie_id", int(movie_id))
        .limit(1)
    )
    if existing:
        sb_exec(
            sb.table("unfinished")
            .update({"progress": progress})
            .eq("id", existing[0]["id"])
        )
    else:
        sb_exec(
            sb.table("unfinished").insert(
                {"user_id": uid, "movie_id": int(movie_id), "progress": progress}
            )
        )


def sb_remove_unfinished(handle: str, movie_id: int) -> None:
    sb = get_supabase()
    p = ensure_profile_for_handle(handle)
    uid = p["profile_uuid"]
    sb_exec(sb.table("unfinished").delete().eq("user_id", uid).eq("movie_id", int(movie_id)))


def sb_add_or_update_watched(handle: str, movie_id: int, rating: float) -> None:
    sb = get_supabase()
    now = iso_now()
    p = ensure_profile_for_handle(handle)
    uid = p["profile_uuid"]

    rating = float(max(0.0, min(5.0, rating)))
    existing = sb_data(
        sb.table("watched")
        .select("id")
        .eq("user_id", uid)
        .eq("movie_id", int(movie_id))
        .limit(1)
    )
    if existing:
        sb_exec(
            sb.table("watched")
            .update({"rating": rating, "completed_at": now})
            .eq("id", existing[0]["id"])
        )
    else:
        sb_exec(
            sb.table("watched").insert(
                {"user_id": uid, "movie_id": int(movie_id), "rating": rating, "completed_at": now}
            )
        )


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

def sb_log_event(user_uuid: str, event_type: str, movie_id: Optional[int] = None):
    sb = get_supabase()
    payload = {
        "user_id": user_uuid,
        "event_type": str(event_type),
    }
    if movie_id is not None:
        payload["movie_id"] = int(movie_id)

    sb_exec(sb.table("activity_events").insert(payload))


def sb_mark_completed_and_award(handle: str, movie_id: int, promoted: bool):
    sb = get_supabase()
    now = iso_now()

    p = ensure_profile_for_handle(handle)
    profile_uuid = p["profile_uuid"]

    pts = 100 if promoted else 25
    reason = "Completed promoted content" if promoted else "Completed movie"

    # watched upsert
    existing = sb_data(
        sb.table("watched")
        .select("id")
        .eq("user_id", profile_uuid)
        .eq("movie_id", int(movie_id))
        .limit(1)
    )
    if existing:
        sb_exec(
            sb.table("watched")
            .update({"rating": 4.0, "completed_at": now})
            .eq("id", existing[0]["id"])
        )
    else:
        sb_exec(
            sb.table("watched").insert(
                {"user_id": profile_uuid, "movie_id": int(movie_id), "rating": 4.0, "completed_at": now}
            )
        )

    # remove unfinished
    sb_exec(sb.table("unfinished").delete().eq("user_id", profile_uuid).eq("movie_id", int(movie_id)))

    # points ledger
    sb_exec(
        sb.table("points_ledger").insert(
            {"user_id": profile_uuid, "points": int(pts), "reason": reason, "movie_id": int(movie_id), "created_at": now}
        )
    )

    # totals
    cur = sb_select_one("user_points_totals", "total_points", user_id=profile_uuid)
    if cur and cur.get("total_points") is not None:
        new_total = int(cur["total_points"]) + int(pts)
        sb_exec(
            sb.table("user_points_totals")
            .update({"total_points": new_total, "updated_at": now})
            .eq("user_id", profile_uuid)
        )
    else:
        sb_exec(
            sb.table("user_points_totals").insert(
                {"user_id": profile_uuid, "total_points": int(pts), "updated_at": now}
            )
        )

    # activity event
    sb_log_event(profile_uuid, "watched", movie_id=int(movie_id))


# ---------------------------
# Social: follow + leaderboards
# ---------------------------

def _sb_chunk(seq, n=200):
    seq = list(seq or [])
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def sb_follow(follower_handle: str, following_handle: str) -> None:
    sb = get_supabase()
    if not follower_handle or not following_handle:
        return
    if follower_handle.strip() == following_handle.strip():
        return

    follower = ensure_profile_for_handle(follower_handle.strip())
    following = ensure_profile_for_handle(following_handle.strip())

    sb_exec(
        sb.table("follows").upsert(
            {"follower_id": follower["profile_uuid"], "following_id": following["profile_uuid"]},
            on_conflict="follower_id,following_id",
        )
    )
    sb_log_event(follower["profile_uuid"], "followed", movie_id=None)



def sb_unfollow(follower_handle: str, following_handle: str) -> None:
    sb = get_supabase()
    if not follower_handle or not following_handle:
        return
    follower = ensure_profile_for_handle(follower_handle.strip())
    following = sb_select_one("profiles_public", "id", username=following_handle.strip())
    if not following:
        return
    sb_exec(
        sb.table("follows")
        .delete()
        .eq("follower_id", follower["profile_uuid"])
        .eq("following_id", following["id"])
    )
    sb_log_event(follower["profile_uuid"], "unfollowed", movie_id=None)



def sb_get_global_leaderboard(limit: int = 30) -> List[Dict[str, Any]]:
    sb = get_supabase()
    totals = sb_data(
        sb.table("user_points_totals")
        .select("user_id,total_points")
        .order("total_points", desc=True)
        .limit(int(limit))
    )
    if not totals:
        return []

    ids = [t["user_id"] for t in totals if t.get("user_id")]
    prof_rows = sb_data(sb.table("profiles_public").select("id,username").in_("id", ids))
    id_to_name = {r["id"]: r.get("username") for r in prof_rows}

    out = []
    for t in totals:
        uid = t.get("user_id")
        out.append(
            {"user_id": uid, "username": id_to_name.get(uid, "(unknown)"), "total_points": int(t.get("total_points") or 0)}
        )
    return out


def sb_get_friend_leaderboard(handle: str, limit: int = 30) -> List[Dict[str, Any]]:
    sb = get_supabase()
    me = ensure_profile_for_handle(handle)
    me_id = me["profile_uuid"]

    follows = sb_data(
        sb.table("follows").select("following_id").eq("follower_id", me_id).limit(500)
    )
    following_ids = [r["following_id"] for r in follows if r.get("following_id")]
    if not following_ids:
        return []

    totals: List[Dict[str, Any]] = []
    for chunk in _sb_chunk(following_ids, 150):
        rows = sb_data(sb.table("user_points_totals").select("user_id,total_points").in_("user_id", chunk))
        totals.extend(rows)

    if not totals:
        return []

    totals = sorted(totals, key=lambda x: int(x.get("total_points") or 0), reverse=True)[: int(limit)]
    prof_rows = sb_data(
        sb.table("profiles_public")
        .select("id,username")
        .in_("id", [t["user_id"] for t in totals if t.get("user_id")])
    )
    id_to_name = {r["id"]: r.get("username") for r in prof_rows}

    out = []
    for t in totals:
        uid = t.get("user_id")
        out.append({"user_id": uid, "username": id_to_name.get(uid, "(unknown)"), "total_points": int(t.get("total_points") or 0)})
    return out


# ---------------------------
# Styling
# ---------------------------

def file_to_b64(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode("utf-8")


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
# Utility / scoring
# ---------------------------

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


def movie_mesh_by_id(meshes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for m in meshes:
        mid = m.get("id")
        if mid is None:
            continue
        out[int(mid)] = m
    return out


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
# Mesh search + masking
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
# Discovery stats
# ---------------------------

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

def render_top_genres_chart(stats: List[Tuple[str, float, int]], title: str = "Top genres (watched)"):
    if not stats:
        st.write("No watched genre data yet.")
        return

    labels = [(tok or "").replace("genre:", "").strip() for tok, _pct, _cnt in stats]
    pct = [float(_pct) for _tok, _pct, _cnt in stats]
    cnt = [int(_cnt) for _tok, _pct, _cnt in stats]

    df = pd.DataFrame({
        "Genre": labels,
        "Percent": pct,
        "Count": cnt
    }).sort_values("Percent", ascending=True)

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Percent"],
                y=df["Genre"],
                orientation="h",
                text=[f"{p:.0f}%  ({c})" for p, c in zip(df["Percent"], df["Count"])],
                textposition="outside",
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=max(320, 40 * len(df) + 120),
        margin=dict(l=120, r=40, t=60, b=40),
        xaxis=dict(title="Share of watched (%)", range=[0, max(5.0, max(pct) * 1.15)]),
        yaxis=dict(title=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------
# Header + carousel
# ---------------------------

def render_header():
    st.markdown(
        """
        <div class="card" style="margin-bottom:16px;">
          <div style="display:flex; align-items:center; justify-content:space-between;">
            <div>
              <div style="font-size:0.9rem; color:rgba(255,255,255,0.65);">ENGINE3</div>
              <div style="font-size:2.1rem; font-weight:900;">The recommendation engine</div>
              <div class="muted">Content-based embeddings • cosine similarity • popularity smoothing • mesh search • Supabase social</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def carousel_html(
    items: List[Dict[str, Any]],
    subtitle_key: str = "sub",
    variant: str = "normal",
    auto_scroll: bool = True,
    interval_ms: int = 30,
) -> str:
    import random as _rnd
    cid = f"car_{_rnd.randint(10**8, 10**9)}"

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
# App
# ---------------------------

def main():
    st.set_page_config(page_title="ENGINE3", layout="wide")
    cfg = Config()

    logo_b64 = file_to_b64(Path(cfg.hemo_logo_path))
    inject_css(logo_b64)
    render_header()

    # Assets
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
    promoted_ids = promoted_movie_ids(items, top_pct=0.10)

    tab_explore, tab_social = st.tabs(["Explore", "Social"])

    # Sidebar (shared)
    with st.sidebar:
        st.markdown("## Profile")
        handle = st.text_input("User ID", value="local_test_user")

        st.markdown("---")
        st.markdown("## Add activity")
        add_mode = st.selectbox("Action", ["Add to watched", "Add to unfinished"], index=0)
        pick = st.selectbox("Movie", options=["—"] + df_items["label"].tolist(), index=0, key="add_pick")

        if add_mode == "Add to watched":
            rating = st.slider("Rating", 0.0, 5.0, 4.0, 0.5)
            award = st.toggle("Award points (Complete)", value=True, help="Adds points + removes from unfinished")
            add_btn = st.button("Save watched")
        else:
            prog = st.slider("Progress", 0.0, 1.0, 0.35, 0.05)
            add_btn = st.button("Save unfinished")

        st.markdown("---")
        st.markdown("## Recommendation settings")
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
        st.markdown("## Search by feature")
        mesh_choice = st.selectbox("Search mesh", ["all", "genre_mesh", "people_mesh", "theme_mesh"], index=0)
        feature_query = st.text_input("Feature query", placeholder="e.g. Sci-Fi or Tom Cruise or space")

        st.markdown("---")
        auto_run = st.toggle("Auto-run recommendations", value=True)
        run_btn = st.button("Run recommendations", disabled=auto_run)

        st.markdown("---")
        st.markdown("## Friends")
        friend_ids = sb_list_friend_user_ids(handle)
        map_target = st.selectbox("View spacial map for", options=["me"] + friend_ids, index=0)

        st.markdown("---")
        st.caption(
            f"Embeddings: {meta.get('embedding_dim')}D • "
            f"TF-IDF vocab: {meta.get('tfidf_vocab_size')} • "
            f"Explained variance (SVD sum): {meta.get('svd_explained_variance_sum'):.3f}"
        )

    # Load profile
    profile = sb_get_or_create_profile(handle)

    # Apply "Add activity" action
    if add_btn and pick != "—":
        row = df_items[df_items["label"] == pick].iloc[0]
        mid = int(row["id"])

        if add_mode == "Add to watched":
            if award:
                sb_mark_completed_and_award(handle=handle, movie_id=mid, promoted=(mid in promoted_ids))
                # also ensure watched rating reflects slider
                sb_add_or_update_watched(handle, mid, rating)
            else:
                sb_add_or_update_watched(handle, mid, rating)
                sb_remove_unfinished(handle, mid)
            profile = sb_get_or_create_profile(handle)
            st.sidebar.success("Saved to watched.")
        else:
            sb_add_or_update_unfinished(handle, mid, prog)
            profile = sb_get_or_create_profile(handle)
            st.sidebar.success("Saved to unfinished.")


    with st.sidebar:
        st.markdown("---")
        st.markdown("## Points")
        st.metric("Points", int(profile.get("points", 0)))

    # DB watched/unfinished -> engine inputs
    watched_ratings, unfinished_list = profile_to_internal(profile)

    # -----------------------
    # Explore tab
    # -----------------------
    with tab_explore:
        # Search results (top)
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

        should_run = auto_run or run_btn
        if not should_run:
            st.info("Enable Auto-run, or click Run recommendations.")
            st.caption("Tip: add at least 1 watched title to generate recommendations.")
            return

        with st.spinner("Computing recommendations..."):
            user_vec, completed_ids, unfinished_ids_set = build_user_embedding(
                emb=emb,
                id_to_row=id_to_row,
                ratings=watched_ratings,
                mode=mode,
                unfinished=unfinished_list,
            )
            if float(np.linalg.norm(user_vec)) == 0.0:
                st.error("User embedding is zero. Add at least 1 watched title in the sidebar.")
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

        watched_sorted = sorted(watched_ratings, key=lambda x: x[1], reverse=True)
        rewatch_ids = [mid for mid, r in watched_sorted if r >= 4.5] or [mid for mid, _ in watched_sorted[:3]]

        def to_carousel_item(mid: int, sub: str) -> Optional[Dict[str, Any]]:
            it = item_by_id.get(int(mid))
            if not it:
                return None
            return {"title": it.get("title"), "year": it.get("year"), "poster_url": poster_url(it.get("poster_path")), "sub": sub}

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

        render_carousel(
            "Top recommendations",
            rec_car,
            height=500,
            variant="big",
            auto_scroll=auto_scroll,
            interval_ms=int(scroll_speed),
        )

        colA, colB = st.columns([0.46, 0.54], gap="large")

        with colA:
            render_carousel("Re-watch", rewatch_car, height=360, variant="normal", auto_scroll=auto_scroll, interval_ms=int(scroll_speed))
            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            render_carousel("Unfinished (continue watching)", unfinished_car, height=360, variant="normal", auto_scroll=auto_scroll, interval_ms=int(scroll_speed))
            st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)
            render_carousel("Watched history", watched_car, height=360, variant="normal", auto_scroll=auto_scroll, interval_ms=int(scroll_speed))

        with colB:
            if map_target == "me":
                map_user_label = "Me"
                map_completed_ids = completed_ids
                map_unfinished_ids = unfinished_ids_set
                map_rec_ids = set(recs["id"].astype(int).tolist()) if not recs.empty else set()
                map_recs_df = recs
            else:
                friend_user_id = map_target
                friend_profile = sb_get_or_create_profile(friend_user_id)
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
                map_recs_df = friend_recs

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

            if map_recs_df is not None and (not map_recs_df.empty) and map_target == "me":
                st.download_button(
                    "Download recommendations (CSV)",
                    map_recs_df.to_csv(index=False).encode("utf-8"),
                    file_name="recommendations.csv",
                    mime="text/csv",
                )

            
            #if show_discovery:
                st.markdown(
                    '<div class="card-title" style="margin-top:14px;">Discovery stats</div>',
                    unsafe_allow_html=True,
                )

                stats = discovery_stats_from_watched(
                    meshes_by_id=meshes_by_id,
                    watched_ids=set(map_completed_ids),
                    field="genre_mesh",
                    top_n=10,
                )

                render_top_genres_chart(stats, title="Top genres watched")


    # -----------------------
    # Social tab
    # -----------------------
    with tab_social:
        st.markdown("## Social")

        st.markdown("### Global leaderboard")
        global_lb = sb_get_global_leaderboard(limit=30)
        if not global_lb:
            st.info("No leaderboard data yet.")
        else:
            gdf = pd.DataFrame(global_lb)
            gdf.insert(0, "rank", range(1, len(gdf) + 1))
            st.dataframe(gdf[["rank", "username", "total_points"]], use_container_width=True, hide_index=True)

        st.markdown("### Friend leaderboard")
        friend_lb = sb_get_friend_leaderboard(handle, limit=30)
        if not friend_lb:
            st.info("No followed users yet. Follow someone below.")
        else:
            fdf = pd.DataFrame(friend_lb)
            fdf.insert(0, "rank", range(1, len(fdf) + 1))
            st.dataframe(fdf[["rank", "username", "total_points"]], use_container_width=True, hide_index=True)

        st.markdown("### Follow / unfollow")
        c1, c2, c3 = st.columns([0.55, 0.225, 0.225])
        with c1:
            follow_name = st.text_input("Username", value="", key="follow_name")
        with c2:
            if st.button("Follow", key="follow_btn") and follow_name.strip():
                sb_follow(handle, follow_name.strip())
                st.success(f"Following: {follow_name.strip()}")
        with c3:
            if st.button("Unfollow", key="unfollow_btn") and follow_name.strip():
                sb_unfollow(handle, follow_name.strip())
                st.success(f"Unfollowed: {follow_name.strip()}")

        st.markdown("### Discovery map (select any user)")
        selectable_users = [r["username"] for r in (global_lb or []) if r.get("username")]
        if handle not in selectable_users:
            selectable_users = [handle] + selectable_users

        selected_user = st.selectbox(
            "Select a user to visualize",
            options=selectable_users[:200] if selectable_users else [handle],
            index=0,
            key="social_user_pick",
        )

        sel_profile = sb_get_or_create_profile(selected_user)
        sel_watched_ratings, sel_unfinished_list = profile_to_internal(sel_profile)
        sel_completed_ids = {mid for mid, _ in sel_watched_ratings}
        sel_unfinished_ids = {mid for mid, _ in sel_unfinished_list}

        st.caption(f"User: {selected_user} • Watched: {len(sel_completed_ids)} • Unfinished: {len(sel_unfinished_ids)}")

        fig_social = plot_overlay_3d(
            emb=emb,
            items=items,
            completed_ids=set(sel_completed_ids),
            unfinished_ids=set(sel_unfinished_ids),
            rec_ids=set(),
        )
        st.plotly_chart(fig_social, use_container_width=True)

        st.markdown("### Top genres (watched)")
        stats = discovery_stats_from_watched(
            meshes_by_id=meshes_by_id,
            watched_ids=set(sel_completed_ids),
            field="genre_mesh",
            top_n=10,
)

        render_top_genres_chart(stats)



if __name__ == "__main__":
    main()