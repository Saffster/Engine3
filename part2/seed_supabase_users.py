import os
import time
import random
from datetime import datetime, timezone
from pathlib import Path

import json
from supabase import create_client
import httpx


# -------------------------
# Config
# -------------------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

MOVIE_INDEX_PATH = Path("movie_index.json")  # adjust if needed

PREFIX = "gpt_user_"          # users will be gpt_user_0001, ...
N_USERS = 50
WATCHED_PER_USER = 18
UNFINISHED_PER_USER = 6

# Batch sizes (Supabase/PostgREST likes smaller chunks)
BATCH_USERS = 200
BATCH_ROWS = 500

# -------------------------
def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def chunked(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def with_retry(fn, *, tries=7, base_sleep=0.35, jitter=0.25):
    """
    Retries common transient network failures (incl. your Errno 35) with exponential backoff.
    """
    for attempt in range(tries):
        try:
            return fn()
        except (httpx.ReadError, httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException) as e:
            if attempt == tries - 1:
                raise
            sleep_s = base_sleep * (2 ** attempt) + random.random() * jitter
            time.sleep(sleep_s)


def sb():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def load_movies():
    data = json.loads(MOVIE_INDEX_PATH.read_text(encoding="utf-8"))
    items = data["items"]
    # movies have "id", "title", "year", "popularity", etc.
    return items


def make_personas(n):
    """
    “GPT-style” personas without calling an LLM: diverse tastes + plausible usernames.
    If you later want real GPT personas, you can swap this function to call an LLM.
    """
    archetypes = [
        ("blockbuster", 0.75, 0.25),  # popular-heavy, newer-heavy
        ("classic",     0.25, 0.80),  # less popular, older-heavy
        ("balanced",    0.50, 0.50),
        ("new_releases",0.70, 0.10),
        ("indie",       0.15, 0.35),
    ]
    out = []
    for i in range(n):
        kind, pop_bias, old_bias = random.choice(archetypes)
        out.append({"kind": kind, "pop_bias": pop_bias, "old_bias": old_bias})
    return out


def weighted_sample_movie_ids(items, k, pop_bias=0.5, old_bias=0.5):
    """
    Weighted sampling using popularity + year.
    pop_bias: 0..1 (1 means pick popular more)
    old_bias: 0..1 (1 means prefer older more)
    """
    # Build weights
    years = [int(it.get("year") or 0) for it in items]
    min_y, max_y = min(years), max(years)
    yr_span = max(1, max_y - min_y)

    pops = [float(it.get("popularity") or 0.0) for it in items]
    max_p = max(pops) if pops else 1.0

    weights = []
    for it, p in zip(items, pops):
        y = int(it.get("year") or 0)
        # normalize
        pop_n = (p / max_p) if max_p else 0.0
        age_n = 1.0 - ((y - min_y) / yr_span)  # 1 = older, 0 = newer
        w = (pop_bias * pop_n) + (old_bias * age_n) + 0.05
        weights.append(max(0.0001, w))

    chosen = set()
    # sample without replacement
    while len(chosen) < min(k, len(items)):
        mid = random.choices(items, weights=weights, k=1)[0]["id"]
        chosen.add(int(mid))
    return list(chosen)


def upsert_profiles(usernames):
    client = sb()
    rows = [{"username": u, "display_name": u, "user_id": u} for u in usernames]

    # Upsert on username (must be UNIQUE in your schema)
    for batch in chunked(rows, BATCH_USERS):
        with_retry(lambda: client.table("profiles_public").upsert(batch, on_conflict="username").execute())


def fetch_profile_ids(usernames):
    client = sb()
    # Pull back ids for the created users
    out = {}
    for batch in chunked(usernames, 200):
        res = with_retry(
            lambda: client.table("profiles_public").select("id,username").in_("username", batch).execute()
        )
        for r in (res.data or []):
            out[r["username"]] = r["id"]
    return out


def bulk_insert(table, rows):
    if not rows:
        return
    client = sb()
    for batch in chunked(rows, BATCH_ROWS):
        with_retry(lambda: client.table(table).insert(batch).execute())


def bulk_delete_by_prefix(prefix):
    """
    Optional cleanup: delete only prefix users + dependents.
    If you don’t need resets, don’t call this.
    """
    client = sb()
    profs = with_retry(
        lambda: client.table("profiles_public").select("id,username").like("username", f"{prefix}%").limit(5000).execute()
    ).data or []
    ids = [p["id"] for p in profs]
    if not ids:
        return 0

    # Delete dependents first
    for batch in chunked(ids, 150):
        with_retry(lambda: client.table("watched").delete().in_("user_id", batch).execute())
        with_retry(lambda: client.table("unfinished").delete().in_("user_id", batch).execute())
        with_retry(lambda: client.table("points_ledger").delete().in_("user_id", batch).execute())
        with_retry(lambda: client.table("user_points_totals").delete().in_("user_id", batch).execute())
        with_retry(lambda: client.table("follows").delete().in_("follower_id", batch).execute())
        with_retry(lambda: client.table("follows").delete().in_("following_id", batch).execute())

    for batch in chunked(ids, 150):
        with_retry(lambda: client.table("profiles_public").delete().in_("id", batch).execute())

    return len(ids)


def seed(reset_first=False):
    items = load_movies()

    usernames = [f"{PREFIX}{i:04d}" for i in range(1, N_USERS + 1)]

    if reset_first:
        n = bulk_delete_by_prefix(PREFIX)
        print(f"Deleted {n} existing users with prefix {PREFIX}")

    # 1) profiles
    upsert_profiles(usernames)
    id_by_username = fetch_profile_ids(usernames)

    # 2) watched + unfinished + points ledger + totals
    personas = make_personas(N_USERS)

    watched_rows = []
    unfinished_rows = []
    ledger_rows = []
    totals_rows = []

    now = iso_now()

    for u, persona in zip(usernames, personas):
        uid = id_by_username[u]

        watched_ids = weighted_sample_movie_ids(
            items, WATCHED_PER_USER, pop_bias=persona["pop_bias"], old_bias=persona["old_bias"]
        )
        remaining_pool = [int(it["id"]) for it in items if int(it["id"]) not in set(watched_ids)]
        unfinished_ids = random.sample(remaining_pool, k=min(UNFINISHED_PER_USER, len(remaining_pool)))

        # watched rows
        for mid in watched_ids:
            rating = round(max(2.5, min(5.0, random.gauss(4.1, 0.55))) * 2) / 2.0
            watched_rows.append(
                {"user_id": uid, "movie_id": int(mid), "rating": float(rating), "completed_at": now}
            )
            ledger_rows.append(
                {"user_id": uid, "points": 25, "reason": "Completed movie", "movie_id": int(mid), "created_at": now}
            )

        # unfinished rows
        for mid in unfinished_ids:
            prog = round(random.uniform(0.08, 0.92), 2)
            unfinished_rows.append({"user_id": uid, "movie_id": int(mid), "progress": float(prog)})

        totals_rows.append({"user_id": uid, "total_points": 25 * len(watched_ids), "updated_at": now})

    bulk_insert("watched", watched_rows)
    bulk_insert("unfinished", unfinished_rows)
    bulk_insert("points_ledger", ledger_rows)

    # totals should be UPSERT
    client = sb()
    for batch in chunked(totals_rows, BATCH_ROWS):
        with_retry(lambda: client.table("user_points_totals").upsert(batch, on_conflict="user_id").execute())

    # 3) optional follow graph (everyone follows user_0001)
    anchor = id_by_username.get(f"{PREFIX}0001")
    if anchor:
        follow_rows = []
        for u in usernames[1:]:
            follow_rows.append({"follower_id": id_by_username[u], "following_id": anchor})
        client = sb()
        for batch in chunked(follow_rows, BATCH_ROWS):
            with_retry(lambda: client.table("follows").upsert(batch, on_conflict="follower_id,following_id").execute())

    print(f"Seeded {len(usernames)} users.")
    print(f"watched rows: {len(watched_rows)} | unfinished rows: {len(unfinished_rows)}")


if __name__ == "__main__":
    # set reset_first=True if you want to wipe prefix users before reseeding
    seed(reset_first=False)
