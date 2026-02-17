import json
import os
from pathlib import Path
from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # service role for import only

sb = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Directory containing your profile JSONs
# If this script is in part2/, and JSONs are in part2/user_profiles/
PROFILES_DIR = Path(__file__).parent / "user_profiles"


def parse_ts(x):
    if not x:
        return None
    try:
        return str(x).replace("Z", "+00:00")
    except Exception:
        return None


def upsert_user(username: str) -> str:
    """
    Ensures a row exists in profiles_public and returns the UUID primary key.
    Assumes profiles_public has columns: id (uuid pk), username, display_name
    """
    # Upsert (don’t rely on returned row)
    sb.table("profiles_public").upsert(
        {"username": username, "display_name": username},
        on_conflict="username",
    ).execute()

    # Fetch the uuid pk
    row = (
        sb.table("profiles_public")
        .select("id")
        .eq("username", username)
        .single()
        .execute()
        .data
    )
    return row["id"]


def main():
    if not PROFILES_DIR.exists():
        raise SystemExit(f"Profiles dir not found: {PROFILES_DIR}")

    files = sorted(PROFILES_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON profiles found in: {PROFILES_DIR}")

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            prof = json.load(f)

        # Your JSON uses "user_id" as the handle (e.g., "safwaan")
        username = prof.get("user_id") or path.stem
        user_uuid = upsert_user(username)

        watched = prof.get("watched") or []
        unfinished = prof.get("unfinished") or []
        trophies = prof.get("trophies") or []
        points = int(prof.get("points", 0))
        points_log = prof.get("points_log") or []

        # ---------- watched ----------
        w_rows = []
        for w in watched:
            try:
                mid = int(w["id"])
            except Exception:
                continue
            w_rows.append(
                {
                    "user_id": user_uuid,     # <-- if your FK column is named "id", change this key to "id"
                    "movie_id": mid,
                    "rating": w.get("rating"),
                    "completed_at": parse_ts(w.get("completed_at")),
                }
            )

        if w_rows:
            # watched should have unique(user_id, movie_id)
            sb.table("watched").upsert(w_rows, on_conflict="user_id,movie_id").execute()

        # ---------- unfinished ----------
        u_rows = []
        for u in unfinished:
            try:
                mid = int(u["id"])
            except Exception:
                continue
            u_rows.append(
                {
                    "user_id": user_uuid,     # <-- if your FK column is named "id", change this key to "id"
                    "movie_id": mid,
                    "progress": float(u.get("progress", 0.0)),
                }
            )

        if u_rows:
            # unfinished should have unique(user_id, movie_id)
            sb.table("unfinished").upsert(u_rows, on_conflict="user_id,movie_id").execute()

        # ---------- trophies ----------
        t_rows = []
        for t in trophies:
            if not t:
                continue
            t_rows.append(
                {
                    "user_id": user_uuid,     # <-- if your FK column is named "id", change this key to "id"
                    "trophy": str(t),
                }
            )

        if t_rows:
            # trophies should have primary key (user_id, trophy)
            sb.table("trophies").upsert(t_rows, on_conflict="user_id,trophy").execute()

        # ---------- points ledger ----------
        if points_log:
            l_rows = []
            for x in points_log:
                try:
                    pts = int(x.get("points", 0))
                except Exception:
                    pts = 0
                if pts == 0:
                    continue

                l_rows.append(
                    {
                        "user_id": user_uuid,  # <-- if your FK column is named "id", change this key to "id"
                        "movie_id": x.get("movie_id"),
                        "points": pts,
                        "reason": x.get("reason", "") or "",
                        "created_at": parse_ts(x.get("ts")),
                    }
                )
            if l_rows:
                sb.table("points_ledger").insert(l_rows).execute()
        elif points > 0:
            sb.table("points_ledger").insert(
                [
                    {
                        "user_id": user_uuid,  # <-- if your FK column is named "id", change this key to "id"
                        "movie_id": None,
                        "points": points,
                        "reason": "Imported legacy points",
                    }
                ]
            ).execute()

        print("Imported:", username)


if __name__ == "__main__":
    main()
