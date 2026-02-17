import os
import time
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

TMDB_TOKEN = os.getenv("TMDB_TOKEN")
if not TMDB_TOKEN:
    raise SystemExit("Set TMDB_TOKEN env var (export TMDB_TOKEN='...')")

BASE = "https://api.themoviedb.org/3"
HEADERS = {"Authorization": f"Bearer {TMDB_TOKEN}", "Accept": "application/json"}

TARGET = 2000
OUT_CSV = "tmdb_movies_full.csv"

DISCOVER_VOTE_COUNT_GTE = 200
TOP_CAST_N = 10

# Speed tuning
MAX_WORKERS = 24          # try 16–32
MAX_RPS = 30              # keep under ~40 req/s upper-limit guidance
REQUEST_TIMEOUT = 30

# --- progress bar ---
def print_progress(done, total, bar_len=40):
    total = max(total, 1)
    done = min(done, total)
    filled = int(bar_len * done / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = (done / total) * 100
    print(f"\r[{bar}] {done}/{total} ({pct:5.1f}%)", end="", flush=True)

# --- simple global rate limiter (thread-safe) ---
class RateLimiter:
    def __init__(self, rps: float):
        self.interval = 1.0 / float(rps)
        self.lock = threading.Lock()
        self.next_time = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if now < self.next_time:
                time.sleep(self.next_time - now)
            self.next_time = max(self.next_time + self.interval, time.monotonic())

rate_limiter = RateLimiter(MAX_RPS)

def tmdb_get(session: requests.Session, path: str, params=None, max_retries=6):
    url = f"{BASE}{path}"
    for attempt in range(max_retries):
        rate_limiter.wait()
        r = session.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)

        if r.status_code == 429:
            retry_after = int(r.headers.get("Retry-After", "2"))
            time.sleep(retry_after)
            continue

        if r.status_code in (500, 502, 503, 504):
            time.sleep(2 ** attempt)
            continue

        r.raise_for_status()
        return r.json()

    raise RuntimeError(f"Failed after retries: {url}")

def discover_movie_ids(session: requests.Session, target: int):
    ids = []
    seen = set()
    page = 1

    while len(ids) < target:
        data = tmdb_get(
            session,
            "/discover/movie",
            params={
                "language": "en-US",
                "sort_by": "popularity.desc",
                "include_adult": "false",
                "include_video": "false",
                "vote_count.gte": DISCOVER_VOTE_COUNT_GTE,
                "page": page,
            },
        )

        results = data.get("results") or []
        if not results:
            break

        for m in results:
            mid = m.get("id")
            if not mid or mid in seen:
                continue
            seen.add(mid)
            ids.append(mid)
            if len(ids) >= target:
                break

        page += 1
        total_pages = data.get("total_pages") or page
        if page > total_pages:
            break

    return ids

def join_names(items, key="name"):
    return "|".join([x.get(key, "").strip() for x in items if x.get(key)])

def parse_credits(credits):
    cast = credits.get("cast") or []
    crew = credits.get("crew") or []

    top_cast = [c for c in cast if c.get("name")][:TOP_CAST_N]
    actors = join_names(top_cast, "name")
    directors = join_names([p for p in crew if p.get("job") == "Director"], "name")

    return actors, directors

def fetch_movie_full(session: requests.Session, movie_id: int):
    data = tmdb_get(
        session,
        f"/movie/{movie_id}",
        params={"language": "en-US", "append_to_response": "credits,keywords"},
    )

    credits = data.get("credits") or {}
    keywords_block = data.get("keywords") or {}

    actors, directors = parse_credits(credits)

    genres = join_names(data.get("genres") or [], "name")
    keywords = join_names(keywords_block.get("keywords") or [], "name")

    return {
        "id": data.get("id"),
        "imdb_id": data.get("imdb_id"),
        "title": data.get("title"),
        "year": (data.get("release_date") or "")[:4],
        "genres": genres,
        "keywords": keywords,
        "actors": actors,
        "directors": directors,
        "overview": (data.get("overview") or "").replace("\n", " ").strip(),
        "vote_average": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "popularity": data.get("popularity"),
        "poster_path": data.get("poster_path"),
        "backdrop_path": data.get("backdrop_path"),
    }

def write_csv(rows, path):
    if not rows:
        raise SystemExit("No movies collected.")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)

def main():
    with requests.Session() as session:
        movie_ids = discover_movie_ids(session, TARGET)

        rows = []
        done = 0
        print_progress(done, len(movie_ids))

        # Fetch movie details concurrently
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(fetch_movie_full, session, mid) for mid in movie_ids]
            for fut in as_completed(futures):
                try:
                    rows.append(fut.result())
                except Exception:
                    pass
                done += 1
                print_progress(done, len(movie_ids))

        print()
        write_csv(rows, OUT_CSV)
        print("Saved:", OUT_CSV)
        print("Count:", len(rows))
        print("Sample:", rows[:1])

if __name__ == "__main__":
    main()
