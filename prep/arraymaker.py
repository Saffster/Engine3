# csv_to_array.py
# Reads the clean TMDB CSV and outputs a Python array (list[dict]) and a JSON file.

import csv
import json

CSV_PATH = "tmdb_movies_full.csv"
JSON_PATH = "tmdb_movies_full.json"

def to_number(x):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s

def split_pipe(s):
    if s is None:
        return []
    s = str(s).strip()
    if not s:
        return []
    return [p.strip() for p in s.split("|") if p.strip()]

movies_array = []

with open(CSV_PATH, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        movies_array.append({
            "id": int(row["id"]) if row.get("id") else None,
            "imdb_id": row.get("imdb_id") or None,
            "title": row.get("title") or "",
            "year": int(row["year"]) if row.get("year") and row["year"].isdigit() else None,

            "genres": split_pipe(row.get("genres")),
            "keywords": split_pipe(row.get("keywords")),
            "actors": split_pipe(row.get("actors")),
            "directors": split_pipe(row.get("directors")),

            "overview": row.get("overview") or "",

            "vote_average": to_number(row.get("vote_average")),
            "vote_count": to_number(row.get("vote_count")),
            "popularity": to_number(row.get("popularity")),

            "poster_path": row.get("poster_path") or None,
            "backdrop_path": row.get("backdrop_path") or None,
        })

print("Movies in array:", len(movies_array))
print("Sample:", movies_array[:2])

# Optional: write as JSON (valid + clean)
with open(JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(movies_array, f, ensure_ascii=False, indent=2)

print("Wrote JSON:", JSON_PATH)
