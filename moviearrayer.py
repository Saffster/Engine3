# export_movies_array_py.py
import json

IN_JSON = "tmdb_movies_with_meshes.json"
OUT_JS = "movies.array.js"


def main():
    with open(IN_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit("Expected top-level JSON array")

    js = f"// AUTO-GENERATED from {IN_JSON}\nexport default {json.dumps(data, ensure_ascii=False, indent=2)};\n"
    with open(OUT_JS, "w", encoding="utf-8") as f:
        f.write(js)
    print("Wrote:", OUT_JS, "items:", len(data))


if __name__ == "__main__":
    main()
