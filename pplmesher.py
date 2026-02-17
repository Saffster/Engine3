# plot_people_3d.py
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

IN_JSON = "/mnt/data/tmdb_movies_with_meshes.json"
OUT_PNG = "people_mesh_3d.png"
MAX_POINTS = 2000

with open(IN_JSON, "r", encoding="utf-8") as f:
    movies = json.load(f)

texts = [(m.get("people_mesh") or "") for m in movies][:MAX_POINTS]

vec = TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+")
X = vec.fit_transform(texts)

svd = TruncatedSVD(n_components=3, random_state=42)
X3 = svd.fit_transform(X)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(X3[:, 0], X3[:, 1], X3[:, 2], s=6)
ax.set_title("People mesh (TF-IDF → SVD 3D)")
ax.set_xlabel("SVD1")
ax.set_ylabel("SVD2")
ax.set_zlabel("SVD3")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200)
print("Saved:", OUT_PNG)
