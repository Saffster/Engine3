# plot_genre_3d_interactive.py
import json
import numpy as np
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

IN_JSON = "tmdb_movies_with_meshes.json"
OUT_HTML = "genre_mesh_3d.html"
MAX_POINTS = 2000

with open(IN_JSON, "r", encoding="utf-8") as f:
    movies = json.load(f)[:MAX_POINTS]

texts = [m.get("genre_mesh", "") for m in movies]

vec = TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+")
X = vec.fit_transform(texts)

svd = TruncatedSVD(n_components=3, random_state=42)
X3 = svd.fit_transform(X)

titles = [m.get("title", "") for m in movies]
genres = [m.get("genres", []) for m in movies]
year = [m.get("year", None) for m in movies]

hover = []
for i, m in enumerate(movies):
    hover.append(
        "<b>{}</b> ({})<br>"
        "genres: {}<br>"
        "genre_mesh: {}<br>"
        "SVD: ({:.3f}, {:.3f}, {:.3f})"
        .format(
            m.get("title",""),
            m.get("year",""),
            ", ".join(m.get("genres", [])[:10]),
            m.get("genre_mesh",""),
            X3[i,0], X3[i,1], X3[i,2],
        )
    )

fig = go.Figure(
    data=[
        go.Scatter3d(
            x=X3[:,0], y=X3[:,1], z=X3[:,2],
            mode="markers",
            marker=dict(size=3),
            text=hover,
            hoverinfo="text",
        )
    ]
)

fig.update_layout(
    title="Genre mesh (TF-IDF → SVD 3D)",
    scene=dict(
        xaxis_title="SVD1",
        yaxis_title="SVD2",
        zaxis_title="SVD3",
    ),
    margin=dict(l=0, r=0, t=50, b=0),
)

fig.write_html(OUT_HTML, include_plotlyjs="cdn")
print("Saved:", OUT_HTML)
