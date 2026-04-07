
import webbrowser
import os

from config import vault_path, cache_file

abs_path_2d = os.path.abspath('output/plot_2d.html')

# if os.path.exists(abs_path_2d):
#     webbrowser.open(f'file://{abs_path_2d}')
#     exit(0)
    
import pickle
import numpy as np
import umap
import plotly.graph_objects as go
import plotly.colors as pc

with open(cache_file, 'rb') as f:
    notes = pickle.load(f)

embeddings = np.array([note['embedding'] for note in notes])
titles = [note['title'] for note in notes]

folders = []
for note in notes:
    path = note.get('path', '')
    if path:
        folder = os.path.relpath(os.path.dirname(path), vault_path)
        if folder == '.':
            folder = 'root'
    else:
        folder = 'unknown'
    folders.append(folder)

unique_folders = list(set(folders))
palette_colors = pc.qualitative.Plotly
colors = [palette_colors[i % len(palette_colors)] for i in range(len(unique_folders))]
folder_to_color = dict(zip(unique_folders, colors))
point_colors = [folder_to_color[f] for f in folders]

print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

# from sklearn.decomposition import PCA
# pca = PCA(n_components=128, random_state=42)
# embeddings= pca.fit_transform(embeddings)

umap_2d = umap.UMAP(
    n_neighbors=15,      # Balance local/global structure
    min_dist=0.1,        # Tightness of clusters
    n_components=2,
    metric='cosine',     # Good for normalized embeddings
)

embeddings_2d = umap_2d.fit_transform(embeddings)

fig_2d = go.Figure(data=go.Scatter(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    mode='markers',
    marker=dict(
        size=8,
        opacity=0.6,
        color=point_colors,
    ),
    text=titles,
    customdata=folders,
    hovertemplate='<b>%{text}</b><br>%{customdata}<extra></extra>'
))

fig_2d.update_layout(
    xaxis=dict(showgrid=False, showticklabels=False, showline=False, zeroline=False, title=None),
    yaxis=dict(showgrid=False, showticklabels=False, showline=False, zeroline=False, title=None),
    showlegend=False,
    title=None,
    margin=dict(l=0, r=0, t=0, b=0),
    autosize=True,
    hovermode='closest',
    template='plotly_dark'
)

umap_3d = umap.UMAP(
    n_neighbors=15,      # Balance local/global structure
    min_dist=0.1,        # Tightness of clusters
    n_components=3,
    metric='cosine',     # Good for normalized embeddings
)

embeddings_3d = umap_3d.fit_transform(embeddings)

fig_3d = go.Figure(data=go.Scatter3d(
    x=embeddings_3d[:, 0],
    y=embeddings_3d[:, 1],
    z=embeddings_3d[:, 2],
    mode='markers',
    marker=dict(
        size=5,
        opacity=0.5,
        color=point_colors,
    ),
    text=titles,
    customdata=folders,
    hovertemplate='<b>%{text}</b><br>%{customdata}<extra></extra>'
))

fig_3d.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False, showline=False, zeroline=False, title=None),
        yaxis=dict(showticklabels=False, showline=False, zeroline=False, title=None),
        zaxis=dict(showticklabels=False, showline=False, zeroline=False, title=None),
        camera=dict(
        eye=dict(x=1.5, y=1.5, z=1.5)
        )
    ),
    showlegend=False,
    title=None,
    margin=dict(l=0, r=0, t=0, b=0),
    autosize=True,
    hovermode='closest',
    template='plotly_dark'
)

fig_2d.write_html(abs_path_2d, config={'displayModeBar': False, 'displaylogo': False})
# Add full screen style
with open(abs_path_2d, 'r', encoding='utf-8') as f:
    html_content = f.read()
full_screen_style = '<style>body { margin: 0; background-color: #111111; color: #ffffff; } .plotly-graph-div { width: 100vw; height: 100vh; }</style>'
html_content = html_content.replace('<head>', '<head>' + full_screen_style)
with open(abs_path_2d, 'w', encoding='utf-8') as f:
    f.write(html_content)

webbrowser.open(f'file://{abs_path_2d}')

abs_path_3d = os.path.abspath('output/plot_3d.html')
fig_3d.write_html(abs_path_3d, config={'displayModeBar': False, 'displaylogo': False})
# Add full screen
with open(abs_path_3d, 'r', encoding='utf-8') as f:
    html_content = f.read()
html_content = html_content.replace('<head>', '<head>' + full_screen_style)
with open(abs_path_3d, 'w', encoding='utf-8') as f:
    f.write(html_content)

webbrowser.open(f'file://{abs_path_3d}')