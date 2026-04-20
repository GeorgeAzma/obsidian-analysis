import os
import json
import pickle
import webbrowser
from string import Template
from pathlib import Path

import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
from pygments.formatters import HtmlFormatter
import umap

from config import image_cache_file, image_path, note_cache_file, vault_path
from src.markdown_preview import build_wikilink_map, render_markdown_preview


def ensure_output_dir():
    os.makedirs("output", exist_ok=True)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def path_uri(path):
    return Path(path).resolve().as_uri()


def color_map(groups):
    unique_groups = list(dict.fromkeys(groups))
    palette_colors = pc.qualitative.Plotly
    color_lookup = {group: palette_colors[i % len(palette_colors)] for i, group in enumerate(unique_groups)}
    return [color_lookup[group] for group in groups]


def compute_umap(embeddings, n_components):
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=n_components,
        metric='cosine',
    )
    return reducer.fit_transform(embeddings)


def write_html(fig, output_path, preview_kind=None):
    fig.write_html(output_path, config={"displayModeBar": False, "displaylogo": False})
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    pygments_css = HtmlFormatter(style='github-dark').get_style_defs('.codehilite')

    full_screen_style = '''<style>
html, body { margin: 0; background: transparent; color: #d7dee8; overflow: hidden; }
body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; }
.plotly-graph-div { width: 100vw; height: 100vh; }
#note-hover-preview { position: absolute; top: 16px; right: 16px; z-index: 1000; display: none; pointer-events: none; background: rgba(15, 17, 20, 0.58); color: #dbe2ea; border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 12px; padding: 14px 16px; max-width: 620px; max-height: 680px; overflow: hidden; box-shadow: 0 16px 48px rgba(0, 0, 0, 0.30); backdrop-filter: blur(12px); }
#note-hover-preview .markdown-preview { color: #dbe2ea; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; font-size: 14px; line-height: 1.55; }
#note-hover-preview .markdown-preview * { box-sizing: border-box; }
#note-hover-preview .markdown-preview h1, #note-hover-preview .markdown-preview h2, #note-hover-preview .markdown-preview h3, #note-hover-preview .markdown-preview h4, #note-hover-preview .markdown-preview h5, #note-hover-preview .markdown-preview h6 { color: #f1f5f9; margin: 0 0 6px; line-height: 1.2; font-weight: 600; }
#note-hover-preview .markdown-preview p { margin: 0 0 10px; }
#note-hover-preview .markdown-preview ul, #note-hover-preview .markdown-preview ol { margin: 0 0 10px 20px; padding: 0; }
#note-hover-preview .markdown-preview li { margin: 0 0 4px; }
#note-hover-preview .markdown-preview code { font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; background: rgba(148, 163, 184, 0.16); padding: 1px 5px; border-radius: 4px; font-size: 0.95em; }
#note-hover-preview .markdown-preview pre { background: rgba(2, 6, 23, 0.92); color: #e2e8f0; padding: 10px 12px; border-radius: 10px; overflow: hidden; margin: 0 0 10px; white-space: pre-wrap; tab-size: 4; font-family: ui-monospace, SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace; }
#note-hover-preview .markdown-preview pre code { background: transparent; padding: 0; white-space: inherit; }
#note-hover-preview .markdown-preview .codehilite { margin: 0 0 10px; }
#note-hover-preview .markdown-preview .codehilite pre { margin: 0; background: rgba(2, 6, 23, 0.92); color: #e2e8f0; }
#note-hover-preview .markdown-preview .codehilite .hll { background-color: rgba(148, 163, 184, 0.14); }
#note-hover-preview .markdown-preview .codehilite .c { color: #64748b; font-style: italic; }
#note-hover-preview .markdown-preview .codehilite .err { color: #fca5a5; background-color: rgba(127, 29, 29, 0.35); }
#note-hover-preview .markdown-preview .codehilite .k { color: #c084fc; }
#note-hover-preview .markdown-preview .codehilite .o { color: #f97316; }
#note-hover-preview .markdown-preview .codehilite .ch, #note-hover-preview .markdown-preview .codehilite .cm, #note-hover-preview .markdown-preview .codehilite .cp, #note-hover-preview .markdown-preview .codehilite .cpf, #note-hover-preview .markdown-preview .codehilite .c1 { color: #64748b; font-style: italic; }
#note-hover-preview .markdown-preview .codehilite .cs { color: #94a3b8; font-style: italic; }
#note-hover-preview .markdown-preview .codehilite .gd { color: #fca5a5; background-color: rgba(127, 29, 29, 0.25); }
#note-hover-preview .markdown-preview .codehilite .ge { font-style: italic; }
#note-hover-preview .markdown-preview .codehilite .gh { color: #f8fafc; font-weight: 700; }
#note-hover-preview .markdown-preview .codehilite .gi { color: #86efac; background-color: rgba(20, 83, 45, 0.25); }
#note-hover-preview .markdown-preview .codehilite .go { color: #94a3b8; }
#note-hover-preview .markdown-preview .codehilite .gp { color: #cbd5e1; }
#note-hover-preview .markdown-preview .codehilite .gr { color: #fca5a5; }
#note-hover-preview .markdown-preview .codehilite .gs { font-weight: 700; }
#note-hover-preview .markdown-preview .codehilite .gu { color: #e2e8f0; font-weight: 700; }
#note-hover-preview .markdown-preview .codehilite .gt { color: #fda4af; }
#note-hover-preview .markdown-preview .codehilite .kc, #note-hover-preview .markdown-preview .codehilite .kd, #note-hover-preview .markdown-preview .codehilite .kn, #note-hover-preview .markdown-preview .codehilite .kp, #note-hover-preview .markdown-preview .codehilite .kr { color: #c084fc; }
#note-hover-preview .markdown-preview .codehilite .kt { color: #67e8f9; }
#note-hover-preview .markdown-preview .codehilite .m, #note-hover-preview .markdown-preview .codehilite .mb, #note-hover-preview .markdown-preview .codehilite .mf, #note-hover-preview .markdown-preview .codehilite .mh, #note-hover-preview .markdown-preview .codehilite .mi, #note-hover-preview .markdown-preview .codehilite .mo, #note-hover-preview .markdown-preview .codehilite .il { color: #fbbf24; }
#note-hover-preview .markdown-preview .codehilite .s, #note-hover-preview .markdown-preview .codehilite .sa, #note-hover-preview .markdown-preview .codehilite .sb, #note-hover-preview .markdown-preview .codehilite .sc, #note-hover-preview .markdown-preview .codehilite .dl, #note-hover-preview .markdown-preview .codehilite .sd, #note-hover-preview .markdown-preview .codehilite .s2, #note-hover-preview .markdown-preview .codehilite .se, #note-hover-preview .markdown-preview .codehilite .sh, #note-hover-preview .markdown-preview .codehilite .si, #note-hover-preview .markdown-preview .codehilite .sx, #note-hover-preview .markdown-preview .codehilite .sr, #note-hover-preview .markdown-preview .codehilite .s1 { color: #86efac; }
#note-hover-preview .markdown-preview .codehilite .na, #note-hover-preview .markdown-preview .codehilite .nb, #note-hover-preview .markdown-preview .codehilite .nc, #note-hover-preview .markdown-preview .codehilite .nd, #note-hover-preview .markdown-preview .codehilite .ni, #note-hover-preview .markdown-preview .codehilite .ne, #note-hover-preview .markdown-preview .codehilite .nf, #note-hover-preview .markdown-preview .codehilite .fm { color: #7dd3fc; }
#note-hover-preview .markdown-preview .codehilite .nl, #note-hover-preview .markdown-preview .codehilite .nn, #note-hover-preview .markdown-preview .codehilite .nx, #note-hover-preview .markdown-preview .codehilite .py { color: #e2e8f0; }
#note-hover-preview .markdown-preview .codehilite .nt { color: #38bdf8; }
#note-hover-preview .markdown-preview .codehilite .nv, #note-hover-preview .markdown-preview .codehilite .vc, #note-hover-preview .markdown-preview .codehilite .vg, #note-hover-preview .markdown-preview .codehilite .vi, #note-hover-preview .markdown-preview .codehilite .vm { color: #f9a8d4; }
#note-hover-preview .markdown-preview .codehilite .ow { color: #f97316; }
#note-hover-preview .markdown-preview .codehilite .w { color: #94a3b8; }
#note-hover-preview .markdown-preview .codehilite .bp { color: #93c5fd; }
#note-hover-preview .markdown-preview .codehilite .cunicode, #note-hover-preview .markdown-preview .codehilite .language-plaintext { color: #e2e8f0; }
#note-hover-preview .markdown-preview blockquote { margin: 0 0 10px; padding-left: 12px; border-left: 3px solid rgba(148, 163, 184, 0.4); color: #cbd5e1; }
#note-hover-preview .markdown-preview a,
#note-hover-preview .markdown-preview a:visited,
#note-hover-preview .markdown-preview a.wikilink,
#note-hover-preview .markdown-preview a.wikilink:visited { color: #a78bfa !important; text-decoration: none; }
#note-hover-preview .markdown-preview a.wikilink { text-decoration: underline; text-underline-offset: 2px; }
#note-hover-preview .markdown-preview .callout { margin: 0 0 10px; padding: 10px 12px 10px 16px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.10); background: rgba(30, 41, 59, 0.38); border-left-width: 4px; border-left-style: solid; border-left-color: var(--callout-accent, #60a5fa); }
#note-hover-preview .markdown-preview .callout-title { font-weight: 700; color: var(--callout-accent, #60a5fa); margin: 0 0 6px; }
#note-hover-preview .markdown-preview .callout-content > :last-child { margin-bottom: 0; }
#note-hover-preview .markdown-preview .callout-note { border-left: 3px solid #a78bfa; }
#note-hover-preview .markdown-preview .callout-info { border-left: 3px solid #60a5fa; }
#note-hover-preview .markdown-preview .callout-tip { border-left: 3px solid #34d399; }
#note-hover-preview .markdown-preview .callout-important { border-left: 3px solid #f472b6; }
#note-hover-preview .markdown-preview .callout-success { border-left: 3px solid #22c55e; }
#note-hover-preview .markdown-preview .callout-question { border-left: 3px solid #38bdf8; }
#note-hover-preview .markdown-preview .callout-warning { border-left: 3px solid #f59e0b; }
#note-hover-preview .markdown-preview .callout-danger { border-left: 3px solid #ef4444; }
#note-hover-preview .markdown-preview .callout-example { border-left: 3px solid #c084fc; }
#note-hover-preview .markdown-preview .callout-quote { border-left: 3px solid #94a3b8; }
#note-hover-preview .markdown-preview img { display: none; }
#note-hover-preview .markdown-preview table { border-collapse: collapse; margin: 0 0 10px; width: 100%; }
#note-hover-preview .markdown-preview th, #note-hover-preview .markdown-preview td { border: 1px solid rgba(148, 163, 184, 0.2); padding: 4px 8px; }
#note-hover-preview .markdown-preview hr { border: none; border-top: 1px solid rgba(148, 163, 184, 0.2); margin: 10px 0; }
#image-hover-preview { position: absolute; top: 16px; right: 16px; z-index: 1000; display: none; pointer-events: none; background: rgba(15, 17, 20, 0.58); border: 1px solid rgba(255, 255, 255, 0.10); border-radius: 12px; padding: 10px; max-width: 520px; max-height: 540px; overflow: hidden; box-shadow: 0 16px 48px rgba(0, 0, 0, 0.30); }
#image-hover-preview img { max-width: 500px; max-height: 500px; display: block; border-radius: 8px; }
''' + pygments_css + '''
</style>
'''

    mathjax_block = '''<script>
window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
    },
    svg: { fontCache: 'global' }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
'''

    note_hover_script = '''<script>
document.addEventListener('DOMContentLoaded', function() {
    var plot = document.querySelector('.plotly-graph-div');
    if (!plot) return;
    var preview = document.createElement('div');
    preview.id = 'note-hover-preview';
    preview.innerHTML = '<div id="note-hover-preview-content"></div>';
    document.body.appendChild(preview);

    function placePreview(clientX) {
        var showOnLeft = typeof clientX === 'number' && clientX > window.innerWidth / 2;
        if (showOnLeft) {
            preview.style.left = '16px';
            preview.style.right = 'auto';
        } else {
            preview.style.left = 'auto';
            preview.style.right = '16px';
        }
    }

    plot.on('plotly_hover', function(event) {
        var pt = event.points && event.points[0];
        if (!pt || !pt.customdata || !pt.customdata[1]) return;
        var previewContent = document.getElementById('note-hover-preview-content');
        var nativeEvent = event.event || {};
        var touchPoint = nativeEvent.touches && nativeEvent.touches[0] ? nativeEvent.touches[0] : (nativeEvent.changedTouches && nativeEvent.changedTouches[0] ? nativeEvent.changedTouches[0] : null);
        var clientX = typeof nativeEvent.clientX === 'number' ? nativeEvent.clientX : (touchPoint ? touchPoint.clientX : null);
        placePreview(clientX);
        previewContent.innerHTML = pt.customdata[1];
        preview.style.display = 'block';
        if (window.MathJax && MathJax.typesetPromise) {
            MathJax.typesetPromise([previewContent]).catch(function() {});
        }
    });

    plot.on('plotly_unhover', function() {
        preview.style.display = 'none';
    });
});
</script>'''

    image_hover_script = '''<script>
document.addEventListener('DOMContentLoaded', function() {
    var plot = document.querySelector('.plotly-graph-div');
    if (!plot) return;
    var preview = document.createElement('div');
    preview.id = 'image-hover-preview';
    preview.innerHTML = '<img id="image-hover-preview-img" src="" alt="Preview">';
    document.body.appendChild(preview);

    function placePreview(clientX) {
        var showOnLeft = typeof clientX === 'number' && clientX > window.innerWidth / 2;
        if (showOnLeft) {
            preview.style.left = '16px';
            preview.style.right = 'auto';
        } else {
            preview.style.left = 'auto';
            preview.style.right = '16px';
        }
    }

    plot.on('plotly_hover', function(event) {
        var pt = event.points && event.points[0];
        if (!pt || !pt.customdata || !pt.customdata[1]) return;
        var img = document.getElementById('image-hover-preview-img');
        var nativeEvent = event.event || {};
        var touchPoint = nativeEvent.touches && nativeEvent.touches[0] ? nativeEvent.touches[0] : (nativeEvent.changedTouches && nativeEvent.changedTouches[0] ? nativeEvent.changedTouches[0] : null);
        var clientX = typeof nativeEvent.clientX === 'number' ? nativeEvent.clientX : (touchPoint ? touchPoint.clientX : null);
        placePreview(clientX);
        img.src = pt.customdata[1];
        preview.style.display = 'block';
    });

    plot.on('plotly_unhover', function() {
        preview.style.display = 'none';
    });
});
</script>'''

    html_content = html_content.replace('<head>', '<head>' + full_screen_style + (mathjax_block if preview_kind == 'notes' else ''))
    if preview_kind == 'notes':
        html_content = html_content.replace('</body>', note_hover_script + '\n</body>')
    elif preview_kind == 'image':
        html_content = html_content.replace('</body>', image_hover_script + '\n</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def make_figure(embeddings, titles, groups, previews=None):
    hover_template = '<b>%{text}</b><br>%{customdata[0]}<extra></extra>'
    customdata = list(zip(groups, previews))
    if embeddings.shape[1] == 2:
        fig = go.Figure(data=go.Scatter(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            mode='markers',
            marker={'size': 8, 'opacity': 0.6, 'color': color_map(groups)},
            text=titles,
            customdata=customdata,
            hovertemplate=hover_template,
        ))
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, showline=False, zeroline=False, title=None),
            yaxis=dict(showgrid=False, showticklabels=False, showline=False, zeroline=False, title=None),
            showlegend=False,
            title=None,
            margin=dict(l=0, r=0, t=0, b=0),
            autosize=True,
            hovermode='closest',
            template='plotly_dark',
        )
        return fig

    fig = go.Figure(data=go.Scatter3d(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        z=embeddings[:, 2],
        mode='markers',
        marker={'size': 5, 'opacity': 0.5, 'color': color_map(groups)},
        text=titles,
        customdata=customdata,
        hovertemplate=hover_template,
    ))
    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, showline=False, zeroline=False, title=None),
            yaxis=dict(showticklabels=False, showline=False, zeroline=False, title=None),
            zaxis=dict(showticklabels=False, showline=False, zeroline=False, title=None),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        showlegend=False,
        title=None,
        margin=dict(l=0, r=0, t=0, b=0),
        autosize=True,
        hovermode='closest',
        template='plotly_dark',
    )
    return fig


def write_index_html(output_path, sections):
    plots = {}
    for section in sections:
        plots[section['id']] = {
            'figure': json.loads(section['figure'].to_json()),
            'kind': section['kind'],
        }
    with open('template.html', 'r', encoding='utf-8') as f:
        template = Template(f.read())
    html_content = template.substitute(
        pygments_css=HtmlFormatter(style='github-dark').get_style_defs('.codehilite'),
        plots_json=json.dumps(plots, separators=(',', ':')),
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def build_groups(items, base_path):
    groups = []
    for item in items:
        path = item.get('path', '')
        if path:
            folder = os.path.relpath(os.path.dirname(path), base_path)
            if folder == '.':
                folder = 'root'
        else:
            folder = 'unknown'
        groups.append(folder)
    return groups


def build_image_uris(items):
    uris = []
    for item in items:
        path = item.get('path')
        if not path or not os.path.exists(path):
            uris.append('')
            continue
        uris.append(path_uri(path))
    return uris


def build_note_previews(items, max_length=1600):
    wikilinks = build_wikilink_map(items)
    previews = []
    for item in items:
        markdown_text = (item.get('text', '') or '').strip()
        markdown_text = markdown_text.replace('\r', '')
        if len(markdown_text) > max_length:
            markdown_text = markdown_text[:max_length].rsplit(' ', 1)[0] + '...'
        previews.append(render_markdown_preview(markdown_text, wikilinks))
    return previews


def process_section(cache_path, base_path, prefix, is_image=False):
    if not os.path.exists(cache_path):
        print(f"Skipping {prefix} plot generation because pickle file does not exist: {cache_path}")
        return

    items = load_pickle(cache_path)
    if not items:
        print(f"Skipping {prefix} plot generation because pickle file is empty: {cache_path}")
        return

    titles = [item['title'] for item in items]
    embeddings = np.array([item['embedding'] for item in items])
    groups = build_groups(items, base_path)
    previews = build_image_uris(items) if is_image else build_note_previews(items)
    kind = 'image' if is_image else 'notes'

    embeddings_2d = compute_umap(embeddings, 2)
    embeddings_3d = compute_umap(embeddings, 3)

    return [
        {
            'id': f'{prefix}-2d',
            'kind': kind,
            'figure': make_figure(embeddings_2d, titles, groups, previews=previews),
        },
        {
            'id': f'{prefix}-3d',
            'kind': kind,
            'figure': make_figure(embeddings_3d, titles, groups, previews=previews),
        },
    ]


def main():
    ensure_output_dir()
    sections = []
    sections.extend(process_section(note_cache_file, vault_path, 'notes', is_image=False) or [])
    sections.extend(process_section(image_cache_file, image_path, 'images', is_image=True) or [])
    index_path = os.path.abspath('output/index.html')
    write_index_html(index_path, sections)
    webbrowser.open(path_uri(index_path))


if __name__ == '__main__':
    main()
