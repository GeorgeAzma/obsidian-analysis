import re
from pathlib import Path
from urllib.parse import quote

import markdown


def normalize_wikilink_target(target):
    return target.strip().lower()


def build_wikilink_map(items):
    wikilinks = {}
    for item in items:
        title = (item.get('title') or '').strip()
        path = item.get('path')
        if not title or not path:
            continue
        wikilinks[normalize_wikilink_target(title)] = Path(path).resolve().as_uri()
    return wikilinks


def replace_wikilinks(markdown_text, wikilinks=None):
    wikilinks = wikilinks or {}

    def repl(match):
        inner = match.group(1).strip()
        if not inner:
            return match.group(0)

        if '|' in inner:
            target, alias = inner.split('|', 1)
        else:
            target, alias = inner, None

        target = target.strip()
        alias = (alias.strip() if alias else target)

        heading = None
        if '#' in target:
            target, heading = target.split('#', 1)
            target = target.strip()
            heading = heading.strip()

        href = wikilinks.get(normalize_wikilink_target(target))
        if href:
            if heading:
                href = f"{href}#{quote(heading)}"
            return f'<a class="wikilink" href="{href}" style="color: #60a5fa !important; text-decoration: none;">{alias}</a>'

        return alias

    return re.sub(r'\[\[([^\]]+)\]\]', repl, markdown_text)


def normalize_math_fractions(markdown_text):
    def repl(match):
        body = match.group(1)

        def frac_repl(frac_match):
            numerator = frac_match.group(1)
            denominator = frac_match.group(2)
            if numerator.startswith('{') and denominator.startswith('{'):
                return frac_match.group(0)
            return f'\\frac{{{numerator}}}{{{denominator}}}'

        body = re.sub(r'\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}', frac_repl, body)
        body = re.sub(r'\\frac\s*([A-Za-z0-9])\s*([A-Za-z0-9])', frac_repl, body)
        return f'${body}$'

    return re.sub(r'\$(.+?)\$', repl, markdown_text, flags=re.DOTALL)


def render_callouts(markdown_text):
    callout_type_map = {
        'note': 'note',
        'abstract': 'note',
        'info': 'info',
        'todo': 'info',
        'tip': 'tip',
        'hint': 'tip',
        'important': 'important',
        'success': 'success',
        'check': 'success',
        'done': 'success',
        'question': 'question',
        'help': 'question',
        'warning': 'warning',
        'caution': 'warning',
        'attention': 'warning',
        'danger': 'danger',
        'error': 'danger',
        'bug': 'danger',
        'example': 'example',
        'quote': 'quote',
    }

    callout_color_map = {
        'note': '#60a5fa',
        'info': '#60a5fa',
        'tip': '#34d399',
        'important': '#f472b6',
        'success': '#22c55e',
        'question': '#38bdf8',
        'warning': '#f59e0b',
        'danger': '#ef4444',
        'example': '#c084fc',
        'quote': '#94a3b8',
    }

    lines = markdown_text.splitlines()
    output_lines = []
    index = 0
    while index < len(lines):
        line = lines[index]
        callout_match = re.match(r'^\s*>\s*\[!([A-Za-z0-9_-]+)\](.*)$', line)
        if not callout_match:
            output_lines.append(line)
            index += 1
            continue

        callout_name = callout_match.group(1).lower()
        callout_title = callout_match.group(2).strip()
        callout_style = callout_type_map.get(callout_name, callout_name)
        callout_color = callout_color_map.get(callout_style, '#60a5fa')

        body_lines = []
        index += 1
        while index < len(lines):
            body_line = lines[index]
            if body_line.startswith('>'):
                body_lines.append(re.sub(r'^\s*>\s?', '', body_line))
                index += 1
                continue
            if body_line.strip() == '':
                body_lines.append('')
                index += 1
                continue
            break

        callout_html = [f'<div class="callout callout-{callout_style}" style="--callout-accent: {callout_color}; border-left: 4px solid {callout_color}; padding: 10px 12px 10px 16px;">']
        callout_html.append(f'<div class="callout-title" style="color: {callout_color};"><strong>{callout_name.capitalize()}</strong>{f" {callout_title}" if callout_title else ""}</div>')
        callout_html.append('<div class="callout-content">')
        if body_lines:
            callout_html.append(render_markdown_html('\n'.join(body_lines)))
        callout_html.append('</div>')
        callout_html.append('</div>')
        output_lines.append('\n'.join(callout_html))

    return '\n'.join(output_lines)


def render_markdown_html(markdown_text):
    return markdown.markdown(
        markdown_text,
        extensions=['extra', 'fenced_code', 'codehilite', 'tables', 'sane_lists'],
        extension_configs={
            'codehilite': {
                'guess_lang': False,
                'noclasses': False,
            },
        },
        output_format='html5',
    )


def force_anchor_color(html_text):
    def repl(match):
        attributes = match.group(1) or ''
        if 'style=' in attributes:
            return '<a' + attributes + '>'
        return '<a' + attributes + ' style="color: #60a5fa !important; text-decoration: none;">'

    return re.sub(r'<a(\b[^>]*)>', repl, html_text)


def render_markdown_preview(markdown_text, wikilinks=None):
    markdown_text = replace_wikilinks(markdown_text, wikilinks)
    markdown_text = normalize_math_fractions(markdown_text)
    markdown_text = render_callouts(markdown_text)
    html_text = render_markdown_html(markdown_text)
    return force_anchor_color(html_text)