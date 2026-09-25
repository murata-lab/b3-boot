"""Read-only view of a source file with line numbers, for reading model.py in the lesson."""
import io
import keyword
import token
import tokenize
from html import escape
from pathlib import Path


STYLE = """
<style>
.pt-file{min-width:0;border:1px solid #e2e7ec;border-radius:10px;overflow:hidden;background:#fff;color:#1f2a37;margin:12px 0}
.pt-file header{display:flex;align-items:center;justify-content:space-between;gap:12px;background:#f3f6f9;border-bottom:1px solid #e2e7ec;padding:9px 15px;font-size:13px;white-space:nowrap}
.pt-file header strong{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-weight:600}
.pt-file header span{font-size:12px;color:#6b7785}
.pt-source{overflow-x:auto;padding:10px 0;tab-size:4;font:13px/1.85 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-ligatures:none}
.pt-line{display:flex;min-width:max-content;padding-right:16px}
.pt-lineno{flex:none;width:40px;text-align:right;padding-right:13px;color:#9aa6b2;user-select:none}
.pt-text{white-space:pre}
.pt-kw{color:#853898}.pt-string{color:#157365}.pt-comment{color:#6b7785;font-style:italic}.pt-number{color:#aa5215}
</style>
"""


def source_html(source):
    """Keep the actual source characters, with separate unselectable line numbers."""
    lines = source.splitlines()
    marks = {number: [] for number in range(1, len(lines) + 1)}
    for item in tokenize.generate_tokens(io.StringIO(source).readline):
        kind = ""
        if item.type == token.NAME and keyword.iskeyword(item.string):
            kind = "kw"
        elif item.type == token.STRING:
            kind = "string"
        elif item.type == tokenize.COMMENT:
            kind = "comment"
        elif item.type == token.NUMBER:
            kind = "number"
        if kind:
            for number in range(item.start[0], item.end[0] + 1):
                if number in marks:
                    start = item.start[1] if number == item.start[0] else 0
                    end = item.end[1] if number == item.end[0] else len(lines[number - 1])
                    marks[number].append((start, end, kind))
    rows = []
    for number, line in enumerate(lines, 1):
        cursor, pieces = 0, []
        for start, end, kind in marks[number]:
            pieces.extend([escape(line[cursor:start]), f'<span class="pt-{kind}">{escape(line[start:end])}</span>'])
            cursor = end
        pieces.append(escape(line[cursor:]))
        rows.append(f'<div class="pt-line"><span class="pt-lineno" aria-hidden="true">{number}</span>'
                    f'<span class="pt-text">{"".join(pieces) or " "}</span></div>')
    return "".join(rows)


def file_preview(path):
    path = Path(path)
    body = source_html(path.read_text(encoding="utf-8"))
    return (f'{STYLE}<section class="pt-file" aria-label="{escape(path.name)} の内容">'
            f'<header><strong>{escape(path.name)}</strong><span>読み取り専用</span></header>'
            f'<div class="pt-source" tabindex="0">{body}</div></section>')
