"""Read-only file views and figures for the lesson (no training on import)."""
import hashlib
import io
import json
import keyword
import token
import tokenize
from html import escape
from pathlib import Path


STYLE = """
<style>
.pt-hero{border-top:5px solid #2563a6;background:#f1f6fb;border-radius:12px;padding:28px 32px;color:#233649;line-height:1.75}.pt-hero small{color:#2563a6;font-size:13px;letter-spacing:.13em;font-weight:700}.pt-hero h1{font-size:30px;line-height:1.45;margin:8px 0 12px}.pt-hero p{margin:0}.pt-note{border-left:4px solid #157365;background:#eff8f5;color:#233649;padding:14px 18px;line-height:1.8}.pt-flow{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:22px 0;color:#233649}.pt-flow b{border:1px solid #dbe4ed;border-radius:9px;background:#fff;padding:12px 15px;font-size:14px}.pt-flow span{color:#6f8394}.pt-files{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);gap:16px;align-items:start}.pt-file{min-width:0;border:1px solid #d2dde7;border-radius:12px;overflow:hidden;background:#fcfdff;color:#22364a;margin:12px 0}.pt-file header{display:flex;align-items:center;justify-content:space-between;gap:12px;background:#eef3f8;border-bottom:1px solid #d2dde7;padding:11px 15px;font-size:13px;white-space:nowrap}.pt-file header strong{font-family:ui-monospace,monospace}.pt-file header span{font-size:11px;color:#60768a}.pt-source{overflow-x:auto;padding:12px 0;max-height:610px;overflow-y:auto;tab-size:4;font:12.5px/1.85 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-variant-ligatures:none}.pt-source.short{max-height:none}.pt-line{display:flex;min-width:max-content;padding-right:16px}.pt-line.focus{background:#e5f0fb;box-shadow:inset 3px 0 #2563a6}.pt-lineno{flex:none;width:40px;text-align:right;padding-right:13px;color:#8a9cac;user-select:none}.pt-text{white-space:pre}.pt-kw{color:#853898}.pt-string{color:#157365}.pt-comment{color:#728397;font-style:italic}.pt-number{color:#aa5215}.pt-caption{font-size:13px;color:#52687c;line-height:1.7}.pt-tree{background:#f5f8fc;border:1px solid #dbe4ed;border-radius:12px;padding:20px;color:#233649;font:13px/1.9 ui-monospace,monospace;overflow-x:auto}.pt-tree div{white-space:pre}.pt-tree .active{color:#2563a6;font-weight:700}.pt-figure{border:1px solid #dbe4ed;border-radius:12px;padding:14px;background:white;margin:15px 0;overflow:auto;color:#233649}.pt-figure svg{display:block;width:100%;min-width:570px;font-family:system-ui,sans-serif}.pt-metrics{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin:18px 0}.pt-metric{background:#f1f6fb;padding:18px;border-radius:12px;color:#233649}.pt-metric span{font-size:12px;color:#5c7184;display:block}.pt-metric strong{font-size:27px;line-height:1.7}.pt-digits{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:18px 0}.pt-digits figure{margin:0;background:#f4f7fb;padding:12px;border-radius:10px;text-align:center;color:#233649}.pt-digits img{width:70px;height:70px;image-rendering:pixelated;border-radius:5px}.pt-digits figcaption{font-size:12px;line-height:1.8}.pt-bad{color:#a94c15}.pt-table{overflow:auto}.pt-table table{border-collapse:collapse;width:100%;font-size:13px;color:#233649}.pt-table td,.pt-table th{padding:10px;border-bottom:1px solid #dbe4ed;text-align:left;white-space:nowrap}.pt-table th{background:#f1f6fb}@media(max-width:760px){.pt-files{grid-template-columns:1fr}.pt-hero{padding:20px}.pt-hero h1{font-size:25px}.pt-metric{padding:12px}.pt-metric strong{font-size:22px}.pt-source{font-size:12px}.pt-digits{gap:8px}}
</style>
"""


def source_html(source, highlights=(), first_line=1):
    """Preserve the actual source characters, with separate unselectable line numbers."""
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
        actual = number + first_line - 1
        focus = " focus" if actual in highlights else ""
        rows.append(f'<div class="pt-line{focus}" data-line="{actual}"><span class="pt-lineno" aria-hidden="true">{actual}</span><span class="pt-text">{"".join(pieces) or " "}</span></div>')
    return "".join(rows)


def file_preview(path, highlights=(), start=None, end=None):
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    count = len(source.splitlines())
    if start is not None:
        source = "\n".join(source.splitlines()[start-1:end]) + "\n"
    # Only whole, syntactically complete sections are used for excerpts.
    body = source_html(source, highlights, start or 1)
    caption = "Python · 読み取り専用" if start is None else f"Python · {start}–{end or count}行"
    short = " short" if len(source.splitlines()) <= 22 else ""
    return f'<section class="pt-file" aria-label="{escape(path.name)} の内容"><header><strong>▤ {escape(path.name)}</strong><span>{caption}</span></header><div class="pt-source{short}" tabindex="0">{body}</div></section>'


def figure_svg():
    items = [(15, "画像", "[B, 1, 28, 28]"), (205, "Flatten", "[B, 784]"),
             (395, "Linear → ReLU", "[B, 128]"), (585, "Linear", "[B, 10]")]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 770 160" role="img" aria-label="画像を平坦化し、128次元の隠れ層から10クラスのスコアを作る">']
    for index, (x, title, shape) in enumerate(items):
        parts.append(f'<rect x="{x}" y="25" width="165" height="100" rx="12" fill="#f1f6fb" stroke="#b8ccdf"/><text x="{x+82}" y="61" text-anchor="middle" fill="#233649" font-size="17">{title}</text><text x="{x+82}" y="96" text-anchor="middle" fill="#2563a6" font-size="16">{shape}</text>')
        if index < 3:
            parts.append(f'<text x="{x+177}" y="82" text-anchor="middle" fill="#7890a5" font-size="23">→</text>')
    return "".join(parts) + '</svg>'


def curves_svg(history):
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 275" role="img" aria-label="epochごとの訓練と検証の損失・正解率。青が訓練、緑の破線が検証">']
    for panel, (key, title) in enumerate([("loss", "損失（小さいほどよい）"), ("accuracy", "正解率（高いほどよい）")]):
        x0, y0, width, height = 52 + panel * 370, 210, 285, 155
        ceiling = max(row[split][key] for row in history for split in ["train", "validation"]) * 1.15 if key == "loss" else 1
        parts.append(f'<text x="{x0}" y="26" fill="#233649" font-size="15">{title}</text>')
        for fraction in [0, .5, 1]:
            y = y0 - height * fraction
            value = ceiling * fraction
            label = f'{value:.0%}' if key == 'accuracy' else f'{value:.2f}'
            parts.append(f'<path d="M{x0} {y}h{width}" stroke="#dde6ee"/><text x="{x0-8}" y="{y+5}" text-anchor="end" fill="#5f7284" font-size="12">{label}</text>')
        for split, color, dash in [("train", "#2563a6", ""), ("validation", "#157365", 'stroke-dasharray="6 3"')]:
            points = [(x0 + width * i / max(1, len(history)-1), y0 - height * row[split][key] / ceiling) for i, row in enumerate(history)]
            coords = ' '.join(f'{x:.2f},{y:.2f}' for x,y in points)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5" {dash}/>')
            parts.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}"/>' for x,y in points)
        for i, row in enumerate(history):
            x = x0 + width * i / max(1, len(history)-1)
            parts.append(f'<text x="{x}" y="230" text-anchor="middle" fill="#5f7284" font-size="12">{row["epoch"]}</text>')
        parts.append(f'<text x="{x0+width}" y="251" text-anchor="end" fill="#5f7284" font-size="12">epoch</text>')
    return ''.join(parts) + '</svg>'


def load_results(directory):
    """Never combine a new run with stale evaluation results from older weights."""
    directory = Path(directory)
    if not (directory / "run.json").is_file():
        return None
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    weights = directory / "mnist_mlp.pt"
    if not weights.is_file() or hashlib.sha256(weights.read_bytes()).hexdigest() != run["weights_sha256"]:
        raise ValueError("重みと学習記録が一致しません。学習を完了してから結果を読み込んでください。")
    evaluation = None
    if (directory / "evaluation.json").is_file():
        candidate = json.loads((directory / "evaluation.json").read_text(encoding="utf-8"))
        if candidate["weights_sha256"] == run["weights_sha256"]:
            evaluation = candidate
    return {"run": run, "history": history, "evaluation": evaluation}


def digits_html(records):
    cards = []
    for row in records:
        bad = ' class="pt-bad"' if row['label'] != row['prediction'] else ''
        cards.append(f'<figure><img src="data:image/png;base64,{row["png_base64"]}" alt="テスト画像{row["index"]}、正解{row["label"]}、予測{row["prediction"]}"/><figcaption>#{row["index"]}<br>正解 {row["label"]} → <b{bad}>予測 {row["prediction"]}</b></figcaption></figure>')
    return '<div class="pt-digits">' + ''.join(cards) + '</div>'


def function_preview(path, name, highlights=()):
    import ast
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    node = next(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    return file_preview(path, highlights=highlights, start=node.lineno, end=node.end_lineno)
