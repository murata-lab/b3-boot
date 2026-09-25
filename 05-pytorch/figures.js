// 05 の図。Python から CONFIG = {name, data, caption} を受け取り、#root に描く。
// 幅が変わるたびに描き直し、描いた後は iframe の高さを内容に合わせる。
// 小さな部品（chart・dots・legendItem など）は 02〜04 の figures.js と同じ。

const NS = 'http://www.w3.org/2000/svg';
const COLOR = {
  ink: '#1f2a37', muted: '#6b7785', grid: '#e6eaef', axis: '#b5bec8', wash: '#f3f6f9',
  train: '#2a78d6', valid: '#eb6834', ghost: '#c5cdd6', test: '#4b5866',
};

const root = document.getElementById('root');
const tip = document.getElementById('tip');
let uid = 0;

// ---- 小さな部品 ---------------------------------------------------------

function svg(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

// $…$ で囲んだ部分は数式用の書体（イタリック）で描く。
function text(parent, x, y, content, attrs = {}) {
  const node = svg('text', { x, y, ...attrs }, parent);
  String(content).split('$').forEach((part, i) => {
    if (!part) return;
    if (i % 2) svg('tspan', { class: 'm' }, node).textContent = part;
    else node.appendChild(document.createTextNode(part));
  });
  return node;
}

// 文字列はすべてこの教材の中で作ったもの。<b> などのタグはそのまま使える。
function richHTML(content) {
  return String(content).split('$').map((part, i) => (i % 2 ? `<i class="m">${part}</i>` : part)).join('');
}

function html(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'onclick' || k === 'onchange') node[k] = v;
    else if (k === 'text') node.textContent = v;
    else if (k === 'html') node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  for (const child of children) node.append(child);
  return node;
}

function scale([d0, d1], [r0, r1]) {
  const f = v => r0 + (v - d0) / (d1 - d0) * (r1 - r0);
  f.invert = p => d0 + (p - r0) / (r1 - r0) * (d1 - d0);
  return f;
}

const num = (v, digits = 2) => (Math.abs(v) < 0.5 * 10 ** -digits ? 0 : v).toFixed(digits).replace('-', '−');
const pct = v => Math.round(v * 100) + '%';

function showTip(event, content) {
  tip.textContent = content;
  tip.style.opacity = 1;
  const w = tip.offsetWidth;
  const x = Math.min(event.clientX + 12, document.documentElement.clientWidth - w - 4);
  tip.style.transform = `translate(${Math.max(4, x)}px, ${Math.max(4, event.clientY - 34)}px)`;
}
const hideTip = () => { tip.style.opacity = 0; };

function hoverable(node, content) {
  node.style.cursor = 'default';
  node.addEventListener('pointerenter', e => showTip(e, content));
  node.addEventListener('pointermove', e => showTip(e, content));
  node.addEventListener('pointerleave', hideTip);
}

// 軸・目盛り・薄い横グリッドを持つ座標系。plot は描画範囲で切り取られる。
function chart(parent, o) {
  const m = { top: 30, right: 16, bottom: 46, left: 48, ...o.margin };
  const { width, height } = o;
  const s = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': o.label || '' }, parent);
  const x = scale(o.xlim, [m.left, width - m.right]);
  const y = scale(o.ylim, [height - m.bottom, m.top]);
  const back = svg('g', {}, s);
  for (const t of o.yticks || []) {
    if (o.grid !== false) svg('line', { x1: m.left, x2: width - m.right, y1: y(t), y2: y(t), stroke: COLOR.grid }, back);
    text(back, m.left - 8, y(t) + 4, (o.yfmt || String)(t), { 'text-anchor': 'end' });
  }
  svg('line', { x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom, stroke: COLOR.axis }, back);
  for (const t of o.xticks || []) {
    svg('line', { x1: x(t), x2: x(t), y1: height - m.bottom, y2: height - m.bottom + 4, stroke: COLOR.axis }, back);
    text(back, x(t), height - m.bottom + 19, (o.xfmt || String)(t), { 'text-anchor': 'middle' });
  }
  if (o.ylabel) text(back, 4, 14, o.ylabel, { class: 'ink' });
  if (o.xlabel) text(back, width - m.right, height - 6, o.xlabel, { 'text-anchor': 'end', class: 'ink' });
  const id = 'clip' + (++uid);
  const clip = svg('clipPath', { id }, svg('defs', {}, s));
  svg('rect', { x: m.left, y: m.top, width: width - m.left - m.right, height: height - m.top - m.bottom }, clip);
  const plot = svg('g', { 'clip-path': `url(#${id})` }, s);
  const over = svg('g', {}, s);
  return { s, x, y, m, width, height, plot, over, back };
}

function path(points, x, y) {
  return points.map(([a, b], i) => `${i ? 'L' : 'M'}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join('');
}

function dot(parent, cx, cy, color, { r = 4.5, hollow = false } = {}) {
  if (hollow) return svg('circle', { cx, cy, r, fill: '#fff', stroke: color, 'stroke-width': 2 }, parent);
  return svg('circle', { cx, cy, r, fill: color, stroke: '#fff', 'stroke-width': 2 }, parent);
}

// 点に大きめの当たり判定を重ね、値をツールチップで出す。
function dots(c, points, color, describe, options = {}) {
  const layer = svg('g', {}, c.plot);
  const hits = svg('g', {}, c.over);
  for (const p of points) {
    dot(layer, c.x(p[0]), c.y(p[1]), color, options);
    if (describe) hoverable(svg('circle', { cx: c.x(p[0]), cy: c.y(p[1]), r: 10, fill: 'transparent' }, hits), describe(p));
  }
}


function key(kind, color) {
  const s = svg('svg', { width: 22, height: 12, 'aria-hidden': 'true' });
  if (kind === 'dot') dot(s, 11, 6, color, { r: 4.5 });
  else if (kind === 'ring') dot(s, 11, 6, color, { r: 4, hollow: true });
  else if (kind === 'box') svg('rect', { x: 3, y: 0.5, width: 16, height: 11, rx: 2, fill: color }, s);
  else if (kind === 'hatch') {
    svg('rect', { x: 3, y: 0.5, width: 16, height: 11, rx: 2, fill: `url(#${hatch(s)})`, stroke: COLOR.ghost }, s);
  } else if (kind === 'outline') svg('rect', { x: 4, y: 1, width: 14, height: 10, rx: 2, fill: '#fff', stroke: color, 'stroke-width': 2 }, s);
  else svg('line', { x1: 1, x2: 21, y1: 6, y2: 6, stroke: color, 'stroke-width': 2.5,
    'stroke-dasharray': kind === 'dash' ? '4 3' : 'none', 'stroke-linecap': 'round' }, s);
  return s;
}

// 斜線の塗り。「使わない」データに使う。
function hatch(s) {
  const id = 'hatch' + (++uid);
  const p = svg('pattern', { id, width: 6, height: 6, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, svg('defs', {}, s));
  svg('rect', { width: 6, height: 6, fill: '#fff' }, p);
  svg('line', { x1: 0, y1: 0, x2: 0, y2: 6, stroke: COLOR.ghost, 'stroke-width': 3 }, p);
  return id;
}

function legendItem(kind, color, label) {
  return html('span', { class: 'key' }, [key(kind, color), html('span', { html: richHTML(label) })]);
}

function legend(items) {
  return html('div', { class: 'legend' }, items.map(([kind, color, label]) => legendItem(kind, color, label)));
}

const comma = v => v.toLocaleString('en-US');
const pct1 = v => (v * 100).toFixed(1) + '%';

// ---- 手書き数字の画像 --------------------------------------------------------

// 28×28 画素の値（0〜255）を16進数で並べた文字列から、白黒の画像を作る。
function digitCanvas(hex, size) {
  const c = html('canvas', { width: 28, height: 28, class: 'digit', role: 'img' });
  c.style.width = c.style.height = size + 'px';
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = parseInt(hex.substr(i * 2, 2), 16);
    img.data.set([v, v, v, 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

// 幅に合わせて1行に並べる枚数と、1枚の大きさを決める。
function gridSize(width, count) {
  const cols = width >= 600 ? count : Math.min(count, 4);
  const size = Math.min(76, Math.floor((width - (cols - 1) * 10) / cols) - 12);
  return { cols, size };
}

const FIGURES = {};

FIGURES.digits = (d) => (width) => {
  const { cols, size } = gridSize(width, 8);
  const grid = html('div', { class: 'digits' });
  grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
  for (const item of d.images) {
    const canvas = digitCanvas(item.pixels, size);
    canvas.setAttribute('aria-label', `正解 ${item.label} の画像`);
    grid.append(html('div', { class: 'card' }, [canvas, html('span', { html: `正解 <b class="ok">${item.label}</b>` })]));
  }
  root.append(grid);
};

FIGURES.pixels = (d) => (width) => {
  const n = d.box.size;
  const cell = width >= 600 ? 34 : Math.max(26, Math.min(34, Math.floor((width - 4) / n)));
  const big = Math.min(252, width >= 600 ? width - n * cell - 40 : width);
  const left = html('div', { class: 'panel' }, [html('h4', { html: '1枚の画像<span>28×28画素</span>' })]);
  const right = html('div', { class: 'panel' }, [html('h4', { html: `橙の枠の中の値<span>${n}×${n}画素</span>` })]);
  left.style.flex = right.style.flex = '0 0 auto';
  root.append(html('div', { class: 'row' }, [left, right]));

  const wrap = html('div', { class: 'pixwrap' });
  wrap.append(digitCanvas(d.pixels, big));
  const px = big / 28;
  const box = html('div', { class: 'pixbox' });
  Object.assign(box.style, { left: `${d.box.col * px - 2}px`, top: `${d.box.row * px - 2}px`, width: `${n * px + 4}px`, height: `${n * px + 4}px` });
  wrap.append(box);
  left.append(wrap);

  const table = html('table', { class: 'pix', 'aria-label': '画素の値' });
  for (let r = 0; r < n; r++) {
    const tr = html('tr');
    for (let c = 0; c < n; c++) {
      const i = (d.box.row + r) * 28 + d.box.col + c;
      const v = parseInt(d.pixels.substr(i * 2, 2), 16);
      const td = html('td', { text: String(v) });
      Object.assign(td.style, { width: `${cell}px`, height: `${cell}px`, background: `rgb(${v},${v},${v})`, color: v > 120 ? COLOR.ink : '#fff' });
      tr.append(td);
    }
    table.append(tr);
  }
  right.append(table);
};

// ---- モデルの中で、データの形がどう変わるか ------------------------------------

FIGURES.shapes = (d) => (width) => {
  const vertical = width < 700;
  const chain = html('div', { class: 'chain' + (vertical ? ' vertical' : '') });
  d.nodes.forEach((node, i) => {
    chain.append(html('div', { class: 'chain-node' }, [
      html('div', { class: 'name', text: node.name }), html('div', { class: 'shape', text: node.shape })]));
    const op = d.ops[i];
    if (!op) return;
    chain.append(html('div', { class: 'chain-op' }, [
      html('div', { class: 'code', text: op.code }), html('div', { class: 'arrow' }),
      html('div', { class: 'note', html: `${op.note}<br>${op.line}行目` })]));
  });
  root.append(chain);
};

// ---- データの分け方 ------------------------------------------------------------

FIGURES.split = (d) => (width) => {
  const kinds = { train: ['box', COLOR.train], valid: ['box', COLOR.valid], unused: ['hatch', ''], test: ['box', COLOR.test] };
  root.append(legend(d.bars.flatMap(b => b.parts).map(p => [...kinds[p.kind], `${p.name} <b>${comma(p.count)}</b>枚`])));
  const total = Math.max(...d.bars.map(b => b.parts.reduce((s, p) => s + p.count, 0)));
  const barH = 34, gap = 30, labelH = 22;
  const height = d.bars.length * (barH + labelH + gap) - gap + 4;
  const s = svg('svg', { width, height, viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': 'データの分け方' }, root);
  const fill = hatch(s);
  const x = scale([0, total], [1, width - 1]);
  d.bars.forEach((bar, k) => {
    const y = k * (barH + labelH + gap) + labelH;
    text(s, 0, y - 8, bar.title, { class: 'strong' });
    let at = 0;
    for (const p of bar.parts) {
      const x0 = x(at), x1 = x(at + p.count);
      const color = kinds[p.kind][1];
      svg('rect', { x: x0, y, width: x1 - x0 - 1, height: barH, rx: 3,
        fill: p.kind === 'unused' ? `url(#${fill})` : color, stroke: p.kind === 'unused' ? COLOR.ghost : 'none' }, s);
      const label = `${p.name} ${comma(p.count)}`;
      if (x1 - x0 > label.length * 8 + 12) {
        text(s, (x0 + x1) / 2, y + barH / 2 + 4.5, label, { 'text-anchor': 'middle', style: `fill:${p.kind === 'unused' ? COLOR.ink : '#fff'};font-weight:600` });
      }
      hoverable(svg('rect', { x: x0, y, width: x1 - x0, height: barH, fill: 'transparent' }, s), `${p.name}：${comma(p.count)}枚`);
      at += p.count;
    }
  });
};

// ---- 学習曲線 -----------------------------------------------------------------

FIGURES.curves = (d) => (width) => {
  const h = d.history, last = h[h.length - 1];
  root.append(legend([
    ['dot', COLOR.train, `訓練データ　損失 <b>${num(last.train.loss)}</b>　正解率 <b>${pct1(last.train.accuracy)}</b>`],
    ['ring', COLOR.valid, `検証データ　損失 <b>${num(last.validation.loss)}</b>　正解率 <b>${pct1(last.validation.accuracy)}</b>`]]));
  const row = html('div', { class: 'row' });
  root.append(row);
  const n = h.length;
  const step = n <= 10 ? 1 : 5;
  const xticks = [1, ...Array.from({ length: Math.floor(n / step) }, (_, k) => (k + 1) * step)].filter((t, i, a) => t <= n && a.indexOf(t) === i);
  const panels = [
    { key: 'loss', title: '損失<span>小さいほどよい</span>', fmt: v => num(v, 1) },
    { key: 'accuracy', title: '正解率<span>高いほどよい</span>', fmt: v => Math.round(v * 100) + '%' },
  ];
  // 二つのパネルを先に並べてから、それぞれの幅で描く
  const boxes = panels.map(p => html('div', { class: 'panel' }, [html('h4', { html: p.title })]));
  row.append(...boxes);
  panels.forEach((p, i) => {
    const box = boxes[i];
    const values = h.flatMap(r => [r.train[p.key], r.validation[p.key]]);
    let ylim, yticks;
    if (p.key === 'loss') {
      const top = Math.ceil(Math.max(...values) * 10 + 0.5) / 10;
      const tick = top > 1 ? 0.5 : top > 0.4 ? 0.2 : 0.1;
      ylim = [0, top];
      yticks = Array.from({ length: Math.floor(top / tick + 1e-9) + 1 }, (_, k) => +(k * tick).toFixed(1));
    } else {
      const low = Math.max(0, Math.floor(Math.min(...values) * 10 - 0.2) / 10);
      ylim = [low, 1];
      const tick = 1 - low > 0.3 ? 0.1 : 0.05;
      yticks = Array.from({ length: Math.round((1 - low) / tick) + 1 }, (_, k) => +(low + k * tick).toFixed(2));
    }
    const w = box.clientWidth;
    const c = chart(box, { width: w, height: Math.min(260, Math.max(200, w * 0.62)), xlim: [0.6, n + 0.4], ylim, xticks, yticks,
      yfmt: p.fmt, xlabel: 'エポック', margin: { top: 12, left: 44 }, label: p.key });
    for (const [split, color, hollow] of [['train', COLOR.train, false], ['validation', COLOR.valid, true]]) {
      const pts = h.map(r => [r.epoch, r[split][p.key]]);
      svg('path', { d: path(pts, c.x, c.y), fill: 'none', stroke: color, 'stroke-width': 2, 'stroke-linejoin': 'round',
        'stroke-dasharray': hollow ? '6 4' : 'none' }, c.plot);
      const name = split === 'train' ? '訓練' : '検証';
      dots(c, pts, color, q => `${q[0]}エポック目：${name}の${p.key === 'loss' ? '損失 ' + num(q[1], 3) : '正解率 ' + pct1(q[1])}`,
        { hollow, r: n > 10 ? 3.5 : 4.5 });
    }
  });
};

// ---- テスト画像の予測と、10個の確率 ---------------------------------------------

FIGURES.predictions = (d) => {
  let picked = [d.rows.length - 1, 0];
  return (width) => {
    root.append(html('p', { class: 'status', text: '画像を押すと、その画像に対するモデルの10個の確率を下のグラフに表示します。' }));
    const { cols, size } = gridSize(width, 8);
    const cards = [];
    d.rows.forEach((row, r) => {
      root.append(html('div', { class: 'row-title', html: row.title + (row.note ? `<span>${row.note}</span>` : '') }));
      const grid = html('div', { class: 'digits' });
      grid.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
      row.items.forEach((item, i) => {
        const wrong = item.pred !== item.label;
        const card = html('button', { type: 'button', class: 'card', 'aria-pressed': 'false', onclick: () => pick(r, i) }, [
          digitCanvas(item.pixels, size),
          html('span', { html: `正解 ${item.label}` }),
          html('span', { html: `予測 <b class="${wrong ? 'wrong' : 'ok'}">${item.pred}</b>` })]);
        card.setAttribute('aria-label', `テスト画像 ${item.index}番：正解 ${item.label}、予測 ${item.pred}`);
        cards.push([r, i, card]);
        grid.append(card);
      });
      root.append(grid);
    });

    const head = html('div', { class: 'row-title', style: 'margin-top:14px' });
    const keys = legend([['box', COLOR.train, '予測（確率が最も大きい数字）'], ['outline', COLOR.valid, '正解']]);
    const holder = html('div');
    root.append(head, keys, holder);

    function pick(r, i) {
      picked = [r, i];
      for (const [rr, ii, card] of cards) card.setAttribute('aria-pressed', String(rr === r && ii === i));
      const item = d.rows[r].items[i];
      head.innerHTML = `テスト画像 ${comma(item.index)}番の確率<span>正解 ${item.label}・予測 ${item.pred}</span>`;
      holder.replaceChildren();
      const c = chart(holder, { width, height: 220, xlim: [-0.5, 9.5], ylim: [0, 1], xticks: [...Array(10).keys()],
        yticks: [0, 0.25, 0.5, 0.75, 1], yfmt: v => Math.round(v * 100) + '%', xlabel: '数字', ylabel: '確率',
        margin: { top: 26, left: 44 }, label: '10個の数字それぞれの確率' });
      const bw = Math.min(34, (c.x(1) - c.x(0)) * 0.62);
      item.probs.forEach((p, k) => {
        const isPred = k === item.pred, isTrue = k === item.label;
        const y = c.y(p), h0 = c.y(0) - y;
        svg('rect', { x: c.x(k) - bw / 2, y, width: bw, height: Math.max(h0, 0.5), rx: 2, fill: isPred ? COLOR.train : COLOR.ghost }, c.plot);
        if (isTrue) svg('rect', { x: c.x(k) - bw / 2 - 3, y: Math.min(y, c.y(0) - 4) - 3, width: bw + 6, height: Math.max(h0, 4) + 3,
          rx: 3, fill: 'none', stroke: COLOR.valid, 'stroke-width': 2.2 }, c.over);
        if (p >= 0.005) text(c.over, c.x(k), y - (isTrue ? 9 : 5), Math.round(p * 100) + '%', { 'text-anchor': 'middle', class: isPred || isTrue ? 'strong' : '' });
        hoverable(svg('rect', { x: c.x(k) - bw / 2, y: c.m.top, width: bw, height: c.y(0) - c.m.top, fill: 'transparent' }, c.over), `${k}：${(p * 100).toFixed(1)}%`);
      });
      fit();
    }
    pick(...picked);
  };
};

// ---- 確認問題 ---------------------------------------------------------------
// 選ぶとすぐに解説を出す。この章のセルは自動で再実行されない設定なので、答え合わせは図の中で行う。

FIGURES.quiz = (d) => {
  const chosen = d.questions.map(() => null);
  return () => {
    d.questions.forEach((q, k) => {
      const box = html('div', { class: 'quiz' });
      box.append(html('p', { class: 'q', html: richHTML(q.question) }));
      const opts = html('div', { class: 'opts', role: 'radiogroup' });
      const feedback = html('div');
      q.options.forEach((option, i) => {
        const button = html('button', { type: 'button', class: 'opt', role: 'radio', 'aria-checked': String(chosen[k] === i),
          onclick: () => { chosen[k] = i; show(); } }, [html('span', { class: 'mark' }), html('span', { html: richHTML(option) })]);
        opts.append(button);
      });
      box.append(opts, feedback);
      root.append(box);
      function show() {
        [...opts.children].forEach((b, i) => b.setAttribute('aria-checked', String(chosen[k] === i)));
        if (chosen[k] === null) return;
        const ok = chosen[k] === q.answer;
        feedback.className = 'feedback ' + (ok ? 'ok' : 'ng');
        feedback.innerHTML = (ok ? '<b>正解です。</b> ' : `<b>正解は「${richHTML(q.options[q.answer])}」です。</b> `) + richHTML(q.explanation);
        fit();
      }
      show();
    });
  };
};

// ---- 描画と高さ合わせ --------------------------------------------------------

function fit() {
  if (window.frameElement) window.frameElement.style.height = Math.ceil(document.body.getBoundingClientRect().height) + 'px';
}

FIGURES.current = FIGURES[CONFIG.name](CONFIG.data);
const baseRender = FIGURES.current;
FIGURES.current = (width) => {
  baseRender(width);
  if (CONFIG.caption) root.append(html('figcaption', { html: richHTML(CONFIG.caption) }));
};

let lastWidth = 0;
new ResizeObserver(() => {
  const width = root.clientWidth;
  // 幅がまだ決まっていない（0に近い）間は描かない。負の幅の図形ができるため
  if (width !== lastWidth && width > 100) {
    lastWidth = width;
    hideTip();
    root.replaceChildren();
    FIGURES.current(width);
  }
  fit();
}).observe(document.body);
