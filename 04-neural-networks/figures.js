// 04 の図。Python から CONFIG = {name, data, caption} を受け取り、#root に描く。
// 幅が変わるたびに描き直し、描いた後は iframe の高さを内容に合わせる。
// 小さな部品（chart・dots・legendItem・segmented など）は 02 の figures.js と同じ。

const NS = 'http://www.w3.org/2000/svg';
const COLOR = {
  ink: '#1f2a37', muted: '#6b7785', grid: '#e6eaef', axis: '#b5bec8', wash: '#f3f6f9',
  train: '#2a78d6', valid: '#eb6834', grad: '#c4531d', ghost: '#c5cdd6',
};
// ニューロンの出力 h の大きさ。0は白、大きいほど濃い青。
const RAMP = ['#ffffff', '#f2f7fe', '#e4effc', '#d6e6fa', '#c8def8', '#b9d4f6', '#aacbf4',
  '#9bc1f1', '#8bb7ee', '#7cadeb', '#6da3e8', '#5e99e4'];
// 快適と予測する確率 p（0〜0.2、0.2〜0.5、0.5〜0.8、0.8〜1）。橙が不快、青が快適の側。
const PROB = ['#fbe1d3', '#fdf1ea', '#eaf2fc', '#d5e5f9'];

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
  else if (kind === 'star') svg('path', { d: starPath(11, 6, 6.5), fill: color }, s);
  else if (kind === 'box') svg('rect', { x: 3, y: 0.5, width: 16, height: 11, rx: 2, fill: color, stroke: COLOR.axis, 'stroke-width': 0.6 }, s);
  else if (kind === 'ramp') [2, 5, 8, 11].forEach((k, i) => svg('circle', { cx: 3 + i * 5.5, cy: 6, r: 2.6, fill: RAMP[k], stroke: COLOR.muted, 'stroke-width': 0.6 }, s));
  else svg('line', { x1: 1, x2: 21, y1: 6, y2: 6, stroke: color, 'stroke-width': kind === 'thin' ? 1.5 : 2.5,
    'stroke-dasharray': kind === 'dash' ? '4 3' : 'none', 'stroke-linecap': 'round' }, s);
  return s;
}

function legendItem(kind, color, label) {
  return html('span', { class: 'key' }, [key(kind, color), html('span', { html: richHTML(label) })]);
}

function legend(items) {
  return html('div', { class: 'legend' }, items.map(([kind, color, label]) => legendItem(kind, color, label)));
}

function starPath(x, y, r) {
  const pts = [];
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r;
    pts.push(`${(x + rr * Math.cos(a)).toFixed(1)},${(y + rr * Math.sin(a)).toFixed(1)}`);
  }
  return 'M' + pts.join('L') + 'Z';
}

function segmented(labels, active, onpick, aria) {
  return html('div', { class: 'seg', role: 'group', 'aria-label': aria }, labels.map((label, i) =>
    html('button', { type: 'button', 'aria-pressed': String(i === active), text: label, onclick: () => onpick(i) })));
}

// ---- 気温・湿度の平面 ------------------------------------------------------

const TLIM = [14, 34], ULIM = [20, 85];

function plane(parent, width, height, small = false) {
  return chart(parent, { width, height, xlim: TLIM, ylim: ULIM, xticks: [15, 20, 25, 30],
    yticks: small ? [20, 50, 80] : [20, 35, 50, 65, 80], xlabel: '気温（℃）', ylabel: '湿度（%）',
    margin: small ? { top: 24, right: 8, bottom: 42, left: 30 } : {}, label: '気温と湿度の平面' });
}

const planeHeight = (width, small = false) => (small ? Math.min(300, Math.round(width * 0.95)) : Math.min(360, Math.round(width * 0.62 + 40)));

// Python の contourpy で求めた、色の段ごとの多角形を塗る。
function bands(c, list, colors) {
  const g = svg('g', {}, c.plot);
  list.forEach((rings, i) => {
    const d = rings.map(r => {
      let s = '';
      for (let k = 0; k < r.length; k += 2) s += (k ? 'L' : 'M') + c.x(r[k]).toFixed(1) + ',' + c.y(r[k + 1]).toFixed(1);
      return s + 'Z';
    }).join('');
    if (d) svg('path', { d, fill: colors[i], stroke: colors[i], 'stroke-width': 0.6, 'fill-rule': 'evenodd' }, g);
  });
}

function polylines(c, lines, attrs) {
  for (const flat of lines) {
    const pts = [];
    for (let k = 0; k < flat.length; k += 2) pts.push([flat[k], flat[k + 1]]);
    svg('path', { d: path(pts, c.x, c.y), fill: 'none', 'stroke-linejoin': 'round', ...attrs }, c.plot);
  }
}

const describeRoom = p => `気温 ${num(p[0], 1)}℃　湿度 ${Math.round(p[1])}%　${p[2] ? '快適' : '不快'}`;

function rooms(c, list, { r = 4.5, skip = null } = {}) {
  const rest = skip ? list.filter(p => p[0] !== skip[0] || p[1] !== skip[1]) : list;
  dots(c, rest.filter(p => p[2]), COLOR.train, describeRoom, { r });
  dots(c, rest.filter(p => !p[2]), COLOR.valid, describeRoom, { r: r - 0.5, hollow: true });
}

function star(c, room, r = 9) {
  const [x, y] = [c.x(room[0]), c.y(room[1])];
  svg('path', { d: starPath(x, y, r), fill: COLOR.train, stroke: '#fff', 'stroke-width': 1.5 }, c.over);
  hoverable(svg('circle', { cx: x, cy: y, r: r + 2, fill: 'transparent' }, c.over), describeRoom([...room, 1]));
}

const ROOM_KEYS = [['dot', COLOR.train, '快適と答えた部屋'], ['ring', COLOR.valid, '不快と答えた部屋']];
const PROB_KEYS = [['box', PROB[3], '快適と予測'], ['box', PROB[0], '不快と予測'], ['line', COLOR.ink, '決定境界']];

// ---- 図 -------------------------------------------------------------------

const FIGURES = {};

FIGURES.rooms = (d) => (width) => {
  root.append(legend(ROOM_KEYS));
  const c = plane(root, width, planeHeight(width));
  rooms(c, d.rooms);
};

FIGURES.probability = (d) => (width) => {
  root.append(html('div', { class: 'legend' }, [...[...ROOM_KEYS, ...PROB_KEYS].map(k => legendItem(...k)),
    html('span', { class: 'key', html: `正解率 <b>${pct(d.acc)}</b>` })]));
  const c = plane(root, width, planeHeight(width));
  bands(c, d.bands, PROB);
  polylines(c, d.boundary, { stroke: COLOR.ink, 'stroke-width': 2.5 });
  rooms(c, d.rooms);
};

FIGURES.relu = () => (width) => {
  const w = Math.min(width, 520);
  const c = chart(root, { width: w, height: Math.min(280, w * 0.6 + 40), xlim: [-3, 3], ylim: [-0.4, 3], xticks: [-3, -2, -1, 0, 1, 2, 3],
    yticks: [0, 1, 2, 3], xfmt: t => num(t, 0), xlabel: '重み付き和 $z$', ylabel: '出力 $h$', label: 'ReLU のグラフ' });
  c.s.style.margin = '0 auto';
  svg('line', { x1: c.x(0), x2: c.x(0), y1: c.y(-0.4), y2: c.y(3), stroke: COLOR.axis, 'stroke-dasharray': '4 4' }, c.plot);
  svg('path', { d: path([[-3, 0], [0, 0], [3, 3]], c.x, c.y), fill: 'none', stroke: COLOR.train, 'stroke-width': 3, 'stroke-linejoin': 'round' }, c.plot);
  text(c.over, c.x(-1.5), c.y(0) - 10, '負なら0', { 'text-anchor': 'middle', class: 'ink' });
  text(c.over, c.x(1.2), c.y(2.75), '正ならそのまま', { 'text-anchor': 'end', class: 'ink' });
  for (const [z, label, dx, dy, anchor] of [[-1, '$z$ = −1 → $h$ = 0', 0, 22, 'middle'], [2, '$z$ = 2 → $h$ = 2', -12, 4, 'end']]) {
    dot(c.over, c.x(z), c.y(Math.max(0, z)), COLOR.ink, { r: 4.5 });
    text(c.over, c.x(z) + dx, c.y(Math.max(0, z)) + dy, label, { 'text-anchor': anchor, class: 'ink' });
  }
};

// よく見るネットワークの図と、その中の1つのニューロンで行う計算。
FIGURES.anatomy = (d) => (width) => {
  const W = Math.min(width, 560), H = 236, top = 14, bottom = 42, R = 15;
  const s = svg('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, style: 'margin:0 auto', role: 'img',
    'aria-label': 'ニューロンを丸、重みを線で描いたネットワークの図' }, root);
  const L = d.layers.length, pad = 44;
  const xs = d.layers.map((_, i) => pad + i * (W - 2 * pad) / (L - 1));
  const gapY = (H - top - bottom) / Math.max(...d.layers);
  const ys = d.layers.map(n => [...Array(n).keys()].map(k => top + (H - top - bottom) / 2 + (k - (n - 1) / 2) * gapY));
  const [fl, fk] = d.focus;
  for (let i = 0; i < L - 1; i++) ys[i].forEach(y1 => ys[i + 1].forEach((y2, k) => {
    const hot = i + 1 === fl && k === fk;
    svg('line', { x1: xs[i] + R, y1, x2: xs[i + 1] - R, y2, stroke: hot ? COLOR.train : '#c9d2dc', 'stroke-width': hot ? 2.2 : 1.1 }, s);
  }));
  ys.forEach((col, i) => col.forEach((y, k) => {
    const hot = i === fl && k === fk;
    svg('circle', { cx: xs[i], cy: y, r: R, fill: hot ? '#dbe9fb' : '#fff', stroke: hot ? COLOR.train : COLOR.axis, 'stroke-width': hot ? 2.4 : 1.4 }, s);
    if (i === 0) text(s, xs[i], y + 5, `$x$${'₁₂₃₄'[k]}`, { 'text-anchor': 'middle', class: 'ink' });
  }));
  d.layers.forEach((_, i) => text(s, xs[i], H - 12, i === 0 ? '入力層' : i === L - 1 ? '出力層' : '隠れ層', { 'text-anchor': 'middle', class: 'ink' }));

  // 青いニューロンの中の計算
  const node = (title, body, focus = false) => html('div', { class: 'chain-node' + (focus ? ' focus' : '') }, [
    html('div', { class: 'name', html: richHTML(title) }), html('div', { class: 'delta', html: richHTML(body) })]);
  const op = note => html('div', { class: 'chain-op' }, [html('div', { class: 'arrow', 'aria-hidden': 'true' }),
    html('div', { class: 'note', html: richHTML(note) })]);
  root.append(html('div', { class: 'chain anatomy' + (width < 600 ? ' vertical' : '') }, [
    node('前の層の値', '$x$₁, $x$₂'), op('重みを掛けて足す'),
    node('① 重み付き和', '$z$ = $w$₁$x$₁ + $w$₂$x$₂ + $b$', true), op('活性化関数に通す'),
    node('② 活性化関数', '$h$ = $f$($z$)', true), op('次の層へ渡す'),
    node('出力', '$h$')]));
};

// 任意の折りたたみ：3つの活性化関数のグラフ。
FIGURES.activations = () => (width) => {
  const fns = [
    ['ReLU', 'max(0, $z$)', z => Math.max(0, z)],
    ['sigmoid', '1 / (1 + $e$⁻ᶻ)', z => 1 / (1 + Math.exp(-z))],
    ['tanh', '($e$ᶻ − $e$⁻ᶻ) / ($e$ᶻ + $e$⁻ᶻ)', z => Math.tanh(z)],
  ];
  const row = html('div', { class: 'row grid4' });
  root.append(row);
  const panels = fns.map(([name, formula]) => {
    const box = html('div', { class: 'panel' }, [html('h4', { html: `${name}<span>${richHTML(formula)}</span>` })]);
    // 折り返したときに1枚だけ大きくならないよう、幅をそろえる
    if (width < 520) box.style.maxWidth = 'calc(50% - 10px)';
    row.append(box);
    return box;
  });
  fns.forEach(([, , f], i) => {
    const w = panels[i].clientWidth;
    const c = chart(panels[i], { width: w, height: Math.min(220, Math.round(w * 0.8)), xlim: [-4, 4], ylim: [-1.3, 2.3],
      xticks: [-4, -2, 0, 2, 4], yticks: [-1, 0, 1, 2], xfmt: t => num(t, 0), yfmt: t => num(t, 0),
      xlabel: '$z$', margin: { top: 16, right: 8, bottom: 40, left: 30 }, label: '活性化関数のグラフ' });
    const pts = [...Array(161).keys()].map(k => { const z = -4 + k * 0.05; return [z, f(z)]; });
    svg('path', { d: path(pts, c.x, c.y), fill: 'none', stroke: COLOR.train, 'stroke-width': 2.5 }, c.plot);
  });
};

FIGURES.neurons = (d) => (width) => {
  root.append(legend([['ramp', '', '色が濃いほど $h$ が大きい'], ['dash', COLOR.ink, '$z$ = 0 の直線（ReLUが折れる場所）']]));
  const row = html('div', { class: 'row grid4' });
  root.append(row);
  const panels = d.panels.map(p => {
    const box = html('div', { class: 'panel' }, [html('h4', { html: `${richHTML(p.title)}<span>${richHTML(p.formula)}</span>` })]);
    row.append(box);
    return box;
  });
  d.panels.forEach((p, i) => {
    const w = panels[i].clientWidth;
    const c = plane(panels[i], w, planeHeight(w, true), true);
    bands(c, p.bands, [RAMP[0], RAMP[3], RAMP[5], RAMP[7], RAMP[9], RAMP[10], RAMP[11]]);
    polylines(c, p.fold, { stroke: COLOR.ink, 'stroke-width': 1.5, 'stroke-dasharray': '5 4' });
  });
};

// ニューロンの数だけを変えて学習した結果を並べる。
FIGURES.counts = (d) => (width) => {
  root.append(legend([...ROOM_KEYS, ...PROB_KEYS]));
  const row = html('div', { class: 'row grid4' });
  root.append(row);
  const panels = d.panels.map(p => {
    const box = html('div', { class: 'panel' }, [html('h4', { html: richHTML(p.title) })]);
    row.append(box);
    return box;
  });
  d.panels.forEach((p, i) => {
    const w = panels[i].clientWidth;
    const c = plane(panels[i], w, planeHeight(w, true), true);
    bands(c, p.bands, PROB);
    polylines(c, p.boundary, { stroke: COLOR.ink, 'stroke-width': 2 });
    rooms(c, d.rooms, { r: 3.2 });
  });
};

// 入力・隠れ層・出力（・損失）の箱を並べたネットワークの図。幅が狭いときは縦に並べる。
FIGURES.network = (d) => (width) => {
  const back = d.mode === 'backward';
  const groups = [
    { title: '入力', boxes: d.inputs.map(b => (d.mode === 'structure'
      ? [b.name, b.detail] : [`${b.name} = ${b.value}`, b.raw])) },
    { title: '隠れ層', boxes: d.hidden.map(h => [h.name, ...h[d.mode]]) },
    { title: '出力', boxes: [['出力', ...d.output[d.mode]]] },
  ];
  if (back) groups.push({ title: '損失', boxes: [['損失', ...d.loss]] });
  const dim = d.hidden.map(h => d.mode !== 'structure' && !h.active);
  const LINE = 19, PAD = 11;
  const colW = back ? [118, 214, 214, 170] : [118, 214, 250];
  const horizontal = width >= colW.reduce((a, b) => a + b, 0) + 44 * (colW.length - 1);
  const font = horizontal ? 13.5 : 12.5;
  const lines = Math.max(...groups[1].boxes.map(b => b.length));
  const hBox = PAD * 2 + LINE * lines;

  // 箱の位置を決める
  const place = [];
  let H;
  if (horizontal) {
    const gap = (width - colW.reduce((a, b) => a + b, 0)) / (colW.length - 1);
    const xs = colW.map((_, i) => colW.slice(0, i).reduce((a, b) => a + b, 0) + gap * i);
    const top = 26, vgap = 12, hiddenH = 4 * hBox + 3 * vgap;
    H = top + hiddenH + 6;
    groups.forEach((g, gi) => {
      g.boxes.forEach((b, bi) => {
        const h = PAD * 2 + LINE * b.length;
        let cy = top + hiddenH / 2;
        if (gi === 1) cy = top + bi * (hBox + vgap) + hBox / 2;
        if (gi === 0) cy = top + (bi === 0 ? hBox + vgap / 2 : 3 * hBox + 2.5 * vgap);
        place.push({ g: gi, i: bi, x: xs[gi], y: cy - (gi === 1 ? hBox : h) / 2, w: colW[gi], h: gi === 1 ? hBox : h, lines: b });
      });
    });
  } else {
    let y = 0;
    const gap = 10;
    groups.forEach((g, gi) => {
      y += 22;
      const perRow = gi === 1 ? 2 : g.boxes.length;
      const w = (width - gap * (perRow - 1)) / perRow;
      const rowsH = [];
      g.boxes.forEach((b, bi) => {
        const r = Math.floor(bi / perRow), col = bi % perRow;
        const h = gi === 1 ? hBox : PAD * 2 + LINE * b.length;
        rowsH[r] = h;
        place.push({ g: gi, i: bi, x: col * (w + gap), y: y + rowsH.slice(0, r).reduce((a, b) => a + b + gap, 0), w, h, lines: b, row: r });
      });
      g.top = y - 22;
      y += rowsH.reduce((a, b) => a + b, 0) + gap * (rowsH.length - 1);
      g.bottom = y;
      y += 34;
    });
    H = y - 34 + 6;
  }

  const s = svg('svg', { width, height: H, viewBox: `0 0 ${width} ${H}`, role: 'img', 'aria-label': 'ネットワークの図' }, root);
  const defs = svg('defs', {}, s);
  for (const [id, color] of [['arrow', COLOR.axis], ['arrowInk', COLOR.muted], ['arrowGrad', COLOR.grad]]) {
    const mk = svg('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
    svg('path', { d: 'M0,0L10,5L0,10Z', fill: color }, mk);
  }
  const edgeLayer = svg('g', {}, s);
  const gradLayer = svg('g', {}, s);
  const boxLayer = svg('g', {}, s);
  const find = (g, i) => place.find(p => p.g === g && p.i === i);

  // 見出し
  groups.forEach((g, gi) => {
    if (horizontal) {
      const p = find(gi, 0);
      text(boxLayer, p.x + p.w / 2, 14, g.title, { 'text-anchor': 'middle', class: 'strong' });
    } else text(boxLayer, 0, g.top + 15, g.title, { class: 'strong' });
  });

  // 矢印
  const edge = (a, b, attrs, reverse = false) => {
    const [x1, y1, x2, y2] = horizontal ? [a.x + a.w, a.y + a.h / 2, b.x - 2, b.y + b.h / 2] : [0, 0, 0, 0];
    const [p, q] = reverse ? [[x2, y2], [x1, y1]] : [[x1, y1], [x2, y2]];
    svg('line', { x1: p[0], y1: p[1], x2: q[0], y2: q[1], ...attrs }, edgeLayer);
  };
  if (horizontal) {
    d.hidden.forEach((h, j) => h.weights.forEach((w, i) => edge(find(0, i), find(1, j), w
      ? { stroke: COLOR.muted, 'stroke-width': 1.5, 'marker-end': 'url(#arrowInk)' }
      : { stroke: '#e3e8ed', 'stroke-width': 1.2, 'marker-end': 'url(#arrow)' })));
    d.hidden.forEach((_, j) => edge(find(1, j), find(2, 0), { stroke: COLOR.muted, 'stroke-width': 1.5, 'marker-end': 'url(#arrowInk)' }));
    if (back) {
      edge(find(2, 0), find(3, 0), { stroke: COLOR.muted, 'stroke-width': 1.5, 'marker-end': 'url(#arrowInk)' });
      // 勾配は右から左へ。前向きの矢印と重ならないよう、少しずらして描く。
      const shift = (a, b) => {
        const [x1, y1, x2, y2] = [a.x + a.w, a.y + a.h / 2 + 7, b.x - 2, b.y + b.h / 2 + 7];
        svg('line', { x1: x2, y1: y2, x2: x1 + 2, y2: y1, stroke: COLOR.grad, 'stroke-width': 2, 'stroke-dasharray': '6 4', 'marker-end': 'url(#arrowGrad)' }, gradLayer);
      };
      shift(find(2, 0), find(3, 0));
      d.hidden.forEach((_, j) => shift(find(1, j), find(2, 0)));
    }
  } else {
    // 縦並びでは、層と層の間に1本ずつ矢印を描く
    groups.slice(0, -1).forEach((g, gi) => {
      const y1 = g.bottom + 4, y2 = groups[gi + 1].top + 2, cx = width / 2;
      svg('line', { x1: back ? cx - 12 : cx, y1, x2: back ? cx - 12 : cx, y2, stroke: COLOR.muted, 'stroke-width': 1.5, 'marker-end': 'url(#arrowInk)' }, edgeLayer);
      if (back && gi >= 1) svg('line', { x1: cx + 12, y1: y2, x2: cx + 12, y2: y1, stroke: COLOR.grad, 'stroke-width': 2, 'stroke-dasharray': '5 3', 'marker-end': 'url(#arrowGrad)' }, gradLayer);
    });
  }

  // 箱
  for (const p of place) {
    const off = p.g === 1 && dim[p.i];
    const hot = back && ((p.g === 1 && !dim[p.i]) || p.g >= 2);
    svg('rect', { x: p.x + 0.5, y: p.y + 0.5, width: p.w - 1, height: p.h - 1, rx: 9,
      fill: off ? '#f6f8fa' : p.g === 1 || p.g === 2 ? '#f4f8fe' : '#fff',
      stroke: hot ? COLOR.grad : off ? '#d7dde3' : COLOR.axis, 'stroke-width': hot ? 1.6 : 1.1 }, boxLayer);
    p.lines.forEach((line, k) => {
      const cls = k === 0 ? 'strong' : line.startsWith('∂') ? 'grad' : 'ink';
      const size = k === 0 ? font + 0.5 : font;
      const t = text(boxLayer, p.x + 12, p.y + PAD + LINE * k + 14, line,
        { class: off && k > 0 && cls !== 'grad' ? '' : cls, style: `font-size:${size}px` });
      // 狭い箱では、はみ出さないように文字を小さくする
      const room = p.w - 20, used = t.getComputedTextLength();
      if (used > room) t.style.fontSize = (size * room / used).toFixed(1) + 'px';
    });
  }
};

// b₁ の小さな変化が、損失まで何倍されながら伝わるか。
FIGURES.chain = (d) => (width) => {
  root.append(html('p', { class: 'chain-head', html: richHTML('各値の下の数が、$b$₁ を0.01増やしたときの変化') }));
  const items = [];
  d.nodes.forEach((name, i) => {
    items.push(html('div', { class: 'chain-node' + (i === d.nodes.length - 1 ? ' last' : '') }, [
      html('div', { class: 'name', html: richHTML(name) }), html('div', { class: 'delta', text: d.changes[i] })]));
    if (i < d.factors.length) items.push(html('div', { class: 'chain-op' }, [
      html('div', { class: 'times', text: `×${d.factors[i].startsWith('−') ? `(${d.factors[i]})` : d.factors[i]}` }),
      html('div', { class: 'arrow', 'aria-hidden': 'true' }), html('div', { class: 'note', html: richHTML(d.notes[i]) })]));
  });
  root.append(html('div', { class: 'chain' + (width < 600 ? ' vertical' : '') }, items));
};

// 3節のデモ：更新回数を選ぶと、その時点の境界と、そこまでの損失の変化を表示する。
FIGURES.training = (d) => {
  const state = { index: 0 };
  const n = d.steps.length;
  const ymax = Math.ceil(Math.max(...d.nn.map(s => s.loss), ...d.linear.map(s => s.loss)) * 5 + 0.01) / 5;
  return (width) => {
    const i = state.index;
    const redraw = () => { root.replaceChildren(); FIGURES.current(root.clientWidth); fit(); };
    const pick = k => { state.index = k; redraw(); };
    root.append(html('div', { class: 'controls' }, [html('div', { class: 'control' }, ['更新回数',
      segmented(d.steps.map(String), i, pick, '更新回数')])]));
    root.append(html('p', { class: 'status', role: 'status', html: richHTML(d.status[i]) }));

    const leftPanel = html('div', { class: 'panel' }, [html('div', { class: 'legend' }, [
      ...ROOM_KEYS.map(k => legendItem(...k)), legendItem('star', COLOR.train, `${d.room[0]}℃・${d.room[1]}%の部屋（4節で使う）`),
      ...PROB_KEYS.map(k => legendItem(...k)), html('span', { class: 'key', html: `正解率 <b>${pct(d.nn[i].acc)}</b>` })])]);
    const rightPanel = html('div', { class: 'panel' }, [html('div', { class: 'legend' }, [
      legendItem('line', COLOR.train, `ネットワーク　損失 <b>${num(d.nn[i].loss, 3)}</b>`),
      legendItem('dash', COLOR.valid, `線形モデル　損失 <b>${num(d.linear[i].loss, 3)}</b>`)])]);
    root.append(html('div', { class: 'row' }, [leftPanel, rightPanel]));
    const size = Math.min(340, Math.max(250, leftPanel.clientWidth * 0.8));

    const c = plane(leftPanel, leftPanel.clientWidth, size);
    bands(c, d.maps[i].bands, PROB);
    polylines(c, d.maps[i].boundary, { stroke: COLOR.ink, 'stroke-width': 2.5 });
    rooms(c, d.rooms, { skip: d.room });
    star(c, d.room);

    // 右：更新回数ごとの損失。選んだ回数までを描く。列を押してもその回数に切り替わる
    const R = chart(rightPanel, { width: rightPanel.clientWidth, height: size, xlim: [-0.5, n - 0.5], ylim: [0, ymax],
      xticks: [...Array(n).keys()], xfmt: k => d.steps[k], yticks: [...Array(Math.round(ymax / 0.2) + 1).keys()].map(k => +(k * 0.2).toFixed(1)),
      xlabel: '更新回数', ylabel: '訓練データの平均損失', label: '更新回数ごとの損失' });
    const band = (R.x(1) - R.x(0)) * 0.8;
    svg('rect', { x: R.x(i) - band / 2, y: R.m.top, width: band, height: R.height - R.m.top - R.m.bottom, fill: COLOR.wash }, R.plot);
    for (const [series, color, dashed, name] of [[d.linear, COLOR.valid, true, '線形モデル'], [d.nn, COLOR.train, false, 'ネットワーク']]) {
      const pts = series.slice(0, i + 1).map((s, k) => [k, s.loss]);
      svg('path', { d: path(pts, R.x, R.y), stroke: color, 'stroke-width': 2, fill: 'none', 'stroke-dasharray': dashed ? '6 4' : 'none', 'stroke-linejoin': 'round' }, R.plot);
      pts.forEach(([k, v]) => {
        dot(R.over, R.x(k), R.y(v), color, { r: k === i ? 5.5 : 4, hollow: dashed });
        hoverable(svg('circle', { cx: R.x(k), cy: R.y(v), r: 11, fill: 'transparent' }, R.over), `${d.steps[k]}回：${name}の損失 ${num(v, 3)}`);
      });
    }
    d.steps.forEach((_, k) => {
      const hit = svg('rect', { x: R.x(k) - band / 2, y: R.height - R.m.bottom, width: band, height: R.m.bottom - 20, fill: 'transparent', style: 'cursor:pointer' }, R.over);
      hit.addEventListener('click', () => pick(k));
    });
  };
};

// ---- 順伝播の図（4節・6節） ---------------------------------------------------

// ReLU・sigmoid の小さなグラフ。更新時は点を曲線に沿って移動させる。
function miniCurve(parent, x0, y0, w, h, kind, points, animate = false) {
  const relu = kind === 'relu';
  const f = relu ? v => Math.max(0, v) : v => 1 / (1 + Math.exp(-v));
  const lim = relu ? [-3, 2] : [-4, 4];
  const xs = scale(lim, [x0, x0 + w]), ys = scale(relu ? [0, 2] : [0, 1], [y0 + h, y0]);
  const g = svg('g', {}, parent);
  svg('line', { x1: x0, x2: x0 + w, y1: y0 + h, y2: y0 + h, stroke: COLOR.axis }, g);
  svg('line', { x1: xs(0), x2: xs(0), y1: y0, y2: y0 + h + 3, stroke: COLOR.grid }, g);
  if (!relu) {
    svg('line', { x1: x0, x2: x0 + w, y1: ys(0.5), y2: ys(0.5), stroke: COLOR.axis, 'stroke-dasharray': '3 3' }, g);
    text(g, x0 + w, ys(0.5) + 13, '0.5', { 'text-anchor': 'end', style: 'font-size:11px' });
  }
  const curve = [...Array(81).keys()].map(k => { const v = lim[0] + k * (lim[1] - lim[0]) / 80; return [v, f(v)]; });
  svg('path', { d: path(curve, xs, ys), fill: 'none', stroke: '#8d99a6', 'stroke-width': 1.8, 'stroke-linejoin': 'round' }, g);
  text(g, x0 + 2, y0 + 10, relu ? 'ReLU' : 'sigmoid', { style: 'font-size:11px' });
  const at = p => { const v = Math.max(lim[0], Math.min(lim[1], p.v)); return [xs(v), ys(f(v))]; };
  const layer = svg('g', {}, g);
  for (const p of points) {
    const [cx, cy] = at(p);
    svg('line', { x1: cx, x2: cx, y1: y0 + h, y2: cy, stroke: p.hollow ? COLOR.muted : COLOR.train, 'stroke-dasharray': '2 2' }, layer);
    const mark = dot(layer, cx, cy, p.hollow ? COLOR.muted : COLOR.train, { r: p.hollow ? 3.5 : 4.5, hollow: p.hollow });
    if (animate && !p.hollow && points.length === 2) {
      const [fromX, fromY] = at(points[0]);
      for (const [attributeName, from, to] of [['cx', fromX, cx], ['cy', fromY, cy]]) {
        const motion = svg('animate', { attributeName, from, to, dur: '0.7s', fill: 'freeze', begin: 'indefinite' }, mark);
        motion.beginElement();
      }
    }
  }
  return layer;
}

// 項を sep でつなぎ、幅に収まるように折り返す。1行目は lead、2行目からは cont で始める。
function wrapJoin(measure, items, maxW, size, { sep = ' + ', lead = '= ', cont = '　+ ', result } = {}) {
  const lines = [];
  let line = lead;
  items.forEach((t, i) => {
    const next = line + (i ? sep : '') + t;
    if (i && measure(next, size) > maxW) { lines.push(line); line = cont + t; } else line = next;
  });
  if (result === undefined) return [...lines, line];
  const end = line + ' = ' + result;
  return measure(end, size) > maxW ? [...lines, line, '= ' + result] : [...lines, end];
}

// 4・5節の図の上部：段階（押すとその段へ移る）、いまの段の説明、次へ進むボタン。
function stepper(d, st, last, go, unlocked) {
  root.append(html('div', { class: 'steps', role: 'group', 'aria-label': '計算の段階' }, d.stages.map((label, i) =>
    html('button', { type: 'button', class: 'step' + (i === st ? ' now' : i < st ? ' done' : ''), 'aria-pressed': String(i === st),
      ...(i > unlocked ? { disabled: '' } : {}), text: `${'①②③④⑤⑥'[i]} ${label}`,
      onclick: () => { if (i <= unlocked) go(i, false); } }))));
  // 説明の長さで図が上下に動かないよう、いちばん長い説明の高さをとっておく
  const status = html('p', { class: 'status', role: 'status' });
  root.append(status);
  status.style.minHeight = Math.max(...d.status.map(t => { status.innerHTML = richHTML(t); return status.offsetHeight; })) + 'px';
  status.innerHTML = richHTML(d.status[st]);
  root.append(html('div', { class: 'actions' }, [st < last
    ? html('button', { type: 'button', class: 'btn primary', text: `次へ：${d.stages[st + 1]}`, onclick: () => go(st + 1, true) })
    : html('button', { type: 'button', class: 'btn', text: '最初から', onclick: () => go(0, false) })]));
}

const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// 4節：1部屋の順伝播を一段ずつ進める。同じ層のニューロンは同時に計算する。
// 6節（mode: 'update'）：重みを変え、それが出力に伝わる順に見せる。
FIGURES.forward = (d) => {
  const update = d.mode === 'update';
  const last = update ? 2 : d.stages.length - 1;
  const state = { stage: 0, animate: false, unlocked: 0 };
  return (width) => {
    const st = state.stage;
    const anim = state.animate && !reducedMotion();
    state.animate = false;
    const go = (stage, animate) => { state.stage = stage; state.unlocked = Math.max(state.unlocked, stage); state.animate = animate; root.replaceChildren(); FIGURES.current(root.clientWidth); fit(); };
    if (update) {
      const status = ['更新前の重みと出力です。重みを更新すると、どの値が変わるでしょうか。',
        '重みを更新しました。同じ部屋をもう一度順伝播します。',
        '変わった重みが隠れ層の出力に伝わり、予測と損失も変わりました。'];
      root.append(html('p', { class: 'status', text: status[st] }));
      root.append(html('div', { class: 'actions' }, [html('button', { type: 'button', class: 'btn' + (st < last ? ' primary' : ''),
        text: ['重みを更新', '更新後の出力を計算', '最初から見る'][st], onclick: () => go(st < last ? st + 1 : 0, st < last) })]));
      if (st === 2) root.append(legend([['ring', COLOR.muted, '更新前'], ['dot', COLOR.train, '更新後']]));
    } else stepper(d, st, last, go, state.unlocked);

    const horizontal = width >= 900;
    const F = horizontal ? 13.5 : 13, LINE = 19, PAD = 11, GW = update ? 116 : 96, GH = 48;
    const s = svg('svg', { width, height: 10, role: 'img', 'aria-label': update ? '1回の更新の前後の値' : '順伝播の計算' }, root);
    const defs = svg('defs', {}, s);
    for (const [id, color] of [['fwArrow', COLOR.axis], ['arrowBlue', COLOR.train]]) {
      const mk = svg('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
      svg('path', { d: 'M0,0L10,5L0,10Z', fill: color }, mk);
    }
    const measure = (str, size = F) => { const t = text(s, 0, -99, str, { style: `font-size:${size}px` }); const w = t.getComputedTextLength(); t.remove(); return w; };

    // 各段で何を表示するか。reveal はこの段で新しく出た値（少し遅れて現れる）
    const reveal = k => (anim && st === k ? (k === 1 || k === 3 ? 'reveal late' : 'reveal') : '');
    const inW = 112, outW = horizontal ? 290 : width;
    const gap = horizontal ? Math.max((width - inW - outW - 460) / 2, Math.min(90, (width - inW - outW - 400) / 2)) : 0;
    const hidW = horizontal ? width - inW - outW - 2 * gap : width;
    const textW = hidW - 24 - GW - 14;
    const zero = d.hidden.map(n => (update ? n.h === 0 && n.h2 === 0 : st >= 2 && n.h === 0));

    const hiddenRows = d.hidden.map((n, j) => {
      const rows = [{ t: n.name, cls: 'strong' }];
      if (update) {
        const changes = n.changes.length ? wrapJoin(measure, st === 0 ? n.beforeParams : n.afterParams, textW, F, { sep: '、', lead: '', cont: '' }) : ['勾配が0なので変わらない'];
        changes.forEach(t => rows.push({ t, cls: n.changes.length ? 'ink' : '', anim: st === 1 ? 'reveal' : '' }));
        rows.push({ t: `${n.zName} = ${st === 2 ? n.zAfter : n.zBefore}`, cls: 'ink', anim: st === 2 ? 'reveal' : '' });
      } else {
        rows.push({ t: n.sym, cls: 'ink' });
        wrapJoin(measure, n.terms, textW, F, { result: n.zText }).forEach((t, k) =>
          rows.push(st >= 1 ? { t, cls: 'ink', anim: reveal(1) } : { t: k ? '' : '= ?', cls: '' }));
      }
      return rows;
    });
    const o = d.output;
    const outRows = [{ t: '出力', cls: 'strong' }];
    const outText = outW - 24;
    if (update) {
      wrapJoin(measure, st === 0 ? o.beforeParams : o.afterParams, outText, F, { sep: '、', lead: '', cont: '' }).forEach(t => outRows.push({ t, cls: 'ink', anim: st === 1 ? 'reveal' : '' }));
      outRows.push({ t: `$s$ = ${st === 2 ? o.sAfter : o.sBefore}`, cls: 'ink', anim: st === 2 ? 'reveal' : '' });
    } else {
      outRows.push({ t: o.sym, cls: 'ink' });
      wrapJoin(measure, o.terms, outText, F, { result: o.sText }).forEach((t, k) =>
        outRows.push(st >= 3 ? { t, cls: 'ink', anim: reveal(3) } : { t: k ? '' : '= ?', cls: '' }));
    }
    const OGW = Math.min(outText, 230), OGH = 78;
    const outTail = update ? [st === 2 ? o.pAfter : o.pBefore, st === 2 ? o.verdictAfter : o.verdictBefore,
      st === 2 ? o.lossAfter : o.lossBefore] : st >= 4 ? [o.pText, o.verdict] : ['$p$ = ?', ''];
    const hH = Math.max(PAD * 2 + LINE * Math.max(...hiddenRows.map(r => r.length)), PAD + 4 + GH + 26 + PAD);
    const oH = PAD * 2 + LINE * (outRows.length + outTail.length) + OGH + 14;
    const iH = PAD * 2 + LINE * 2;

    // 箱の位置
    const box = {};
    let H;
    if (horizontal) {
      const top = 26, vgap = 12, colH = 4 * hH + 3 * vgap;
      const xs = [0, inW + gap, inW + gap + hidW + gap];
      d.hidden.forEach((_, j) => { box['h' + j] = { x: xs[1], y: top + j * (hH + vgap), w: hidW, h: hH }; });
      const mid = j => top + j * (hH + vgap) + hH / 2;
      d.inputs.forEach((_, i) => { box['i' + i] = { x: 0, y: (mid(2 * i) + mid(2 * i + 1)) / 2 - iH / 2, w: inW, h: iH }; });
      box.o = { x: xs[2], y: top + Math.max(0, (colH - oH) / 2), w: outW, h: oH };
      H = top + Math.max(colH, oH) + 4;
      [['入力', 'i0'], ['隠れ層', 'h0'], ['出力', 'o']].forEach(([t, k]) =>
        text(s, box[k].x + box[k].w / 2, 14, t, { 'text-anchor': 'middle', class: 'strong' }));
    } else {
      let y = 0;
      const heads = [];
      const head = t => { heads.push([t, y]); y += 22; };
      head('入力');
      d.inputs.forEach((_, i) => { box['i' + i] = { x: i * (width + 10) / 2, y, w: (width - 10) / 2, h: iH }; });
      y += iH + 34;
      head('隠れ層');
      d.hidden.forEach((_, j) => { box['h' + j] = { x: 0, y, w: width, h: hH }; y += hH + 10; });
      y += 24;
      head('出力');
      box.o = { x: 0, y, w: width, h: oH };
      H = y + oH + 4;
      heads.forEach(([t, hy]) => text(s, 0, hy + 15, t, { class: 'strong' }));
    }
    s.setAttribute('height', H);
    s.setAttribute('viewBox', `0 0 ${width} ${H}`);

    // 矢印。いま計算している段の矢印を青くし、値が流れる様子を点で見せる
    const edgeLayer = svg('g', {}, s);
    const pulses = [];
    const edge = (x1, y1, x2, y2, hot, faint) => {
      svg('line', { x1, y1, x2, y2, stroke: hot ? COLOR.train : faint ? '#e3e8ed' : COLOR.axis, 'stroke-width': hot ? 2 : 1.3,
        'marker-end': `url(#${hot ? 'arrowBlue' : 'fwArrow'})` }, edgeLayer);
      if (hot && anim) pulses.push([x1, y1, x2, y2]);
    };
    if (horizontal) {
      const right = b => [b.x + b.w, b.y + b.h / 2], left = b => [b.x - 2, b.y + b.h / 2];
      d.hidden.forEach((_, j) => d.inputs.forEach((_, i) => edge(...right(box['i' + i]), ...left(box['h' + j]), !update && st === 1, false)));
      d.hidden.forEach((_, j) => edge(...right(box['h' + j]), ...left(box.o), !update && st === 3 && !zero[j], zero[j]));
    } else {
      const down = (a, b, hot) => edge(width / 2, a.y + a.h + 6, width / 2, b.y - 26, hot, false);
      down(box.i0, box.h0, !update && st === 1);
      down(box.h3, box.o, !update && st === 3);
    }

    // 箱と中の文字
    const boxLayer = svg('g', {}, s);
    const frame = (b, { active = false, off = false, tint = true } = {}) => svg('rect', { x: b.x + 0.5, y: b.y + 0.5, width: b.w - 1, height: b.h - 1, rx: 9,
      fill: off ? '#f6f8fa' : tint ? '#f4f8fe' : '#fff', stroke: active ? COLOR.train : off ? '#d7dde3' : COLOR.axis, 'stroke-width': active ? 1.8 : 1.1 }, boxLayer);
    const write = (x, y, rows, maxW, off = false) => rows.forEach((r, k) => {
      if (!r.t) return;
      const size = r.cls === 'strong' ? F + 0.5 : F;
      const t = text(boxLayer, x, y + LINE * k + 14, r.t, { class: [off && r.cls !== 'strong' ? '' : r.cls, r.anim || ''].join(' ').trim(), style: `font-size:${size}px` });
      const used = t.getComputedTextLength();
      if (used > maxW) t.style.fontSize = (size * maxW / used).toFixed(1) + 'px';
    });

    d.inputs.forEach((inp, i) => {
      const b = box['i' + i];
      frame(b, { active: !update && st === 0, tint: false });
      write(b.x + 12, b.y + PAD, [{ t: `${inp.name} = ${inp.value}`, cls: 'strong' }, { t: inp.raw, cls: 'ink' }], b.w - 20);
    });
    d.hidden.forEach((n, j) => {
      const b = box['h' + j];
      frame(b, { active: !update && (st === 1 || st === 2) && !zero[j], off: zero[j] });
      write(b.x + 12, b.y + PAD, hiddenRows[j], textW, zero[j]);
      const gx = b.x + b.w - 12 - GW, gy = b.y + PAD + 4;
      const pts = update ? (st === 2 && n.z !== n.z2 ? [{ v: n.z, hollow: true }, { v: n.z2 }] : [{ v: n.z }]) : st >= 2 ? [{ v: n.z }] : [];
      const layer = miniCurve(boxLayer, gx, gy, GW, GH, 'relu', pts, update && anim && st === 2);
      if (reveal(2)) layer.setAttribute('class', reveal(2));
      const hText = update ? `${n.hName} = ${st === 2 ? n.hAfter : n.hBefore}` : st >= 2 ? `${n.hName} = ${n.hText}` : `${n.hName} = ?`;
      write(gx, gy + GH + 6, [{ t: hText, cls: (update ? st === 2 : st >= 2) && !zero[j] ? 'strong' : zero[j] ? '' : 'ink', anim: reveal(2) }], GW + 4, zero[j]);
    });
    {
      const b = box.o;
      frame(b, { active: !update && st >= 3 });
      write(b.x + 12, b.y + PAD, outRows, outText);
      const gy = b.y + PAD + LINE * outRows.length + 8;
      const pts = update ? (st === 2 ? [{ v: o.s, hollow: true }, { v: o.s2 }] : [{ v: o.s }]) : st >= 4 ? [{ v: o.s }] : [];
      const layer = miniCurve(boxLayer, b.x + 12, gy, OGW, OGH, 'sigmoid', pts, update && anim && st === 2);
      if (reveal(4)) layer.setAttribute('class', reveal(4));
      write(b.x + 12, gy + OGH + 6, outTail.map((t, k) => ({ t, cls: k === 0 && (update || st >= 4) ? 'strong' : 'ink', anim: reveal(4) })), outText);
    }

    // 値が矢印に沿って流れる点。計算の結果は、点が届いたあとに現れる
    for (const [x1, y1, x2, y2] of pulses) {
      const c = svg('circle', { r: 4.5, fill: COLOR.train, cx: 0, cy: 0 }, s);
      const m = svg('animateMotion', { dur: '0.55s', fill: 'freeze', begin: 'indefinite', path: `M${x1},${y1}L${x2},${y2}` }, c);
      m.beginElement();
      setTimeout(() => c.remove(), 650);
    }
  };
};

// 5節：1部屋の逆伝播を一段ずつ進める。損失から入力側へ戻りながら、橙の値（損失の微分）を求める。
FIGURES.backward = (d) => {
  const last = d.stages.length - 1;
  const state = { stage: 0, animate: false, unlocked: 0 };
  return (width) => {
    const st = state.stage;
    const anim = state.animate && !reducedMotion();
    state.animate = false;
    const go = (stage, animate) => { state.stage = stage; state.unlocked = Math.max(state.unlocked, stage); state.animate = animate; root.replaceChildren(); FIGURES.current(root.clientWidth); fit(); };
    stepper(d, st, last, go, state.unlocked);

    const horizontal = width >= 900;
    const F = horizontal ? 13.5 : 13, LINE = 19, PAD = 11, GW = 96, GH = 48;
    const s = svg('svg', { width, height: 10, role: 'img', 'aria-label': '逆伝播の計算' }, root);
    const defs = svg('defs', {}, s);
    for (const [id, color] of [['bwArrow', COLOR.axis], ['bwGrad', COLOR.grad]]) {
      const mk = svg('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
      svg('path', { d: 'M0,0L10,5L0,10Z', fill: color }, mk);
    }
    const measure = (str, size = F) => { const t = text(s, 0, -99, str, { style: `font-size:${size}px` }); const w = t.getComputedTextLength(); t.remove(); return w; };
    const reveal = k => (anim && st === k ? 'reveal late' : '');

    const inW = 112, outW = horizontal ? 290 : width;
    const gap = horizontal ? Math.max((width - inW - outW - 460) / 2, Math.min(90, (width - inW - outW - 400) / 2)) : 0;
    const hidW = horizontal ? width - inW - outW - 2 * gap : width;
    const textW = hidW - 24 - GW - 14, outText = outW - 24;
    const dead = d.hidden.map(n => st >= 3 && !n.active);

    // 行の数は段によらず同じにして、箱の大きさが変わらないようにする。まだ求めていない値は「?」
    const join = (items, maxW) => wrapJoin(measure, items, maxW, F, { sep: '、', lead: '', cont: '' });
    const hiddenRows = d.hidden.map((n, j) => {
      const k = '₁₂₃₄'[j];
      const grads = join(n.grads, textW);
      return [{ t: n.name, cls: 'strong' }, { t: n.value, cls: 'ink' },
        st >= 2 ? { t: n.dh, cls: 'grad', anim: reveal(2) } : { t: `∂ℓ/∂$h$${k} = ?`, cls: '' },
        st >= 3 ? { t: n.dz, cls: 'grad', anim: reveal(3) } : { t: `∂ℓ/∂$z$${k} = ?`, cls: '' },
        ...grads.map((t, i) => (st >= 4 ? { t, cls: n.active ? 'grad' : '', anim: reveal(4) } : { t: i ? '' : '重みとバイアスの勾配 = ?', cls: '' }))];
    });
    const o = d.output;
    const outGrads = join(o.grads, outText);
    const outRows = [{ t: '出力', cls: 'strong' }, ...o.values.map(t => ({ t, cls: 'ink' })),
      st >= 1 ? { t: o.ds, cls: 'grad', anim: reveal(1) } : { t: '∂ℓ/∂$s$ = ?', cls: '' },
      ...outGrads.map((t, i) => (st >= 4 ? { t, cls: 'grad', anim: reveal(4) } : { t: i ? '' : '∂ℓ/∂$v$, ∂ℓ/∂$c$ = ?', cls: '' })),
      st >= 4 ? { t: o.rest, cls: '', anim: reveal(4) } : { t: '', cls: '' }];
    const hH = Math.max(PAD * 2 + LINE * Math.max(...hiddenRows.map(r => r.length)), PAD + 4 + GH + 26 + PAD);
    const oH = PAD * 2 + LINE * outRows.length;
    const iH = PAD * 2 + LINE * 2;

    // 箱の位置（順伝播の図と同じ並べ方）
    const box = {};
    let H;
    if (horizontal) {
      const top = 26, vgap = 12, colH = 4 * hH + 3 * vgap;
      const xs = [0, inW + gap, inW + gap + hidW + gap];
      d.hidden.forEach((_, j) => { box['h' + j] = { x: xs[1], y: top + j * (hH + vgap), w: hidW, h: hH }; });
      const mid = j => top + j * (hH + vgap) + hH / 2;
      d.inputs.forEach((_, i) => { box['i' + i] = { x: 0, y: (mid(2 * i) + mid(2 * i + 1)) / 2 - iH / 2, w: inW, h: iH }; });
      box.o = { x: xs[2], y: top + Math.max(0, (colH - oH) / 2), w: outW, h: oH };
      H = top + Math.max(colH, oH) + 4;
      [['入力', 'i0'], ['隠れ層', 'h0'], ['出力と損失', 'o']].forEach(([t, k]) =>
        text(s, box[k].x + box[k].w / 2, 14, t, { 'text-anchor': 'middle', class: 'strong' }));
    } else {
      let y = 0;
      const heads = [];
      const head = t => { heads.push([t, y]); y += 22; };
      head('入力');
      d.inputs.forEach((_, i) => { box['i' + i] = { x: i * (width + 10) / 2, y, w: (width - 10) / 2, h: iH }; });
      y += iH + 34;
      head('隠れ層');
      d.hidden.forEach((_, j) => { box['h' + j] = { x: 0, y, w: width, h: hH }; y += hH + 10; });
      y += 24;
      head('出力と損失');
      box.o = { x: 0, y, w: width, h: oH };
      H = y + oH + 4;
      heads.forEach(([t, hy]) => text(s, 0, hy + 15, t, { class: 'strong' }));
    }
    s.setAttribute('height', H);
    s.setAttribute('viewBox', `0 0 ${width} ${H}`);

    // 矢印。灰色は順伝播の向き、橙は勾配を戻す向き。いまの段では橙の点が流れる
    const edgeLayer = svg('g', {}, s);
    const pulses = [];
    const edge = (x1, y1, x2, y2, faint) => svg('line', { x1, y1, x2, y2, stroke: faint ? '#e3e8ed' : COLOR.axis, 'stroke-width': 1.2,
      'marker-end': 'url(#bwArrow)' }, edgeLayer);
    const back = (x1, y1, x2, y2, now) => {
      svg('line', { x1, y1, x2, y2, stroke: COLOR.grad, 'stroke-width': now ? 2.2 : 1.5, 'stroke-dasharray': '6 4',
        'marker-end': 'url(#bwGrad)', opacity: now ? 1 : 0.55 }, edgeLayer);
      if (now && anim) pulses.push([x1, y1, x2, y2]);
    };
    if (horizontal) {
      const right = b => [b.x + b.w, b.y + b.h / 2], left = b => [b.x - 2, b.y + b.h / 2];
      d.hidden.forEach((n, j) => {
        d.inputs.forEach((_, i) => {
          const [x1, y1] = right(box['i' + i]), [x2, y2] = left(box['h' + j]);
          edge(x1, y1, x2, y2, dead[j]);
          if (st >= 4 && n.active) back(x2, y2 + 6, x1 + 2, y1 + 6, st === 4);
        });
        const [x1, y1] = right(box['h' + j]), [x2, y2] = left(box.o);
        edge(x1, y1, x2, y2, false);
        if (st >= 2) back(x2, y2 + 6, x1 + 2, y1 + 6, st === 2);
      });
    } else {
      const pair = (a, b, stage, show) => {
        const y1 = a.y + a.h + 6, y2 = b.y - 26;
        edge(width / 2 - 12, y1, width / 2 - 12, y2, false);
        if (show) back(width / 2 + 12, y2, width / 2 + 12, y1, st === stage);
      };
      pair(box.i0, box.h0, 4, st >= 4);
      pair(box.h3, box.o, 2, st >= 2);
    }

    // 箱と中の文字
    const boxLayer = svg('g', {}, s);
    const frame = (b, { active = false, off = false, tint = true } = {}) => svg('rect', { x: b.x + 0.5, y: b.y + 0.5, width: b.w - 1, height: b.h - 1, rx: 9,
      fill: off ? '#f6f8fa' : tint ? '#f4f8fe' : '#fff', stroke: active ? COLOR.grad : off ? '#d7dde3' : COLOR.axis, 'stroke-width': active ? 1.8 : 1.1 }, boxLayer);
    const write = (x, y, rows, maxW, off = false) => rows.forEach((r, k) => {
      if (!r.t) return;
      const size = r.cls === 'strong' ? F + 0.5 : F;
      const cls = off && r.cls === 'ink' ? '' : r.cls;
      const t = text(boxLayer, x, y + LINE * k + 14, r.t, { class: [cls, r.anim || ''].join(' ').trim(), style: `font-size:${size}px` });
      const used = t.getComputedTextLength();
      if (used > maxW) t.style.fontSize = (size * maxW / used).toFixed(1) + 'px';
    });

    d.inputs.forEach((inp, i) => {
      const b = box['i' + i];
      frame(b, { tint: false });
      write(b.x + 12, b.y + PAD, [{ t: `${inp.name} = ${inp.value}`, cls: 'strong' }, { t: inp.raw, cls: 'ink' }], b.w - 20);
    });
    d.hidden.forEach((n, j) => {
      const b = box['h' + j];
      frame(b, { active: (st === 2 || st === 3) || (st === 4 && n.active), off: dead[j] });
      write(b.x + 12, b.y + PAD, hiddenRows[j], textW, dead[j]);
      const gx = b.x + b.w - 12 - GW, gy = b.y + PAD + 4;
      miniCurve(boxLayer, gx, gy, GW, GH, 'relu', [{ v: n.z }]);
      write(gx, gy + GH + 6, [st >= 3 ? { t: n.slope, cls: n.active ? 'grad' : '', anim: reveal(3) } : { t: '傾き ?', cls: '' }], GW + 4);
    });
    frame(box.o, { active: st <= 1 || st === 4 });
    write(box.o.x + 12, box.o.y + PAD, outRows, outText);

    // 勾配が矢印に沿って戻る点。値は、点が届いたあとに現れる
    for (const [x1, y1, x2, y2] of pulses) {
      const c = svg('circle', { r: 4.5, fill: COLOR.grad, cx: 0, cy: 0 }, s);
      const m = svg('animateMotion', { dur: '0.55s', fill: 'freeze', begin: 'indefinite', path: `M${x1},${y1}L${x2},${y2}` }, c);
      m.beginElement();
      setTimeout(() => c.remove(), 650);
    }
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

// 文字の幅や説明の高さはフォントで変わるので、フォントの読み込み後にもう一度描く
if (document.fonts) document.fonts.ready.then(() => { if (lastWidth) { root.replaceChildren(); FIGURES.current(lastWidth); fit(); } });

let lastWidth = 0;
new ResizeObserver(() => {
  const width = root.clientWidth;
  if (width !== lastWidth) {
    lastWidth = width;
    hideTip();
    root.replaceChildren();
    FIGURES.current(width);
  }
  fit();
}).observe(document.body);
