// 03 の図。Python から CONFIG = {name, data, caption} を受け取り、#root に描く。
// 幅が変わるたびに描き直し、描いた後は iframe の高さを内容に合わせる。
// 「小さな部品」は 02 の figures.js と同じもの。

const NS = 'http://www.w3.org/2000/svg';
const COLOR = {
  ink: '#1f2a37', muted: '#6b7785', grid: '#e6eaef', axis: '#b5bec8', wash: '#f3f6f9',
  train: '#2a78d6', valid: '#eb6834', ghost: '#c5cdd6', bar: '#5b6b7b',
  passTint: '#e6f0fc', failTint: '#fdeee6',
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
const signed = (v, digits = 1) => (v > 0 ? '+' : '') + num(v, digits);
const sigmoid = s => 1 / (1 + Math.exp(-s));
const range = (a, b, step) => Array.from({ length: Math.round((b - a) / step) + 1 }, (_, i) => a + i * step);

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

function starPath(x, y, r) {
  const pts = [];
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r;
    pts.push(`${(x + rr * Math.cos(a)).toFixed(1)},${(y + rr * Math.sin(a)).toFixed(1)}`);
  }
  return 'M' + pts.join('L') + 'Z';
}

// 矢印の先端。一つの svg に一度だけ定義する。
function arrowHead(s, color) {
  const id = 'arrow' + (++uid);
  const marker = svg('marker', { id, viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' },
    svg('defs', {}, s));
  svg('path', { d: 'M0,0L10,5L0,10Z', fill: color }, marker);
  return `url(#${id})`;
}

function key(kind, color) {
  const s = svg('svg', { width: 22, height: 12, 'aria-hidden': 'true' });
  if (kind === 'dot') dot(s, 11, 6, color, { r: 4.5 });
  else if (kind === 'small') dot(s, 11, 6, color, { r: 3.2 });
  else if (kind === 'ring') dot(s, 11, 6, color, { r: 4, hollow: true });
  else if (kind === 'star') svg('path', { d: starPath(11, 6, 6), fill: color }, s);
  else if (kind === 'swatch') svg('rect', { x: 3, y: 1, width: 16, height: 10, rx: 2, fill: color, stroke: COLOR.axis, 'stroke-width': 0.6 }, s);
  else if (kind === 'gap') {
    svg('line', { x1: 11, x2: 11, y1: 1, y2: 10, stroke: color, 'stroke-width': 1.5 }, s);
    dot(s, 11, 10, COLOR.train, { r: 2.4 });
  }
  else if (kind === 'arrow') {
    svg('line', { x1: 2, x2: 16, y1: 6, y2: 6, stroke: color, 'stroke-width': 1.6, 'stroke-dasharray': '3 2' }, s);
    svg('path', { d: 'M15,2L21,6L15,10Z', fill: color }, s);
  }
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

function value(label, v) {
  return html('span', { class: 'key', html: `${richHTML(label)} <b>${v}</b>` });
}

function segmented(labels, active, onpick, aria) {
  return html('div', { class: 'seg', role: 'group', 'aria-label': aria }, labels.map((label, i) =>
    html('button', { type: 'button', 'aria-pressed': String(i === active), text: label, onclick: () => onpick(i) })));
}

function button(label, onclick, primary = false) {
  return html('button', { class: 'btn' + (primary ? ' primary' : ''), type: 'button', text: label, onclick });
}

// ---- 湿度と雨の例の共通部分 --------------------------------------------------------

// 横軸は朝の湿度、縦軸は雨が降ったか（0/1）または雨の確率。
function dayChart(parent, { width, height, ylabel = '雨の確率 $p$', yticks = [0, 0.5, 1], yfmt, label = '', margin = {} }) {
  return chart(parent, { width, height, xlim: [43, 97], ylim: [-0.08, 1.1], xticks: [50, 60, 70, 80, 90],
    yticks, yfmt, xlabel: '朝の湿度（%）', ylabel, label, margin });
}

const DAY_KEYS = [['dot', COLOR.train, '雨が降った日'], ['ring', COLOR.valid, '降らなかった日']];

function dayPoints(c, points) {
  // 狭い画面では点を小さくして、隣の点と重ならないようにする
  const small = c.width < 480;
  const describe = p => `湿度 ${p[0]}%　${p[1] ? '雨が降った' : '降らなかった'}`;
  dots(c, points.filter(p => p[1] === 1), COLOR.train, describe, { r: small ? 3.5 : 4.5 });
  dots(c, points.filter(p => p[1] === 0), COLOR.valid, describe, { r: small ? 3.2 : 4, hollow: true });
}

// 雨の確率 p = σ(w(湿度 − center) + b) の曲線
function rainCurve(c, w, b, center = 0, attrs = {}) {
  return svg('path', { d: path(range(43, 97, 0.25).map(h => [h, sigmoid(w * (h - center) + b)]), c.x, c.y),
    fill: 'none', stroke: COLOR.ink, 'stroke-width': 2.5, ...attrs }, c.plot);
}

// 湿度の区間ごとの「雨が降った割合」の棒
function rateBars(c, bins, { labels = true, fill = COLOR.train, opacity = 0.18 } = {}) {
  for (const bin of bins) {
    const [x0, x1] = [c.x(bin.lo) + 2, c.x(bin.hi) - 2], top = c.y(bin.rate);
    svg('rect', { x: x0, y: top, width: x1 - x0, height: c.y(0) - top, fill, 'fill-opacity': opacity }, c.plot);
    if (!labels) continue;  // 点と重ねる図では、棒の上辺が点の行と重ならないよう塗りだけにする
    svg('line', { x1: x0, x2: x1, y1: top, y2: top, stroke: fill, 'stroke-width': 2 }, c.plot);
    text(c.over, (x0 + x1) / 2, top - 8, `${bin.rain}/${bin.days}日`, { 'text-anchor': 'middle', class: 'ink' });
  }
}

function hline(c, v, attrs = {}) {
  const [x1, x2] = [c.m.left, c.width - c.m.right];
  return svg('line', { x1, x2, y1: c.y(v), y2: c.y(v), stroke: COLOR.axis, 'stroke-dasharray': '3 4', ...attrs }, c.plot);
}

function vline(c, v, attrs = {}) {
  const [y1, y2] = [c.m.top, c.height - c.m.bottom];
  return svg('line', { x1: c.x(v), x2: c.x(v), y1, y2, stroke: COLOR.axis, 'stroke-dasharray': '3 4', ...attrs }, c.plot);
}

// ---- 図 -------------------------------------------------------------------

const FIGURES = {};

FIGURES.rainDays = (d) => (width) => {
  root.append(legend(DAY_KEYS));
  const c = dayChart(root, { width, height: Math.min(260, width * 0.4 + 80), ylabel: '', yticks: [0, 1],
    yfmt: v => (v ? '降った' : '降らず'), margin: { left: 58 }, label: '30日の朝の湿度と、その日に雨が降ったか' });
  vline(c, d.query, { stroke: COLOR.ink, 'stroke-width': 1.2, 'stroke-dasharray': '4 4' });
  // 右に入りきらない狭い画面では、線の左側に置く
  const label = text(c.over, c.x(d.query) + 8, c.y(0.5) + 5, `湿度${d.query}%の日は？`, { class: 'strong' });
  if (label.getBBox().x + label.getBBox().width > c.width - 2) {
    label.setAttribute('x', c.x(d.query) - 8);
    label.setAttribute('text-anchor', 'end');
  }
  dayPoints(c, d.points);
};

FIGURES.rainRates = (d) => (width) => {
  const c = dayChart(root, { width, height: Math.min(300, width * 0.45 + 90), ylabel: '雨が降った日の割合', yticks: [0, 0.5, 1],
    margin: { top: 36 }, label: '湿度の区間ごとの、雨が降った日の割合' });
  rateBars(c, d.bins, { opacity: 0.35 });
};

FIGURES.sigmoid = (d) => (width) => {
  const c = chart(root, { width, height: Math.min(290, width * 0.45 + 80), xlim: [-6, 6], ylim: [0, 1.05],
    xticks: [-6, -4, -2, 0, 2, 4, 6], xfmt: v => num(v, 0), yticks: [0, 0.5, 1], xlabel: 'スコア $s$',
    ylabel: '確率 $p$', label: 'sigmoid関数のグラフ' });
  hline(c, 0.5);
  svg('path', { d: path(range(-6, 6, 0.05).map(s => [s, sigmoid(s)]), c.x, c.y), stroke: COLOR.train, 'stroke-width': 2.5, fill: 'none' }, c.plot);
  // sigmoid は右上がりなので、点の左上には曲線が来ない。ラベルはそこに置く。
  for (const s of d.marks) {
    const p = sigmoid(s);
    dot(c.over, c.x(s), c.y(p), COLOR.ink);
    text(c.over, c.x(s) - 10, c.y(p) - 10, `$s$ = ${num(s, 0)} → $p$ ${s === 0 ? '=' : '≈'} ${num(p, s === 0 ? 1 : 2)}`, { 'text-anchor': 'end', class: 'ink' });
  }
};

FIGURES.rainModel = (d) => (width) => {
  root.append(legend([...DAY_KEYS, ['swatch', '#c9dcf3', '区間ごとの雨の割合（図2）'], ['line', COLOR.ink, 'モデルが出す雨の確率 $p$']]));
  const c = dayChart(root, { width, height: Math.min(330, width * 0.5 + 90), margin: { top: 44 },
    label: '30日のデータと、学習したモデルが出す雨の確率' });
  rateBars(c, d.bins, { labels: false });
  const x0 = -d.b / d.w;
  hline(c, 0.5);
  vline(c, x0, { stroke: COLOR.muted });
  text(c.over, c.x(x0) - 8, c.m.top - 10, '← 降らないと予報', { 'text-anchor': 'end' });
  text(c.over, c.x(x0) + 8, c.m.top - 10, '雨と予報 →');
  rainCurve(c, d.w, d.b);
  dayPoints(c, d.points);
  const p = sigmoid(d.w * d.query + d.b);
  dot(c.over, c.x(d.query), c.y(p), COLOR.ink, { r: 6 });
  text(c.over, c.x(d.query) - 12, c.y(p) - 10, `湿度${d.query}% → 約${Math.round(p * 100)}%`, { 'text-anchor': 'end', class: 'strong' });
};

// 凸多角形を半平面 f ≥ 0 で切り取る
function clipHalf(poly, f) {
  const out = [];
  poly.forEach((p, i) => {
    const q = poly[(i + 1) % poly.length], fp = f(p), fq = f(q);
    if (fp >= 0) out.push(p);
    if ((fp >= 0) !== (fq >= 0)) {
      const t = fp / (fp - fq);
      out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
    }
  });
  return out;
}

FIGURES.boundary2d = (d) => (width) => {
  root.append(legend([...DAY_KEYS, ['line', COLOR.ink, '決定境界（$p$ = 0.5）'],
    ['swatch', COLOR.passTint, '雨と予報する側'], ['swatch', COLOR.failTint, '降らないと予報する側']]));
  const size = Math.min(width, 480);
  const box = html('div', { style: `width:${size}px;margin:0 auto` });
  root.append(box);
  const XL = [43, 97], YL = [-9, 9];
  const c = chart(box, { width: size, height: Math.round(size * 0.9), xlim: XL, ylim: YL, xticks: [50, 60, 70, 80, 90],
    yticks: [-8, -4, 0, 4, 8], yfmt: v => signed(v, 0), xlabel: '朝の湿度 $x$₁（%）', ylabel: '気圧の変化 $x$₂（hPa、前日比）',
    margin: { left: 40 }, label: '湿度と気圧の変化の散布図と、決定境界' });
  const [w1, w2, b] = d.theta;
  const f = ([x1, x2]) => w1 * x1 + w2 * x2 + b;
  const box4 = [[XL[0], YL[0]], [XL[1], YL[0]], [XL[1], YL[1]], [XL[0], YL[1]]];
  for (const [sign, fill] of [[1, COLOR.passTint], [-1, COLOR.failTint]]) {
    svg('path', { d: path(clipHalf(box4, p => sign * f(p)), c.x, c.y) + 'Z', fill }, c.plot);
  }
  svg('path', { d: path([[XL[0], -(w1 * XL[0] + b) / w2], [XL[1], -(w1 * XL[1] + b) / w2]], c.x, c.y), stroke: COLOR.ink, 'stroke-width': 2.5 }, c.plot);
  const describe = p => `湿度 ${p[0]}%　気圧 ${signed(p[1])} hPa　${p[2] ? '雨が降った' : '降らなかった'}`;
  dots(c, d.points.filter(p => p[2] === 1), COLOR.train, describe);
  dots(c, d.points.filter(p => p[2] === 0), COLOR.valid, describe, { r: 4, hollow: true });
};

FIGURES.threeModels = (d) => (width) => {
  root.append(legend([...DAY_KEYS, ['line', COLOR.ink, 'モデルが出す雨の確率 $p$'], ['gap', COLOR.muted, '正解とのずれ']]));
  const boxes = d.models.map(m => html('div', { class: 'panel', style: 'flex-basis:220px' },
    [html('h4', { html: `${richHTML(m.title)}<span>${richHTML(m.stats)}</span>` })]));
  root.append(html('div', { class: 'row' }, boxes));
  d.models.forEach((m, i) => {
    const w = boxes[i].clientWidth;
    const c = dayChart(boxes[i], { width: w, height: Math.min(270, Math.max(210, w * 0.85)), label: `${m.title}のモデル` });
    for (const [h, y] of d.points) {
      svg('line', { x1: c.x(h), x2: c.x(h), y1: c.y(y), y2: c.y(sigmoid(m.w * (h - d.center))), stroke: COLOR.muted, 'stroke-width': 1 }, c.plot);
    }
    rainCurve(c, m.w, 0, d.center);
    dayPoints(c, d.points);
  });
};

FIGURES.logLoss = (d) => (width) => {
  const c = chart(root, { width, height: Math.min(320, Math.max(300, width * 0.45 + 90)), xlim: [0, 1], ylim: [0, 4.2],
    xticks: [0, 0.2, 0.4, 0.6, 0.8, 1], yticks: [0, 1, 2, 3, 4], xlabel: '正解に与えた確率 $q$',
    ylabel: '1日分の損失 −log $q$', label: '正解に与えた確率と、1日分の損失のグラフ' });
  svg('path', { d: path(range(0.015, 1, 0.005).map(q => [q, -Math.log(q)]), c.x, c.y), stroke: COLOR.train, 'stroke-width': 2.5, fill: 'none' }, c.plot);
  // 曲線は右下がりなので、点の右上には曲線が来ない。右端に近い点は真上に置き、はみ出す分だけ左へ寄せる。
  for (const q of d.marks) {
    const v = -Math.log(q), cx = c.x(q), cy = c.y(v);
    dot(c.over, cx, cy, COLOR.ink);
    const label = `$q$ = ${q} → ${num(v, 2)}`;
    if (q < 0.7) { text(c.over, cx + 10, cy - 10, label, { class: 'ink' }); continue; }
    const node = text(c.over, cx, cy - 22, label, { class: 'ink', 'text-anchor': 'middle' });
    const over = node.getBBox().x + node.getBBox().width - (c.width - 4);
    if (over > 0) node.setAttribute('x', cx - over);
  }
};

// 勾配降下法の1回の更新を「① 予測して損失を計算 → ② 傾きを求める → ③ w を更新する」の3段に分け、ボタン1回で1段ずつ進める。
// ③では w だけが変わり、新しい w での損失は次の①で計算する（学習のループと同じ順番）。
FIGURES.descent = (d) => {
  const N = d.states.length - 1;
  const PHASES = ['① 予測して損失を計算', '② 傾きを求める', '③ w を更新する'];
  // phase: 'start'（まだ何も計算していない）→ 'pred' → 'slope' → 'moved' → 'pred' → …
  const state = { t: 0, phase: 'start', history: [{ t: 0, phase: 'start' }], cursor: 0 };
  const NEXT = { start: 'pred', pred: 'slope', slope: 'moved', moved: 'pred' };
  const finished = () => state.t === N && state.phase === 'pred';
  let ui = null;

  function advance() {
    if (state.cursor < state.history.length - 1) {
      state.cursor += 1;
      Object.assign(state, state.history[state.cursor]);
      return;
    }
    const next = NEXT[state.phase];
    if (next === 'moved') state.t += 1;
    state.phase = next;
    state.history.push({ t: state.t, phase: state.phase });
    state.cursor += 1;
  }

  function goPhase(index) {
    const cycle = state.phase === 'moved' ? state.t - 1 : state.t;
    const phase = ['pred', 'slope', 'moved'][index];
    const target = state.history.findIndex(s => s.phase === phase && s.t === cycle + (phase === 'moved' ? 1 : 0));
    if (target < 0) return;
    state.cursor = target;
    Object.assign(state, state.history[target]);
    update();
  }

  function update() {
    const { steps, status, actions, left, right, leftLegend, rightValues } = ui;
    const { t, phase } = state;
    const cur = d.states[t], prev = t > 0 ? d.states[t - 1] : null;
    const active = { start: -1, pred: 0, slope: 1, moved: 2 }[phase];
    steps.replaceChildren(
      html('span', { class: 'step done', text: finished() ? `${N}回の更新が完了` : `更新 ${phase === 'moved' ? t : t + 1}回目` }),
      ...PHASES.map((s, i) => {
        const cycle = phase === 'moved' ? t - 1 : t;
        const p = ['pred', 'slope', 'moved'][i];
        const reached = state.history.some(h => h.phase === p && h.t === cycle + (p === 'moved' ? 1 : 0));
        return html('button', { type: 'button', class: 'step' + (i === active ? ' now' : ''), ...(!reached ? { disabled: '' } : {}),
          'aria-pressed': String(i === active), html: richHTML(s), onclick: () => goPhase(i) });
      }));

    const dir = g => (g < 0 ? '右（大きくする向き）' : '左（小さくする向き）');
    if (phase === 'start') {
      status.innerHTML = richHTML(`重み $w$ の初期値を 0 にしました。まだ何も計算していません。「${PHASES[0]}」を押してください。`);
    } else if (phase === 'pred') {
      status.innerHTML = richHTML(`<b>① 予測して損失を計算</b>：今の $w$ = ${num(cur.w, 3)} で、30日それぞれの雨の確率 $p$ を計算しました（「データと予測」の曲線）。`
        + `正解と比べた損失は $L$ = ${num(cur.loss, 3)} です（「重み w と損失 L」の●）。`
        + (t === 0 ? '$w$ = 0 では、どの日にも確率0.5を出しています。' : '')
        + (finished() ? `　${N}回の更新で、谷底（損失 ${num(d.best[1], 3)}）のすぐ近くまで来ました。` : ''));
    } else if (phase === 'slope') {
      status.innerHTML = richHTML(`<b>② 傾きを求める</b>：今の場所での損失の傾きは $L'(w)$ = ${num(cur.grad, 3)} です（「重み w と損失 L」の橙の線）。`
        + `${cur.grad < 0 ? '負' : '正'}なので、$w$ を${cur.grad < 0 ? '大きく' : '小さく'}すると損失が下がります。`);
    } else {
      status.innerHTML = richHTML(`<b>③ $w$ を更新する</b>：$w$ ← $w$ − $η L'(w)$ = ${num(prev.w, 3)} − ${d.eta} × (${num(prev.grad, 3)}) = ${num(cur.w, 3)}。`
        + `傾きと逆の${dir(prev.grad)}へ ${num(Math.abs(cur.w - prev.w), 3)} 動かしました。左の曲線は更新前に計算した予測です。新しい $w$ での予測と損失は、次の①で計算します。`);
    }

    // ボタンは、押す意味がある段階でだけ出す
    const buttons = [];
    if (!finished()) buttons.push(button(`次へ：${PHASES[{ start: 0, pred: 1, slope: 2, moved: 0 }[phase]]}`, () => { advance(); update(); }, true));
    if (state.cursor > 0) buttons.push(button('前の段階に戻る', () => {
      state.cursor -= 1;
      Object.assign(state, state.history[state.cursor]);
      update();
    }));
    if (finished()) buttons.push(button('はじめからやり直す', () => {
      Object.assign(state, { t: 0, phase: 'start', history: [{ t: 0, phase: 'start' }], cursor: 0 }); update();
    }));
    actions.replaceChildren(...buttons);

    // 左：予測の曲線は①で計算したときに変わる（③の直後は、まだ前の w の曲線のまま）
    const shownW = phase === 'moved' ? prev.w : cur.w;
    const ghostW = phase === 'pred' && prev ? prev.w : null;
    left.dyn.replaceChildren();
    if (ghostW !== null) left.dyn.appendChild(rainCurve(left, ghostW, 0, d.center, { stroke: COLOR.ghost, 'stroke-width': 2 }));
    if (phase !== 'start') left.dyn.appendChild(rainCurve(left, shownW, 0, d.center));
    leftLegend.replaceChildren(...DAY_KEYS.map(k => legendItem(...k)),
      ...(phase !== 'start' ? [legendItem('line', COLOR.ink, phase === 'moved' ? '更新前に計算した予測 $p$' : '今の $w$ での予測 $p$')] : []),
      ...(ghostW !== null ? [legendItem('line', COLOR.ghost, '1回前の予測')] : []));

    // 右：通ってきた点、今の点、傾き、動かした幅
    const g = right.dyn, ov = right.labels;
    g.replaceChildren();
    ov.replaceChildren();
    const known = d.states.slice(0, phase === 'moved' || phase === 'start' ? t : t + 1);
    if (known.length > 1) svg('path', { d: path(known.map(s => [s.w, s.loss]), right.x, right.y), stroke: COLOR.train, 'stroke-width': 1.5, fill: 'none' }, g);
    const done = phase === 'moved' ? known : known.slice(0, -1);
    done.forEach(s => dot(g, right.x(s.w), right.y(s.loss), COLOR.train, { r: 3.2 }));
    const at = phase === 'moved' ? prev : cur;
    if (phase === 'slope' || phase === 'moved') {
      // 傾きの線は、画面上で同じ長さになるようにする
      const kx = right.x(1) - right.x(0), ky = right.y(1) - right.y(0);
      const half = 40 / Math.hypot(kx, ky * at.grad);
      const [x1, y1, x2, y2] = [right.x(at.w - half), right.y(at.loss - at.grad * half), right.x(at.w + half), right.y(at.loss + at.grad * half)];
      svg('line', { x1, y1, x2, y2, stroke: COLOR.valid, 'stroke-width': 2.5, 'stroke-linecap': 'round', opacity: phase === 'moved' ? 0.35 : 1 }, g);
      if (phase === 'slope') text(ov, Math.max(x1, x2) + 6, Math.min(y1, y2) + 4, `傾き ${num(at.grad, 2)}`, { class: 'strong' });
    }
    if (phase === 'moved') {
      const y = right.y(prev.loss), [xa, xb] = [right.x(prev.w), right.x(cur.w)];
      svg('line', { x1: xa, x2: xb, y1: y, y2: y, stroke: COLOR.ink, 'stroke-width': 2, 'marker-end': right.arrow }, g);
      dot(g, xb, y, COLOR.ink, { r: 5.5, hollow: true });
      text(ov, (xa + xb) / 2, y - 12, `${signed(cur.w - prev.w, 3)}`, { 'text-anchor': 'middle', class: 'strong' });
    }
    if (phase === 'pred' || phase === 'slope') dot(g, right.x(cur.w), right.y(cur.loss), COLOR.ink, { r: 6 });
    if (phase === 'start') {
      svg('line', { x1: right.x(0), x2: right.x(0), y1: right.m.top, y2: right.height - right.m.bottom, stroke: COLOR.ink, 'stroke-dasharray': '4 4' }, g);
      text(ov, right.x(0) + 6, right.m.top + 14, '初期値 $w$ = 0', { class: 'strong' });
    }

    const items = [value('$w$ =', num(phase === 'moved' ? cur.w : at.w, 3))];
    if (phase === 'pred' || phase === 'slope') items.push(value('損失 $L$ =', num(cur.loss, 3)));
    if (phase === 'slope') items.push(value('傾き $L′(w)$ =', num(cur.grad, 3)));
    rightValues.replaceChildren(...items);
  }

  return (width) => {
    const steps = html('div', { class: 'steps' });
    const status = html('p', { class: 'status', role: 'status' });
    const actions = html('div', { class: 'actions' });
    root.append(html('div', { class: 'toolbar' }, [steps, actions]), status);
    const leftLegend = html('div', { class: 'legend' });
    const rightValues = html('div', { class: 'legend' });
    const leftPanel = html('div', { class: 'panel' }, [html('h4', { text: 'データと予測' }), leftLegend]);
    const rightPanel = html('div', { class: 'panel' }, [html('h4', { text: '重み w と損失 L' }), rightValues,
      legend([['line', COLOR.ghost, '損失 $L(w)$ の曲線'], ['star', COLOR.valid, '谷底']])]);
    root.append(html('div', { class: 'row' }, [leftPanel, rightPanel]));
    const size = Math.min(330, Math.max(250, leftPanel.clientWidth * 0.78));
    const left = dayChart(leftPanel, { width: leftPanel.clientWidth, height: size, label: 'データと、今の w での予測' });
    left.dyn = svg('g', {}, left.plot);
    left.plot.insertBefore(left.dyn, left.plot.firstChild);
    dayPoints(left, d.points);
    const right = chart(rightPanel, { width: rightPanel.clientWidth, height: size, xlim: d.wlim, ylim: d.llim,
      xticks: d.wticks, yticks: d.lticks, xfmt: v => num(v, d.wdigits), yfmt: v => num(v, 1), xlabel: '重み $w$', ylabel: '損失 $L$',
      label: '横軸が重み w、縦軸が損失 L。今の w と、その場所での傾き' });
    svg('path', { d: path(d.curve, right.x, right.y), stroke: COLOR.ghost, 'stroke-width': 3, fill: 'none' }, right.plot);
    const [bx, by] = [right.x(d.best[0]), right.y(d.best[1])];
    svg('path', { d: starPath(bx, by, 9), fill: COLOR.valid, stroke: '#fff', 'stroke-width': 1.5 }, right.plot);
    right.dyn = svg('g', {}, right.plot);
    right.labels = svg('g', {}, right.over);
    right.arrow = arrowHead(right.s, COLOR.ink);
    ui = { steps, status, actions, left, right, leftLegend, rightValues };
    update();
  };
};

// 学習率ごとの w の動きと損失の変化
FIGURES.rates = (d) => {
  const state = { index: d.initial };
  return (width) => {
    const redraw = () => { root.replaceChildren(); FIGURES.current(root.clientWidth); fit(); };
    const run = d.runs[state.index];
    const last = run.states[run.states.length - 1];
    root.append(html('div', { class: 'controls' }, [html('div', { class: 'control' }, [html('span', { html: richHTML('学習率 $η$') }),
      segmented(d.runs.map(r => String(r.eta)), state.index, i => { state.index = i; redraw(); }, '学習率')])]));
    root.append(html('div', { class: 'legend' }, [
      value(`$η$ = ${run.eta} で${run.states.length - 1}回更新した後の損失`, num(last.loss, 3)),
      html('span', { html: richHTML(`（損失の最小値は ${num(d.best[1], 3)}）`) })]));
    const leftPanel = html('div', { class: 'panel' }, [html('h4', { text: '損失の曲線の上での w の動き' }),
      legend([['line', COLOR.ghost, '損失 $L$($w$)'], ['small', COLOR.train, '更新ごとの $w$'], ['star', COLOR.valid, '谷底']])]);
    const rightPanel = html('div', { class: 'panel' }, [html('h4', { text: '更新回数と損失' }),
      legend([['line', COLOR.train, `$η$ = ${run.eta}`], ['thin', COLOR.ghost, 'ほかの学習率'], ['dash', COLOR.muted, '損失の最小値']])]);
    root.append(html('div', { class: 'row' }, [leftPanel, rightPanel]));
    const size = Math.min(320, Math.max(240, leftPanel.clientWidth * 0.75));

    const L = chart(leftPanel, { width: leftPanel.clientWidth, height: size, xlim: d.wlim, ylim: d.llim, xticks: d.wticks,
      yticks: d.lticks, xfmt: v => num(v, 1), yfmt: v => num(v, 1), xlabel: '重み $w$', ylabel: '損失 $L$',
      label: '損失の曲線と、更新ごとの w の位置' });
    svg('path', { d: path(d.curve, L.x, L.y), stroke: COLOR.ghost, 'stroke-width': 3, fill: 'none' }, L.plot);
    svg('path', { d: starPath(L.x(d.best[0]), L.y(d.best[1]), 9), fill: COLOR.valid, stroke: '#fff', 'stroke-width': 1.5 }, L.plot);
    const pts = run.states.map(s => [s.w, s.loss]);
    svg('path', { d: path(pts, L.x, L.y), stroke: COLOR.train, 'stroke-width': 1.6, fill: 'none', 'stroke-linejoin': 'round' }, L.over);
    pts.forEach(([w, l], t) => {
      const node = dot(L.over, L.x(w), L.y(l), COLOR.train, { r: t === 0 ? 4.5 : 3.2, hollow: t === 0 });
      hoverable(node, `${t}回更新後：w = ${num(w, 3)}　損失 ${num(l, 3)}`);
    });
    // 最初の数回だけ番号を付ける（近すぎる番号は省く）
    const placed = [];
    pts.slice(0, 5).forEach(([w, l], t) => {
      const [px, py] = [L.x(w), L.y(l) - 10];
      if (placed.some(([qx, qy]) => Math.hypot(px - qx, py - qy) < 18)) return;
      placed.push([px, py]);
      text(L.over, px, py, String(t), { 'text-anchor': 'middle', class: 'ink' });
    });

    const n = pts.length - 1;
    const R = chart(rightPanel, { width: rightPanel.clientWidth, height: size, xlim: [0, n], ylim: d.llim,
      xticks: range(0, n, n >= 10 ? 3 : 1), yticks: d.lticks, yfmt: v => num(v, 1), xlabel: '更新回数', ylabel: '損失 $L$',
      label: '更新回数ごとの損失' });
    hline(R, d.best[1], { stroke: COLOR.muted, 'stroke-dasharray': '4 3' });
    d.runs.forEach((r, i) => {
      if (i === state.index) return;
      svg('path', { d: path(r.states.map((s, t) => [t, s.loss]), R.x, R.y), stroke: COLOR.ghost, 'stroke-width': 1.5, fill: 'none' }, R.plot);
    });
    svg('path', { d: path(pts.map(([, l], t) => [t, l]), R.x, R.y), stroke: COLOR.train, 'stroke-width': 2.2, fill: 'none', 'stroke-linejoin': 'round' }, R.plot);
    pts.forEach(([, l], t) => dot(R.over, R.x(t), R.y(l), COLOR.train, { r: 3.2 }));
  };
};

// バッチ・ミニバッチ・SGD：エポックごとの損失（全データで測る）
FIGURES.batches = (d) => (width) => {
  root.append(legend([['line', COLOR.train, '全員の平均損失（更新するたびに測る）'], ['dash', COLOR.muted, '損失の最小値']]));
  const boxes = d.panels.map(p => html('div', { class: 'panel', style: 'flex-basis:220px' },
    [html('h4', { html: `${richHTML(p.title)}<span>${richHTML(p.stats)}</span>` })]));
  root.append(html('div', { class: 'row' }, boxes));
  d.panels.forEach((p, i) => {
    const w = boxes[i].clientWidth;
    const c = chart(boxes[i], { width: w, height: Math.min(250, Math.max(200, w * 0.8)), xlim: [0, d.epochs], ylim: d.llim,
      xticks: range(0, d.epochs, 1), yticks: d.lticks, yfmt: v => num(v, 1), xlabel: 'エポック', ylabel: '損失 $L$',
      label: `${p.title}での損失の変化` });
    hline(c, d.best, { stroke: COLOR.muted, 'stroke-dasharray': '4 3' });
    svg('path', { d: path(p.pts, c.x, c.y), stroke: COLOR.train, 'stroke-width': 1.8, fill: 'none', 'stroke-linejoin': 'round' }, c.plot);
    if (p.pts.length <= 12) p.pts.forEach(([e, l]) => dot(c.over, c.x(e), c.y(l), COLOR.train, { r: 3.2 }));
  });
};

// softmax：スコア → e^s → 合計で割る、を3列の棒で並べる
FIGURES.softmax = (d) => (width) => {
  const columns = [
    { title: '① スコア $s$', values: d.scores, digits: 1, color: COLOR.bar, lo: Math.min(0, ...d.scores), hi: Math.max(...d.scores) },
    { title: '② $eˢ$（すべて正になる）', values: d.exps, digits: 2, color: COLOR.bar, lo: 0, hi: Math.max(...d.exps) },
    { title: `③ 合計 ${num(d.exps.reduce((a, b) => a + b, 0))} で割る`, values: d.probs, digits: 3, color: COLOR.train, lo: 0, hi: 1 },
  ];
  const panels = columns.map(col => {
    const rows = col.values.map((v, k) => {
      const span = col.hi - col.lo, from = (Math.min(0, v) - col.lo) / span, size = Math.abs(v) / span;
      const track = html('div', { style: 'position:relative;height:18px;background:var(--wash);border-radius:3px' }, [
        html('div', { style: `position:absolute;top:0;bottom:0;left:${from * 100}%;width:${size * 100}%;background:${col.color};border-radius:3px` })]);
      if (col.lo < 0) track.append(html('div', { style: `position:absolute;top:-3px;bottom:-3px;left:${-col.lo / span * 100}%;border-left:1.5px solid ${COLOR.ink}` }));
      return html('div', { style: 'display:grid;grid-template-columns:3.6em 1fr 3.6em;align-items:center;gap:10px;margin:8px 0;font-size:14px' }, [
        html('span', { text: d.classes[k] }), track, html('span', { style: 'text-align:right;font-variant-numeric:tabular-nums', text: num(v, col.digits) })]);
    });
    return html('div', { class: 'panel', style: 'flex-basis:220px' }, [html('h4', { html: richHTML(col.title) }), ...rows]);
  });
  root.append(html('div', { class: 'row' }, panels));
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
  if (width !== lastWidth) {
    lastWidth = width;
    hideTip();
    root.replaceChildren();
    FIGURES.current(width);
  }
  fit();
}).observe(document.body);
