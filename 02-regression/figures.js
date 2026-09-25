// 02 の図。Python から CONFIG = {name, data, caption} を受け取り、#root に描く。
// 幅が変わるたびに描き直し、描いた後は iframe の高さを内容に合わせる。

const NS = 'http://www.w3.org/2000/svg';
const COLOR = {
  ink: '#1f2a37', muted: '#6b7785', grid: '#e6eaef', axis: '#b5bec8', wash: '#f3f6f9',
  train: '#2a78d6', valid: '#eb6834', ghost: '#c5cdd6', bar: '#5b6b7b',
};
// 損失の地形の色。MSEが小さいほど白に近く、大きいほど濃い青。
const RAMP = ['#ffffff', '#f2f7fe', '#e4effc', '#d6e6fa', '#c8def8', '#b9d4f6', '#aacbf4',
  '#9bc1f1', '#8bb7ee', '#7cadeb', '#6da3e8', '#5e99e4'];

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

function big(v) {
  if (v >= 1e4) return '約' + Math.round(v / 1e4).toLocaleString('ja-JP') + '万';
  return v >= 100 ? Math.round(v).toString() : v.toFixed(1);
}

function power(label) {
  // "1e-6" → "10⁻⁶"
  const m = /^1e(-?\d+)$/.exec(label);
  if (!m) return label;
  const e = Number(m[1]);
  if (e === 0) return '1';
  if (e === 1) return '10';
  const sup = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴', 5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };
  return '10' + String(e).split('').map(c => sup[c]).join('');
}

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
  else if (kind === 'star') svg('path', { d: starPath(11, 6, 6), fill: color }, s);
  else if (kind === 'resid') {
    svg('line', { x1: 4, x2: 20, y1: 2, y2: 2, stroke: COLOR.ink, 'stroke-width': 1.5 }, s);
    svg('line', { x1: 12, x2: 12, y1: 2, y2: 9, stroke: color, 'stroke-width': 1.5 }, s);
    dot(s, 12, 9, COLOR.train, { r: 2.8 });
  }
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

// MSEの値を、損失の地形と同じ段の色にする
const rampColor = (levels, v) => RAMP[Math.min(RAMP.length - 1, levels.filter(l => v >= l).length)];

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

// ---- 図 -------------------------------------------------------------------

const FIGURES = {};

FIGURES.dataRoles = () => (width) => {
  const compact = width < 590;
  const W = compact ? 360 : 760;
  const H = compact ? 328 : 146;
  const s = svg('svg', { width: Math.min(width, W), height: Math.min(width, W) * H / W,
    viewBox: `0 0 ${W} ${H}`, style: 'margin:0 auto', role: 'img',
    'aria-label': '訓練データで重みを求め、検証データでモデルを選び、テストデータで最後に確かめる' }, root);
  const stages = [
    ['訓練データ', '重みを求める', COLOR.train, '#eef5fd'],
    ['検証データ', '次数・λを選ぶ', COLOR.valid, '#fff3ed'],
    ['テストデータ', '最後に確かめる', COLOR.ink, COLOR.wash],
  ];
  const boxW = compact ? 300 : 216;
  const boxH = compact ? 78 : 100;
  stages.forEach(([title, role, stroke, fill], i) => {
    const x = compact ? 30 : 8 + i * 268;
    const y = compact ? 8 + i * 108 : 12;
    svg('rect', { x, y, width: boxW, height: boxH, rx: 11, fill, stroke, 'stroke-width': 1.5 }, s);
    text(s, x + boxW / 2, y + (compact ? 31 : 40), title,
      { 'text-anchor': 'middle', class: 'strong', style: 'font-size:17px' });
    text(s, x + boxW / 2, y + (compact ? 56 : 71), role,
      { 'text-anchor': 'middle', class: 'ink', style: 'font-size:14px' });
    if (i < 2) {
      if (compact) {
        svg('path', { d: `M180,${y + boxH + 4}v20m-5,-5l5,5 5,-5`, fill: 'none',
          stroke: COLOR.muted, 'stroke-width': 1.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, s);
      } else {
        const ax = x + boxW + 7;
        svg('path', { d: `M${ax},62h36m-6,-5l6,5 -6,5`, fill: 'none',
          stroke: COLOR.muted, 'stroke-width': 1.5, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, s);
      }
    }
  });
};

FIGURES.rentScatter = (d) => (width) => {
  const c = chart(root, { width, height: Math.min(340, width * 0.62 + 40), xlim: [10, 50], ylim: [0, 12],
    xticks: [10, 20, 30, 40, 50], yticks: [0, 3, 6, 9, 12], xlabel: '広さ（㎡）', ylabel: '家賃（万円）',
    label: '30件の部屋の広さと家賃の散布図' });
  const qx = c.x(d.query);
  svg('line', { x1: qx, x2: qx, y1: c.y(0), y2: c.y(12), stroke: COLOR.ink, 'stroke-width': 1.2, 'stroke-dasharray': '4 4' }, c.plot);
  text(c.over, qx + 8, c.y(11), `${d.query}㎡の部屋は？`, { class: 'strong' });
  dots(c, d.points, COLOR.train, p => `広さ ${p[0]}㎡　家賃 ${p[1]}万円`);
};

FIGURES.pipeline = () => (width) => {
  const W = 760, H = 282;
  const s = svg('svg', { width: Math.min(width, W), height: Math.min(width, W) * H / W, viewBox: `0 0 ${W} ${H}`,
    style: 'margin:0 auto', role: 'img', 'aria-label': '入力、モデル、予測、正解、損失と、損失からパラメータへ戻る学習の流れ' }, root);
  const defs = svg('defs', {}, s);
  const marker = svg('marker', { id: 'arrow', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto-start-reverse' }, defs);
  svg('path', { d: 'M0,0L10,5L0,10Z', fill: COLOR.ink }, marker);
  const box = (x, y, w, h, title, sub, strong = false) => {
    svg('rect', { x, y, width: w, height: h, rx: 10, fill: strong ? '#eef5fd' : '#fff', stroke: strong ? COLOR.train : COLOR.axis, 'stroke-width': strong ? 2 : 1.2 }, s);
    text(s, x + w / 2, y + h / 2 - 4, title, { 'text-anchor': 'middle', class: 'strong', style: 'font-size:17px' });
    text(s, x + w / 2, y + h / 2 + 18, sub, { 'text-anchor': 'middle', style: 'font-size:13.5px' });
  };
  const arrow = (d, dashed = false) => svg('path', { d, fill: 'none', stroke: COLOR.ink, 'stroke-width': 1.6,
    'marker-end': 'url(#arrow)', 'stroke-dasharray': dashed ? '6 4' : 'none' }, s);
  box(8, 26, 138, 78, '入力 $x$', '広さ');
  box(196, 26, 208, 78, 'モデル $ŷ$ = $wx$ + $b$', 'パラメータ $w$, $b$', true);
  box(454, 26, 132, 78, '予測 $ŷ$', '式で計算した家賃');
  box(454, 142, 132, 78, '正解 $y$', '実際の家賃');
  box(632, 84, 120, 78, '損失 $L$', 'ずれを一つの数に');
  arrow('M146,65H194');
  arrow('M404,65H452');
  arrow('M586,65C612,65 612,100 630,108');
  arrow('M586,181C612,181 612,146 630,138');
  arrow('M692,162V246H300V106', true);
  text(s, 496, 272, '学習：損失が小さくなるように $w$, $b$ を直す', { 'text-anchor': 'middle', class: 'ink', style: 'font-size:14px' });
};

FIGURES.residualSquares = (d) => (width) => {
  const compact = width < 650;
  const c = chart(root, { width, height: Math.min(360, width * 0.66 + 40), xlim: [10, 50], ylim: [0, 12],
    xticks: [10, 20, 30, 40, 50], yticks: [0, 3, 6, 9, 12], xlabel: '広さ（㎡）', ylabel: '家賃（万円）',
    label: '5件の部屋と直線、残差と二乗誤差の正方形' });
  svg('path', { d: path([[10, d.w * 10 + d.b], [50, d.w * 50 + d.b]], c.x, c.y), stroke: COLOR.ink, 'stroke-width': 2.5, fill: 'none' }, c.plot);
  for (const [a, r] of d.points) {
    const p = d.w * a + d.b, px = c.x(a), side = Math.abs(c.y(p) - c.y(r)), top = Math.min(c.y(p), c.y(r));
    svg('rect', { x: px, y: top, width: side, height: side, fill: COLOR.valid, 'fill-opacity': 0.14, stroke: COLOR.valid, 'stroke-width': 1.2 }, c.plot);
    svg('line', { x1: px, x2: px, y1: c.y(p), y2: c.y(r), stroke: COLOR.ink, 'stroke-width': 1.5 }, c.plot);
    // 点から直線と反対側へ離し、ラベルを直線と重ねない。
    if (!compact) {
      const ly = r > p ? c.y(r) - 16 : c.y(r) + 22;
      text(c.over, px + (side > 16 ? side + 8 : 4), ly, `(${num(p - r)})² = ${num((p - r) ** 2)}`, { class: 'ink' });
    }
  }
  dots(c, d.points, COLOR.train, p => `広さ ${p[0]}㎡　家賃 ${p[1]}万円　予測 ${num(d.w * p[0] + d.b)}万円`);
  text(c.over, c.x(11), c.y(d.w * 11 + d.b) - 14, `$ŷ$ = ${d.w}$x$ + ${d.b}`, { class: 'ink' });
  if (compact) {
    root.append(legend(d.points.map(([a, r]) => {
      const residual = d.w * a + d.b - r;
      return ['dot', COLOR.train, `${a}㎡：(${num(residual)})² = ${num(residual ** 2)}`];
    })));
  }
};

FIGURES.landscape = (d) => {
  // stage 0: いろいろ試す → 1: 地形を表示 → 2: 谷底を表示
  const state = { current: null, trail: [], stage: 0, unlocked: 0 };
  const loss = (w, b) => d.points.reduce((s, [x, y]) => s + (w * x + b - y) ** 2, 0) / d.points.length;
  const bestLoss = loss(...d.best);
  const STEPS = ['① いろいろな $w$, $b$ を試す', '② 損失の地形を見る', '③ 谷底を確かめる'];
  const TRIES = 3;
  let ui = null;

  function go(stage) {
    state.stage = stage;
    state.unlocked = Math.max(state.unlocked, stage);
    state.current = stage === 2 ? [...d.best] : state.trail.length ? state.trail[state.trail.length - 1].slice(0, 2) : null;
    update();
  }

  function update() {
    const { left, right, bands, heading, steps, status, actions, leftLegend, rightLegend } = ui;
    const n = state.trail.length;
    steps.replaceChildren(...STEPS.map((s, i) =>
      html('button', { type: 'button', class: 'step' + (i === state.stage ? ' now' : i < state.stage ? ' done' : ''),
        'aria-pressed': String(i === state.stage), ...(i > state.unlocked ? { disabled: '' } : {}),
        html: richHTML(s), onclick: () => { if (i <= state.unlocked) go(i); } })));
    status.innerHTML = richHTML(
      state.stage === 2 ? `★が谷底です。左の直線が、30件全体でMSEが最も小さくなる直線です（MSE ${num(bestLoss)}）。`
      : state.stage === 1 ? `これが、このデータの損失の地形です。${n ? '試した点の色と見比べてください。' : '平面の点を押すと、その場所の直線とMSEが分かります。'}白い谷のどこが一番低いか、平面を押して探してから「谷底を表示」で確かめましょう。`
      : n === 0 ? '右の平面のどこかを押してください。押した場所の $w$, $b$ の直線が左に描かれ、MSEが計算されます。'
      : n < TRIES ? 'ほかの場所も試し、直線とMSEを見比べてください。試した点は、MSEが小さいほど白く、大きいほど濃い青で残ります。'
      : 'MSEが小さい（白い）点は、どのあたりに並んでいますか？ 見当がついたら、平面全体のMSEを表示してみましょう。');
    // ボタンは、押す意味がある段階でだけ出す
    const buttons = [];
    if (state.stage === 0 && n >= TRIES) buttons.push(html('button', { class: 'btn primary', type: 'button', text: '損失の地形を表示', onclick: () => go(1) }));
    if (state.stage === 1) buttons.push(html('button', { class: 'btn primary', type: 'button', text: '谷底を表示', onclick: () => go(2) }));
    if (n > 0) buttons.push(html('button', { class: 'btn', type: 'button', text: 'はじめからやり直す', onclick: () => { Object.assign(state, { current: null, trail: [], stage: 0, unlocked: 0 }); update(); } }));
    actions.replaceChildren(...buttons);
    bands.style.display = state.stage >= 1 ? '' : 'none';
    heading.textContent = state.stage >= 1 ? '損失の地形（色が濃いほどMSEが大きい）' : '傾き w と切片 b の平面';

    left.dyn.replaceChildren();
    right.dyn.replaceChildren();
    const cur = state.current;
    if (cur) {
      const [w, b] = cur;
      for (const [x, y] of d.points) {
        svg('line', { x1: left.x(x), x2: left.x(x), y1: left.y(y), y2: left.y(w * x + b), stroke: COLOR.muted, 'stroke-width': 1 }, left.dyn);
      }
      svg('path', { d: path([[0, b], [60, w * 60 + b]], left.x, left.y), stroke: COLOR.ink, 'stroke-width': 2.5, fill: 'none' }, left.dyn);
      leftLegend.replaceChildren(
        legendItem('line', COLOR.ink, `直線 $ŷ$ = ${num(w, 3)}$x$ ${b < 0 ? '−' : '+'} ${num(Math.abs(b))}`),
        legendItem('resid', COLOR.muted, '残差'),
        html('span', { class: 'key', html: `MSE <b>${num(loss(w, b))}</b>` }));
    } else {
      leftLegend.replaceChildren(html('span', { text: '右の平面を押すと、ここに直線が描かれます' }));
    }
    for (const [w, b, v] of state.trail) {
      svg('circle', { cx: right.x(w), cy: right.y(b), r: 5, fill: rampColor(d.levels, v), stroke: COLOR.muted, 'stroke-width': 1 }, right.dyn);
    }
    const atBest = cur && cur[0] === d.best[0] && cur[1] === d.best[1];
    if (cur && !atBest) dot(right.dyn, right.x(cur[0]), right.y(cur[1]), COLOR.ink, { r: 6 });
    if (state.stage === 2) {
      const [bx, by] = [right.x(d.best[0]), right.y(d.best[1])];
      svg('path', { d: starPath(bx, by, 11), fill: COLOR.valid, stroke: '#fff', 'stroke-width': 1.5 }, right.dyn);
      text(right.dyn, bx + 15, by + 5, '谷底', { class: 'strong' });
    }
    rightLegend.replaceChildren(
      legendItem('dot', COLOR.ink, cur ? `今の $w$ = ${num(cur[0], 3)}, $b$ = ${num(cur[1])}` : '今の $w$, $b$'),
      legendItem('ramp', '', '試した点（色がMSE）'),
      ...(state.stage === 2 ? [legendItem('star', COLOR.valid, '谷底（最小二乗法の解）')] : []));
  }

  function choose(w, b, remember) {
    w = Math.min(d.wlim[1], Math.max(d.wlim[0], w));
    b = Math.min(d.blim[1], Math.max(d.blim[0], b));
    state.current = [w, b];
    if (remember && !state.trail.some(([pw, pb]) => Math.hypot((w - pw) / (d.wlim[1] - d.wlim[0]), (b - pb) / (d.blim[1] - d.blim[0])) < 0.05)) {
      state.trail.push([w, b, loss(w, b)]);
    }
    update();
  }

  return (width) => {
    const steps = html('div', { class: 'steps' });
    const status = html('p', { class: 'status', role: 'status' });
    const actions = html('div', { class: 'actions' });
    root.append(html('div', { class: 'toolbar' }, [steps, actions]), status);
    const leftLegend = html('div', { class: 'legend' });
    const rightLegend = html('div', { class: 'legend' });
    const heading = html('h4');
    const leftPanel = html('div', { class: 'panel' }, [html('h4', { text: 'データと直線' }), leftLegend]);
    const rightPanel = html('div', { class: 'panel' }, [heading, rightLegend]);
    root.append(html('div', { class: 'row' }, [leftPanel, rightPanel]));
    const size = Math.min(380, Math.max(260, leftPanel.clientWidth * 0.9));

    const left = chart(leftPanel, { width: leftPanel.clientWidth, height: size, xlim: [10, 50], ylim: [0, 12], xticks: [10, 20, 30, 40, 50],
      yticks: [0, 3, 6, 9, 12], xlabel: '広さ（㎡）', ylabel: '家賃（万円）', label: 'データと、選んだ w, b の直線' });
    left.dyn = svg('g', {}, left.plot);
    dots(left, d.points, COLOR.train, p => `広さ ${p[0]}㎡　家賃 ${p[1]}万円`);

    const right = chart(rightPanel, { width: rightPanel.clientWidth, height: size, xlim: d.wlim, ylim: d.blim,
      xticks: [0.1, 0.2, 0.3], yticks: [-2, 0, 2, 4], xlabel: '傾き $w$', ylabel: '切片 $b$',
      label: '横軸 w、縦軸 b の平面。押した場所の w, b を選ぶ' });
    // 押せる範囲だと分かるよう、平面に薄い背景と枠を付ける
    const plane = svg('rect', { x: right.m.left, y: right.m.top, width: right.width - right.m.left - right.m.right,
      height: right.height - right.m.top - right.m.bottom, fill: '#f7f9fb', stroke: COLOR.axis });
    right.back.insertBefore(plane, right.back.firstChild);
    const bands = svg('g', {}, right.plot);
    d.bands.forEach((rings, i) => {
      const dd = rings.map(r => {
        let s = '';
        for (let k = 0; k < r.length; k += 2) s += (k ? 'L' : 'M') + right.x(r[k]).toFixed(1) + ',' + right.y(r[k + 1]).toFixed(1);
        return s + 'Z';
      }).join('');
      svg('path', { d: dd, fill: RAMP[i], stroke: RAMP[i], 'stroke-width': 0.6, 'fill-rule': 'evenodd' }, bands);
    });
    const pad = svg('rect', { x: right.m.left, y: right.m.top, width: right.width - right.m.left - right.m.right,
      height: right.height - right.m.top - right.m.bottom, fill: 'transparent', style: 'cursor:crosshair' }, right.over);
    right.dyn = svg('g', { 'pointer-events': 'none' }, right.over);
    const at = e => {
      const r = right.s.getBoundingClientRect();
      return [right.x.invert(e.clientX - r.left), right.y.invert(e.clientY - r.top)];
    };
    // 押した点と離した点を記録し、ドラッグ中は直線だけ動かす
    pad.addEventListener('pointerdown', e => { pad.setPointerCapture(e.pointerId); choose(...at(e), true); });
    pad.addEventListener('pointermove', e => { if (pad.hasPointerCapture(e.pointerId)) choose(...at(e), false); });
    pad.addEventListener('pointerup', e => {
      const last = state.trail[state.trail.length - 1];
      const [w, b] = at(e);
      if (last && Math.hypot((w - last[0]) / 0.3, (b - last[1]) / 8) > 0.02) choose(w, b, true);
    });
    right.s.setAttribute('tabindex', '0');
    right.s.addEventListener('keydown', e => {
      const step = { ArrowLeft: [-0.005, 0], ArrowRight: [0.005, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[e.key];
      if (!step) return;
      e.preventDefault();
      const [w, b] = state.current || [0.2, 1];
      choose(w + step[0], b + step[1], true);
    });
    ui = { left, right, bands, heading, steps, status, actions, leftLegend, rightLegend };
    update();
  };
};

FIGURES.multiFeature = (d) => {
  const state = { model: 'simple' };
  const near = p => p[2] <= d.split;
  return (width) => {
    const redraw = () => { root.replaceChildren(); FIGURES.current(root.clientWidth); fit(); };
    const models = ['simple', 'multi'];
    root.append(html('div', { class: 'controls' }, [html('div', { class: 'control' }, ['使う特徴量',
      segmented(['広さだけ', '広さ ＋ 駅徒歩'], models.indexOf(state.model), i => { state.model = models[i]; redraw(); }, '使う特徴量')])]));
    const m = d[state.model];
    const items = [legendItem('dot', COLOR.train, `駅から${d.split}分以内`), legendItem('ring', COLOR.valid, `${d.split}分より遠い`)];
    if (state.model === 'simple') items.push(legendItem('line', COLOR.ink, `予測 $ŷ$ = ${num(m.w, 3)}$x$ ${m.b < 0 ? '−' : '+'} ${num(Math.abs(m.b))}`));
    else items.push(legendItem('line', COLOR.train, `徒歩${d.examples[0]}分の部屋の予測`), legendItem('dash', COLOR.valid, `徒歩${d.examples[1]}分の部屋の予測`));
    items.push(html('span', { class: 'key', html: `MSE <b>${num(m.mse)}</b>` }));
    root.append(html('div', { class: 'legend' }, items));

    const c = chart(root, { width, height: Math.min(360, width * 0.62 + 40), xlim: [10, 50], ylim: [0, 12],
      xticks: [10, 20, 30, 40, 50], yticks: [0, 3, 6, 9, 12], xlabel: '広さ（㎡）', ylabel: '家賃（万円）',
      label: '駅からの徒歩時間で分けた部屋の広さと家賃' });
    const line = (w, b, color, dashed) => svg('path', { d: path([[10, w * 10 + b], [50, w * 50 + b]], c.x, c.y), stroke: color, 'stroke-width': 2.5,
      fill: 'none', 'stroke-dasharray': dashed ? '7 5' : 'none' }, c.plot);
    if (state.model === 'simple') line(m.w, m.b, COLOR.ink, false);
    else d.examples.forEach((minutes, i) => {
      const b = m.b + m.w2 * minutes;
      line(m.w1, b, i ? COLOR.valid : COLOR.train, i > 0);
      text(c.over, c.x(46), c.y(m.w1 * 46 + b) + (i ? 20 : -10), `徒歩${minutes}分`, { class: 'ink', 'text-anchor': 'middle' });
    });
    const describe = p => `広さ ${p[0]}㎡　徒歩 ${p[2]}分　家賃 ${p[1]}万円`;
    dots(c, d.points.filter(near), COLOR.train, describe);
    dots(c, d.points.filter(p => !near(p)), COLOR.valid, describe, { r: 4, hollow: true });
  };
};

FIGURES.temperatureLine = (d) => (width) => {
  const c = chart(root, { width, height: Math.min(340, width * 0.6 + 40), xlim: [0, 24], ylim: [4, 28],
    xticks: [0, 6, 12, 18, 24], yticks: [4, 10, 16, 22, 28], xlabel: '時刻（時）', ylabel: '気温（℃）',
    label: '10回測った気温と直線' });
  svg('path', { d: path(d.line, c.x, c.y), stroke: COLOR.ink, 'stroke-width': 2.5, fill: 'none' }, c.plot);
  dots(c, d.train, COLOR.train, p => `${num(p[0], 1)}時　${num(p[1], 1)}℃`);
};

// 気温の図の共通部分：訓練データ・検証データ・曲線を一つの座標に描く
function temperatureChart(parent, d, { width, height, train, curve, ghosts = [], truth = false, showValid = true }) {
  const c = chart(parent, { width, height, xlim: [0, 24], ylim: [2, 30], xticks: [0, 6, 12, 18, 24], yticks: [2, 9, 16, 23, 30],
    xlabel: '時刻（時）', ylabel: '気温（℃）', label: '訓練データと学習した曲線' });
  const line = (ys, attrs) => svg('path', { d: path(d.grid.map((t, i) => [t, Math.max(-100, Math.min(150, ys[i]))]), c.x, c.y),
    fill: 'none', 'stroke-linejoin': 'round', ...attrs }, c.plot);
  ghosts.forEach(g => line(g, { stroke: COLOR.ghost, 'stroke-width': 1.3 }));
  if (truth) line(d.truth, { stroke: COLOR.muted, 'stroke-width': 1.5, 'stroke-dasharray': '5 4' });
  if (showValid) dots(c, d.valid, COLOR.valid, p => `検証　${num(p[0], 1)}時　${num(p[1], 1)}℃`, { r: 3.5, hollow: true });
  line(curve, { stroke: COLOR.ink, 'stroke-width': 2.5 });
  dots(c, train, COLOR.train, p => `訓練　${num(p[0], 1)}時　${num(p[1], 1)}℃`, { r: 5 });
  return c;
}

FIGURES.compareFits = (d) => (width) => {
  root.append(html('div', { class: 'legend' }, [
    legendItem('dot', COLOR.train, '訓練データ（学習に使う）'),
    ...(d.showValid ? [legendItem('ring', COLOR.valid, '検証データ（学習に使わない）')] : []),
    legendItem('line', COLOR.ink, '学習した曲線')]));
  const boxes = d.panels.map(p => html('div', { class: 'panel' }, [html('h4', { html: `${richHTML(p.title)}<span>${richHTML(p.stats)}</span>` })]));
  root.append(html('div', { class: 'row' }, boxes));
  const height = Math.min(320, Math.max(240, boxes[0].clientWidth * 0.75));
  d.panels.forEach((p, i) => temperatureChart(boxes[i], d, { width: boxes[i].clientWidth, height, train: p.train, curve: p.curve, showValid: d.showValid }));
};

FIGURES.fitExplorer = (d) => {
  const ridge = d.mode === 'ridge';
  const state = { index: 0 };
  const labels = d.labels.map(power);
  const YMAX = 25;

  return (width) => {
    const step = d.steps[state.index];
    const redraw = () => { root.replaceChildren(); FIGURES.current(root.clientWidth); fit(); };
    const pick = i => { state.index = i; redraw(); };
    root.append(html('div', { class: 'controls' }, [html('div', { class: 'control' }, [ridge ? '罰の強さ λ' : '次数 d',
      segmented(labels, state.index, pick, ridge ? '罰の強さ' : '次数')])]));

    const leftPanel = html('div', { class: 'panel' }, [legend([
      ['dot', COLOR.train, '訓練データ'], ['ring', COLOR.valid, '検証データ'], ['line', COLOR.ink, '学習した曲線'],
      ['thin', COLOR.ghost, '測り直したデータで学習'], ['dash', COLOR.muted, '本当の変化']])]);
    const rightPanel = html('div', { class: 'panel' }, [html('div', { class: 'legend' }, [
      legendItem('line', COLOR.train, `訓練MSE <b>${num(step.train)}</b>`),
      legendItem('line', COLOR.valid, `検証MSE <b>${num(step.valid)}</b>`),
      html('span', { text: ridge ? `（λ = ${labels[state.index]} のとき）` : `（${d.labels[state.index]}次のとき）` })])]);
    root.append(html('div', { class: 'row' }, [leftPanel, rightPanel]));
    const size = Math.min(360, Math.max(250, leftPanel.clientWidth * 0.82));

    temperatureChart(leftPanel, d, { width: leftPanel.clientWidth, height: size, train: d.train, curve: step.curve, ghosts: step.ghosts, truth: true });

    // 右：設定ごとのMSE。列を押してもその設定に切り替わる
    const n = d.steps.length;
    const R = chart(rightPanel, { width: rightPanel.clientWidth, height: size, xlim: [-0.5, n - 0.5], ylim: [0, YMAX],
      xticks: [...Array(n).keys()], xfmt: i => labels[i], yticks: [0, 5, 10, 15, 20, 25],
      xlabel: ridge ? '罰の強さ $λ$' : '次数 $d$', ylabel: 'MSE（小さいほどよい）', label: '設定ごとの訓練MSEと検証MSE',
      margin: { top: 46 } });
    const band = (R.x(1) - R.x(0)) * 0.8;
    svg('rect', { x: R.x(state.index) - band / 2, y: R.m.top, width: band, height: R.height - R.m.top - R.m.bottom, fill: COLOR.wash }, R.plot);
    for (const [field, color, name] of [['train', COLOR.train, '訓練'], ['valid', COLOR.valid, '検証']]) {
      const pts = d.steps.map((s, i) => [i, Math.min(s[field], YMAX + 3)]);
      svg('path', { d: path(pts, R.x, R.y), stroke: color, 'stroke-width': 2, fill: 'none', 'stroke-linejoin': 'round' }, R.plot);
      d.steps.forEach((s, i) => {
        const v = s[field], cx = R.x(i), cy = R.y(Math.min(v, YMAX));
        if (v > YMAX) {
          svg('path', { d: `M${cx},${cy - 2}l6,10h-12Z`, fill: color }, R.over);
          text(R.over, cx, R.m.top - 6, big(v), { 'text-anchor': 'middle', class: 'ink' });
        } else dot(R.over, cx, cy, color, { r: i === state.index ? 5.5 : 4 });
        hoverable(svg('circle', { cx, cy: cy + 4, r: 11, fill: 'transparent' }, R.over), `${labels[i]}${ridge ? '' : '次'}：${name}MSE ${num(v)}`);
      });
    }
    d.steps.forEach((_, i) => {
      const hit = svg('rect', { x: R.x(i) - band / 2, y: R.height - R.m.bottom, width: band, height: R.m.bottom - 20, fill: 'transparent', style: 'cursor:pointer' }, R.over);
      hit.addEventListener('click', () => pick(i));
    });

    if (ridge) {
      const WMAX = 60;
      const panel = html('div', { style: 'margin-top:14px' }, [html('h4', { style: 'font-size:14px;margin:0 0 4px;font-weight:600', text: '学習された重みの大きさ |w₁|〜|w₉|' })]);
      root.append(panel);
      const B = chart(panel, { width: panel.clientWidth, height: 170, xlim: [0.5, 9.5], ylim: [0, WMAX], xticks: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        xfmt: j => `|w${j}|`, yticks: [0, 20, 40, 60], margin: { top: 22 }, label: '重みの大きさの棒グラフ' });
      const bw = Math.min(24, (B.x(2) - B.x(1)) * 0.55);
      step.weights.forEach((v, j) => {
        const cx = B.x(j + 1), top = B.y(Math.min(v, WMAX)), base = B.y(0), r = Math.min(4, (base - top) / 2);
        svg('path', { d: `M${cx - bw / 2},${base}V${top + r}Q${cx - bw / 2},${top} ${cx - bw / 2 + r},${top}H${cx + bw / 2 - r}Q${cx + bw / 2},${top} ${cx + bw / 2},${top + r}V${base}Z`, fill: COLOR.bar }, B.plot);
        if (v > WMAX) text(B.over, cx, B.m.top - 6, big(v), { 'text-anchor': 'middle', class: 'ink' });
        hoverable(svg('rect', { x: cx - bw, y: B.m.top, width: bw * 2, height: base - B.m.top, fill: 'transparent' }, B.over), `|w${j + 1}| = ${big(v)}`);
      });
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
