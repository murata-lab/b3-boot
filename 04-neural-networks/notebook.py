import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="04 ニューラルネットワークと逆伝播")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import inspect
    from pathlib import Path
    from uuid import uuid4
    player_channel = "nn04-" + uuid4().hex
    from dataclasses import dataclass
    from html import escape
    import base64
    import json
    import numpy as np
    return (Path, base64, dataclass, escape, inspect, json, mo, np, player_channel, uuid4,)


@app.cell(hide_code=True)
def _(Path, np):
    ROOT = Path(__file__).resolve().parent
    BLUE, ORANGE, GREEN, INK = '#2563a6', '#b45a16', '#157365', '#233649'
    X = np.array([1., 2.])
    Y = 2.
    ETA = .05
    NODES = {'x0':(60,125), 'x1':(60,245), 'h10':(255,85), 'h11':(255,185), 'h12':(255,285),
             'h20':(455,85), 'h21':(455,185), 'h22':(455,285), 'out':(655,185)}
    LABELS = {'x0':'x₁','x1':'x₂','h10':'h¹₁','h11':'h¹₂','h12':'h¹₃','h20':'h²₁','h21':'h²₂','h22':'h²₃','out':'ŷ'}
    PARAMETER_LABELS={'W1':'W¹','b1':'b¹','W2':'W²','b2':'b²','v':'v','c':'c'}
    return (BLUE, ETA, GREEN, INK, LABELS, NODES, ORANGE, PARAMETER_LABELS, ROOT, X, Y,)


@app.cell(hide_code=True)
def _(np):
    def initial_parameters():
        return {
            'W1': np.array([[1, .5], [.5, .5], [-1, .5]]),
            'b1': np.array([0, -.5, 1.]),
            'W2': np.array([[.5, .5, .5], [.5, .5, 1], [-.5, 0, .5]]),
            'b2': np.array([-.5, -2., 1.5]),
            'v': np.array([.5, .5, .5]), 'c': np.array(0.),
        }
    return (initial_parameters,)


@app.cell(hide_code=True)
def _(X, np):
    def forward(parameters, x=X, y=None):
        """Inference needs only input and parameters; an optional target adds loss."""
        z1 = parameters['W1'] @ x + parameters['b1']
        h1 = np.maximum(0, z1)
        z2 = parameters['W2'] @ h1 + parameters['b2']
        h2 = np.maximum(0, z2)
        prediction = float(parameters['v'] @ h2 + parameters['c'])
        result = dict(x=np.array(x, copy=True), z1=z1, h1=h1, z2=z2, h2=h2, prediction=prediction)
        if y is not None:
            result.update(target=float(y), loss=.5 * (prediction - y) ** 2)
        return result
    return (forward,)


@app.cell(hide_code=True)
def _(Y, np):
    def backward(parameters, cache, y=Y):
        """All gradients use the same pre-update weights and forward values."""
        dy = cache['prediction'] - y
        gh2 = dy * parameters['v']
        d2 = gh2 * (cache['z2'] > 0)
        gh1 = parameters['W2'].T @ d2
        d1 = gh1 * (cache['z1'] > 0)
        gradients = dict(W1=np.outer(d1, cache['x']), b1=d1.copy(),
                         W2=np.outer(d2, cache['h1']), b2=d2.copy(),
                         v=dy * cache['h2'], c=np.array(dy))
        signals = dict(dy=dy, gh2=gh2, d2=d2, gh1=gh1, d1=d1)
        return gradients, signals
    return (backward,)


@app.cell(hide_code=True)
def _(ETA):
    def update(parameters, gradients, eta=ETA):
        """Return new parameters without modifying weights or cached activations."""
        return {key: value - eta * gradients[key] for key, value in parameters.items()}
    return (update,)


@app.cell(hide_code=True)
def _():
    def number(value):
        value = float(value)
        if abs(value) < 5e-10:
            value = 0.
        return f'{value:.6f}'.rstrip('0').rstrip('.').replace('-', '−')
    return (number,)


@app.cell(hide_code=True)
def _(number):
    def vector(values):
        return '[' + ', '.join(number(v) for v in values) + ']ᵀ'
    return (vector,)


@app.cell(hide_code=True)
def _(np, number, vector):
    def array_text(values):
        a = np.asarray(values)
        if a.ndim == 0:
            return number(a)
        if a.ndim == 1:
            return vector(a)
        return '[ ' + '; '.join(', '.join(number(v) for v in row) for row in a) + ' ]'
    return (array_text,)


@app.cell(hide_code=True)
def _(escape):
    def table(headers, rows):
        return '<div class="nn-table"><table><thead><tr>' + ''.join(f'<th>{escape(str(v))}</th>' for v in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'
    return (table,)


@app.cell(hide_code=True)
def _(escape):
    def svg_start(title, width=720, height=360):
        return f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{escape(title)}" viewBox="0 0 {width} {height}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>'
    return (svg_start,)


@app.cell(hide_code=True)
def _(INK, escape):
    def text(x, y, value, *, size=14, color=INK, anchor='middle', weight=400):
        return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" font-size="{size}" font-weight="{weight}">{escape(str(value))}</text>'
    return (text,)


@app.cell(hide_code=True)
def _():
    def edges(parameters):
        result = []
        for layer, count, previous in ((1, 2, ['x0','x1']), (2, 3, ['h10','h11','h12'])):
            for j in range(3):
                for i in range(count):
                    result.append((previous[i], f'h{layer}{j}', parameters[f'W{layer}'][j,i]))
        result.extend((f'h2{i}', 'out', w) for i,w in enumerate(parameters['v']))
        return result
    return (edges,)


@app.cell(hide_code=True)
def _():
    def all_known():
        return {f'{kind}{layer}{j}' for layer in (1,2) for j in range(3) for kind in ('z','h')} | {'out'}
    return (all_known,)


@app.cell(hide_code=True)
def _(BLUE, LABELS, NODES, ORANGE, all_known, edges, np, number, svg_start, text):
    def network_svg(parameters, cache, known=None, active=(), active_edges=(), reverse=False, signals=None):
        """A stable layout; unfinished activations remain distinct from zero."""
        known = all_known() if known is None else set(known)
        active, active_edges = set(active), set(active_edges)
        color = ORANGE if reverse else BLUE
        parts = [svg_start('隠れ層2層・各層3ニューロン。橙の破線は勾配、青の実線は値。')]
        for x,label in ((60,'入力 2個'),(255,'隠れ層1 · 3個'),(455,'隠れ層2 · 3個'),(655,'出力 1個')):
            parts.append(text(x,23,label,size=14,weight=600))
        for start,end,w in edges(parameters):
            x1,y1=NODES[start];x2,y2=NODES[end]
            dx,dy=x2-x1,y2-y1;length=np.hypot(dx,dy)
            xa,ya=x1+31*dx/length,y1+31*dy/length
            xb,yb=x2-31*dx/length,y2-31*dy/length
            on=(start,end) in active_edges
            if reverse and on: xa,ya,xb,yb=xb,yb,xa,ya
            parts.append(f'<line x1="{xa:.2f}" y1="{ya:.2f}" x2="{xb:.2f}" y2="{yb:.2f}" stroke="{color if on else "#d7e1eb"}" stroke-width="{3 if on else 1.2}" {"stroke-dasharray=\"6 4\"" if reverse and on else ""} marker-end="url(#arrow)"/>')
            if on:
                t=.64
                lx,ly=(1-t)*x1+t*x2,(1-t)*y1+t*y2
                parts.append(f'<rect x="{lx-20:.1f}" y="{ly-11:.1f}" width="40" height="19" rx="5" fill="white"/>')
                parts.append(text(round(lx,1),round(ly+3,1),number(w),size=12,color=color,weight=600))
        for name,(x,y) in NODES.items():
            highlighted=name in active
            if name.startswith('x'): value=number(cache['x'][int(name[1])]); sub=''; gradient=''
            elif name=='out':
                value=number(cache['prediction']) if name in known else '';sub='';gradient=''
                if signals and 'dy' in signals: gradient='∂L/∂ŷ = '+number(signals['dy'])
            else:
                layer,j=int(name[1]),int(name[2]); zkey=f'z{layer}{j}'
                value=number(cache[f'h{layer}'][j]) if name in known else ''
                zv=number(cache[f'z{layer}'][j]) if zkey in known else ''
                sub=f'z{str(layer).translate(str.maketrans("12","¹²"))}{str(j+1).translate(str.maketrans("123","₁₂₃"))} = {zv}' if zkey in known else ''; gradient=''
                if signals and f'd{layer}' in signals:gradient='∂L/∂z = '+number(signals[f'd{layer}'][j])
            fill='#fff3e9' if highlighted and reverse else '#edf5ff' if highlighted else '#fff'
            parts.append(f'<circle cx="{x}" cy="{y}" r="30" fill="{fill}" stroke="{color if highlighted else "#9db1c5"}" stroke-width="{2.8 if highlighted else 1.5}"/>')
            parts.append(text(x,y-8 if value else y+5,LABELS[name],size=13 if value else 17,color='#52687c'))
            if value:parts.append(text(x,y+13,value,size=15,weight=600))
            if sub:parts.append(text(x,y+45,sub,size=12))
            if gradient:parts.append(text(x,y+60,gradient,size=11,color=ORANGE,weight=600))
        parts.append(text(12,351,'破線 ← 勾配（重みは固定）' if reverse else '実線 → 値（辺の数値は重み）',size=13,color=color,anchor='start'))
        parts.append('</svg>')
        return ''.join(parts)
    return (network_svg,)


@app.cell(hide_code=True)
def _(BLUE, svg_start, text):
    def relu_svg():
        p=[svg_start('ReLUは負の入力を0にし、正の入力をそのまま通す。',720,255)]
        p.append('<path d="M50 195 H350 M170 225 V25" fill="none" stroke="#9db1c5" stroke-width="1.5"/>')
        p.append(f'<path d="M55 195 H170 L330 35" fill="none" stroke="{BLUE}" stroke-width="4"/>')
        for x,y,s in [(170,215,'0'),(338,215,'z'),(150,30,'h'),(77,181,'負なら0'),(277,68,'正ならそのまま')]:p.append(text(x,y,s,size=14))
        p.extend([text(400,65,'h = max(0, z)',size=23,anchor='start',weight=600),text(400,115,'z = −1  →  h = 0',size=18,anchor='start'),text(400,150,'z = 2   →  h = 2',size=18,anchor='start'),text(400,200,'0のところで、直線が折れる。',size=15,anchor='start')])
        return ''.join(p)+'</svg>'
    return (relu_svg,)


@app.cell(hide_code=True)
def _(svg_start, text):
    def neuron_svg():
        p=[svg_start('入力x₁とx₂が重み付きの矢印で総和Σへ入り、上から定数1が重みbとして加わる。総和をReLUに通して出力hを得る。',720,310)]
        # A bias is an additional input, shown separately from the weighted inputs.
        p.append('<g fill="none" stroke="#2563a6" stroke-width="2" marker-end="url(#arrow)"><path d="M118 105 L279 153"/><path d="M118 225 L279 177"/><path d="M315 62 V129"/><path d="M351 165 H462"/><path d="M538 165 H638"/></g>')
        for x,y,r,label in [(85,95,32,'x₁'),(85,235,32,'x₂'),(315,165,36,'Σ'),(500,165,36,'ReLU'),(675,165,30,'h¹₁')]:
            p.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#fff" stroke="#9db1c5" stroke-width="1.8"/>')
            p.append(text(x,y+6,label,size=26 if label=='Σ' else 18,weight=600))
        p.append('<circle cx="315" cy="40" r="22" fill="#f5f8fc" stroke="#9db1c5" stroke-width="1.5"/>')
        p.append(text(315,46,'1',size=18,weight=600))
        p.append(text(362,94,'b¹₁ = 0',size=16))
        p.append(text(315,12,'バイアス用の入力',size=13))
        p.append(text(192,106,'W¹₁₁ = 1',size=16))
        p.append(text(192,239,'W¹₁₂ = 0.5',size=16))
        p.append(text(404,147,'z¹₁',size=16))
        p.append(text(85,143,'1',size=15))
        p.append(text(85,283,'2',size=15))
        p.append(text(675,216,'2',size=15))
        p.append(text(430,273,'z¹₁ = 1×1 + 0.5×2 + 0×1 = 2',size=16))
        p.append(text(500,301,'h¹₁ = max(0, 2) = 2',size=16))
        return ''.join(p)+'</svg>'
    return (neuron_svg,)


@app.cell(hide_code=True)
def _(ROOT, base64, json):
    def digits_html():
        data=json.loads((ROOT/'assets/manifest.json').read_text())
        cards=[]
        for row in data['images']:
            content=base64.b64encode((ROOT/'assets'/row['file']).read_bytes()).decode()
            cards.append(f'<figure><img width="84" height="84" src="data:image/png;base64,{content}" alt="MNIST訓練画像 {row["index"]}、正解 {row["label"]}"/><figcaption>正解 {row["label"]}</figcaption></figure>')
        return '<div class="nn-digits">'+''.join(cards)+'</div>'
    return (digits_html,)


@app.cell(hide_code=True)
def _(dataclass):
    @dataclass(frozen=True)
    class Frame:
        stage: str
        title: str
        equation: str
        explanation: str
        figure: str
        detail: str = ''

        def as_dict(self):
            return dict(stage=self.stage,title=self.title,equation=self.equation,explanation=self.explanation,figure=self.figure,detail=self.detail)
    return (Frame,)


@app.cell(hide_code=True)
def _(number):
    def sum_expression(weights, values, bias):
        return ' + '.join(f'({number(w)})×({number(v)})' for w,v in zip(weights,values)) + f' + ({number(bias)})'
    return (sum_expression,)


@app.cell(hide_code=True)
def _(Frame, forward, initial_parameters, network_svg, number, sum_expression):
    def forward_frames(parameters=None, *, target=None):
        p=initial_parameters() if parameters is None else parameters
        cache=forward(p,y=target);known=set(); frames=[]
        frames.append(Frame('推論','入力を受け取る','x = [1, 2]ᵀ','重みを固定して、入力から予測を作ります。正解はまだ使いません。',network_svg(p,cache,known,('x0','x1'))))
        for layer in (1,2):
            source=cache['x'] if layer==1 else cache['h1']
            previous=['x0','x1'] if layer==1 else ['h10','h11','h12']
            for j in range(3):
                name=f'h{layer}{j}';zname=f'z{layer}{j}';z=cache[f'z{layer}'][j];h=cache[f'h{layer}'][j]
                layerlabel='第一' if layer==1 else '第二'; symbol='¹' if layer==1 else '²';sub=str(j+1).translate(str.maketrans('123','₁₂₃'))
                edge=[(prev,name) for prev in previous]
                known.add(zname)
                formula=f'z{symbol}{sub} = '+sum_expression(p[f'W{layer}'][j],source,p[f'b{layer}'][j])+f' = {number(z)}'
                frames.append(Frame('推論',f'{layerlabel}隠れ層 · ニューロン{j+1}の和',formula,'入ってくる値に各重みを掛け、足し合わせ、バイアスを加えます。丸の下のzが計算されました。',network_svg(p,cache,known,[name],edge)))
                known.add(name)
                frames.append(Frame('推論',f'{layerlabel}隠れ層 · ニューロン{j+1}のReLU',f'h{symbol}{sub} = max(0, {number(z)}) = {number(h)}','負の値なので、次の層へ渡す値は0です。' if z<0 else '正の値なので、そのまま次の層へ渡します。重みは変わりません。',network_svg(p,cache,known,[name])))
        known.add('out')
        frames.append(Frame('推論','予測ができた','ŷ = '+sum_expression(p['v'],cache['h2'],p['c'])+f' = {number(cache["prediction"])}','出力層は重み付き和をそのまま予測にします。ここで推論は完了です。',network_svg(p,cache,known,['out'],[(f'h2{i}','out') for i in range(3)])))
        return tuple(frames)
    return (forward_frames,)


@app.cell(hide_code=True)
def _(Frame, Y, array_text, backward, escape, forward, gradient_table, initial_parameters, network_svg, table, vector):
    def backward_frames():
        p=initial_parameters();c=forward(p,y=Y);g,s=backward(p,c)
        path=[('h20','out'),('h10','h20'),('x0','h10')]
        def frame(stage,title,equation,explanation,active=(),edge=(),signals=None,detail=''):
            return Frame(stage,title,equation,explanation,network_svg(p,c,active=active,active_edges=edge,reverse=stage!='損失',signals=signals),detail)
        frames=[frame('損失','正解と比較する','ŷ = 1.5, y = 2 → L = ½(1.5 − 2)² = 0.125','ここで初めて正解yを使います。損失は予測のずれを測る一つの数です。',['out']),
                frame('逆伝播','損失から始める','∂L/∂ŷ = ŷ − y = −0.5','ŷを少し増やすと損失が減るので、この微分は負です。これは予測値ではなく、損失の変わり方です。',['out'],signals={'dy':s['dy']})]
        factors=[('出力の重みを通る','∂ŷ/∂h²₁ = v₁ = 0.5','−0.5 × 0.5', ['out','h20'],path[:1]),
                 ('第二隠れ層のReLUを通る','∂h²₁/∂z²₁ = 1 （z²₁ = 1.5 > 0）','−0.5 × 0.5 × 1',['h20'],path[:1]),
                 ('層と層をつなぐ重みを通る','∂z²₁/∂h¹₁ = W²₁₁ = 0.5','−0.5 × 0.5 × 1 × 0.5',['h20','h10'],path[:2]),
                 ('第一隠れ層のReLUを通る','∂h¹₁/∂z¹₁ = 1 （z¹₁ = 2 > 0）','−0.5 × 0.5 × 1 × 0.5 × 1',['h10'],path[:2]),
                 ('入力側の重みにたどり着く','∂z¹₁/∂W¹₁₁ = x₁ = 1','−0.5 × 0.5 × 1 × 0.5 × 1 × 1 = −0.125',['h10','x0'],path)]
        for title,local,product,active,edge in factors:
            frames.append(frame('連鎖律',title,local,'一本の経路をさかのぼり、局所微分を掛けます。この経路からの寄与を求めている段階です。',active,edge,detail=f'<div class="contribution">この経路の積：{escape(product)}</div>'))
        frames += [
            frame('逆伝播','出力の全パラメータへ','∂L/∂v = (−0.5)h² = [−0.75, −0.25, −0.5]ᵀ ／ ∂L/∂c = −0.5','入力h²を掛けてvの勾配を求めます。バイアスcの局所微分は1。勾配が求まっても、まだ更新しません。',['out'],[(f'h2{i}','out') for i in range(3)],{'dy':s['dy']}),
            frame('逆伝播','第二隠れ層の3個へ戻る','∂L/∂h² = (−0.5)v = [−0.25, −0.25, −0.25]ᵀ','出力を作ったときの重みvを使います。逆伝播で伝えるのは損失の微分です。',['h20','h21','h22'],[(f'h2{i}','out') for i in range(3)],{'dy':s['dy']}),
            frame('逆伝播','第二隠れ層のReLUを通る','δ² = ∂L/∂z² = [−0.25, −0.25, −0.25]ᵀ','この入力ではz²が3個とも正なので、ReLUの局所微分はすべて1です。戻ってきた勾配に1を掛けます。',['h20','h21','h22'],signals={'d2':s['d2']}),
            frame('逆伝播','第二隠れ層の重みの勾配','∂L/∂W² = δ²(h¹)ᵀ ／ ∂L/∂b² = δ²','例：∂L/∂W²₁₁ = (−0.25)×2 = −0.5。各行は行き先のδ、各列は入力元のhに対応します。',['h20','h21','h22'],[(f'h1{i}',f'h2{j}') for i in range(3) for j in range(3)],{'d2':s['d2']},table(['パラメータ','勾配'],[('W²',array_text(g['W2'])),('b²',vector(g['b2']))])),
            frame('逆伝播','分かれた経路の寄与を足す','∂L/∂h¹₁ = (−0.25)×0.5 + (−0.25)×0.5 + (−0.25)×(−0.5) = −0.125','h¹₁は第二隠れ層の3個すべてに使われています。戻ってくる3経路の寄与−0.125、−0.125、＋0.125を足します。最初の一本だけでは勾配は完成しません。',['h10'],[('h10',f'h2{i}') for i in range(3)],{'d2':s['d2']},'<div class="contribution">第一隠れ層全体：∂L/∂h¹ = [−0.125, −0.25, −0.5]ᵀ</div>'),
            frame('逆伝播','第一隠れ層でもReLUを通る','δ¹ = ∂L/∂z¹ = [−0.125, −0.25, −0.5]ᵀ','各ニューロンで、合計した勾配にReLUの局所微分を掛けます。この入力では3個とも正なので、局所微分はすべて1です。',['h10','h11','h12'],signals={'d2':s['d2'],'d1':s['d1']}),
            frame('逆伝播','入力側の重みの勾配が求まる','∂L/∂W¹ = δ¹xᵀ ／ ∂L/∂b¹ = δ¹','W¹₁₁の勾配は(−0.125)×1 = −0.125です。全3経路を足した結果です。この例では第二・第三経路の寄与が打ち消し合います。',['h10','h11','h12'],[(f'x{i}',f'h1{j}') for i in range(2) for j in range(3)],s,table(['パラメータ','勾配'],[('W¹',array_text(g['W1'])),('b¹',vector(g['b1']))])),
            frame('逆伝播','勾配がすべて揃った','∇L = すべてのパラメータに対する偏微分','ここまでは微分を計算しただけです。入力・重み・予測は更新前のまま。次に、全パラメータを一緒に更新します。',signals=s,detail=gradient_table(p,g)),
        ]
        return tuple(frames)
    return (backward_frames,)


@app.cell(hide_code=True)
def _(ETA, PARAMETER_LABELS, array_text, number, table):
    def gradient_table(p,g,with_update=False):
        headers=['パラメータ','更新前','勾配'] + ([f'更新量 −{number(ETA)}∇L','更新後'] if with_update else [])
        rows=[]
        for key in p:
            row=[PARAMETER_LABELS[key],array_text(p[key]),array_text(g[key])]
            if with_update:row += [array_text(-ETA*g[key]),array_text(p[key]-ETA*g[key])]
            rows.append(row)
        return table(headers,rows)
    return (gradient_table,)


@app.cell(hide_code=True)
def _(number):
    def metrics(cache,known,show_loss=False):
        pred=number(cache['prediction']) if 'out' in known else ''
        loss=number(cache['loss']) if show_loss else ''
        return f'<div class="metrics"><span>入力 <b>[1, 2]ᵀ</b></span><span>正解 <b>2</b></span><span>予測 <b>{pred}</b></span><span>損失 <b>{loss}</b></span></div>'
    return (metrics,)


@app.cell(hide_code=True)
def _(Frame, Y, all_known, backward, forward, forward_frames, gradient_table, initial_parameters, metrics, network_svg, update):
    def update_frames():
        p=initial_parameters();old=forward(p,y=Y);g,_=backward(p,old);newp=update(p,g);new=forward(newp,y=Y)
        inference=forward_frames(newp)
        left=network_svg(p,old)+metrics(old,all_known(),True)
        def pair(right, left_figure=left):
            return '<div class="comparison-scroll"><div class="comparison"><section><h3>左 · 更新前</h3>'+left_figure+'</section><section><h3>右 · 更新後</h3>'+right+'</section></div></div>'
        frames=[Frame('更新','全パラメータを一緒に更新する','θ_new = θ_old − 0.05∇L','左は更新前のまま残します。右は更新後の重みで計算する準備ができました。これから中間値を順に求めます。',pair(network_svg(newp,new,set())+metrics(new,set())),gradient_table(p,g,True))]
        for i,f in enumerate(inference):
            if i == 0:
                active, edge = ['x0', 'x1'], []
            elif i == len(inference)-1:
                active, edge = ['out'], [(f'h2{j}', 'out') for j in range(3)]
            else:
                layer, j = 1+(i-1)//6, ((i-1)%6)//2
                name = f'h{layer}{j}'
                sources = ['x0','x1'] if layer == 1 else ['h10','h11','h12']
                active, edge = [name], [(src,name) for src in sources] if (i-1)%2 == 0 else []
            matching_left = network_svg(p, old, active=active, active_edges=edge)+metrics(old,all_known(),True)
            right=f.figure+metrics(new,{'out'} if i==len(inference)-1 else set())
            frames.append(Frame('更新後の推論',f.title,f.equation,'同じ入力を、新しい重みで計算し直します。左の更新前と、右の対応する値を見比べてください。' if i<len(inference)-1 else '予測は1.5から1.962へ変わりました。正解2との差は、損失を再計算して確認します。',pair(right, matching_left)))
        frames.append(Frame('更新後の損失','予測のずれが小さくなった','L_new = ½(1.962 − 2)² = 0.000722','予測が正解2に近づき、ずれの絶対値は0.5から0.038へ小さくなりました。これはこの一回の更新結果です。どんな学習率でも減るとは限りません。',pair(network_svg(newp,new)+metrics(new,all_known(),True))))
        return tuple(frames)
    return (update_frames,)


@app.cell(hide_code=True)
def _(PARAMETER_LABELS, array_text, initial_parameters, table):
    def initial_table():
        p=initial_parameters()
        return table(['パラメータ','値（行列は ; で行を区切る）','形'],[(PARAMETER_LABELS[k],array_text(v),'スカラー' if v.ndim==0 else ' × '.join(str(i) for i in v.shape)) for k,v in p.items()])
    return (initial_table,)


@app.cell(hide_code=True)
def _(ROOT, escape, json):
    def player_html(frames, title, channel):
        payload=json.dumps([f.as_dict() for f in frames],ensure_ascii=False).replace('<','\\u003c')
        template=(ROOT/'player.html').read_text()
        return template.replace('__TITLE__',escape(title)).replace('__CHANNEL__',json.dumps(channel)).replace('__FRAMES__',payload).replace('__INITIAL__',frames[0].figure)
    return (player_html,)


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""<style>
    .nn-hero{border-top:5px solid #2563a6;background:#f1f6fb;border-radius:12px;padding:28px 32px;color:#233649;line-height:1.75}.nn-hero .eyebrow{color:#2563a6;font-size:13px;letter-spacing:.13em;font-weight:700}.nn-hero h1{font-size:30px;line-height:1.45;margin:8px 0 12px}.nn-hero p{margin:0}.nn-figure{border:1px solid #dbe4ed;border-radius:14px;padding:12px;background:#fff;color:#233649;margin:14px 0;overflow-x:auto}.nn-figure svg{width:100%;min-width:580px;display:block;font-family:ui-sans-serif,system-ui,sans-serif}.nn-digits{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;background:#f5f8fc;border-radius:14px;padding:20px}.nn-digits figure{margin:0;text-align:center;color:#52687c;font-size:13px}.nn-digits img{width:100%;max-width:84px;height:auto;border-radius:7px;image-rendering:pixelated}.nn-table{overflow:auto;margin:10px 0}.nn-table table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}.nn-table td,.nn-table th{padding:9px 12px;border-bottom:1px solid #dbe4ed;text-align:left;white-space:nowrap}.nn-table th{background:#f5f8fc}.nn-note{border-left:4px solid #157365;padding:12px 18px;background:#eff8f5;color:#233649;line-height:1.8}@media(max-width:600px){.nn-hero{padding:18px}.nn-hero h1{font-size:25px}.nn-digits{gap:7px;padding:12px}.nn-figure{padding:3px}}
    </style><div class="nn-hero"><div class="eyebrow">B3 MACHINE LEARNING · 04</div><h1>ニューラルネットワークと逆伝播</h1><p>画像から数字を読む。<br>予測を作る計算と、学習する計算をたどる。</p></div>""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    この章では、手書き数字の認識を入口に、ニューラルネットワーク（NN）が**どう予測し、どう重みを学習するか**を見ていきます。
    文章・数式・図で説明を読み、計算の流れはコマ送りや再生で追えます。コードの入力・書き換えは不要です。

    **目安は約60分**です。①手書き数字とNNの利点 → ②構造と活性化 → ③順伝播 → ④損失と逆伝播 → ⑤更新と再推論、と進みます。
    03の「重み付き和から予測を作る」「損失の勾配と反対向きに重みを更新する」を前提にします。連鎖律はこの章で説明します。

    最後に、**順伝播は予測を作る、逆伝播は勾配を求める、更新は重みを変える**という違いを説明できることが目標です。
    再生は最初は止まっています。本文と図を読んでから、一つずつ進めてください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 手書き数字を認識したい

    次の画像は、手書き数字のデータセット**MNIST**に含まれる実際の画像です。上段も下段も、それぞれ同じ数字です。
    同じ「3」でも、線の太さや曲がり方、傾きが違います。「5」と区別するには、一つの画素の明るさだけでは足りません。

    やりたいことは、**画像を入力して、0〜9のどの数字かを予測すること**です。
    """)
    return


@app.cell(hide_code=True)
def _(digits_html, mo):
    mo.Html(digits_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    上段の正解は3、下段の正解は5です。MNISTの訓練データから各数字の最初の6枚を、そのままの順番で表示しています。
    ここでは書き方の違いを見ています。モデルの予測結果や正解率の比較ではありません。
    [MNISTの出典](https://yalecun.org/exdb/mnist/index.html) · 画像：Yann LeCun / Corinna Cortes、[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)。

    ### 線形モデルだと、何を計算する？

    一枚の画像は28×28画素です。画素の明るさを0〜1の数にして並べると、**784個の数を持つ入力ベクトル** $\boldsymbol{x}$ になります。
    03の多クラス分類と同じように、数字 $k$ ごとにスコアを作れます。

    $$s_k = \boldsymbol{w}_k^\mathsf{T}\boldsymbol{x}+b_k
    =\sum_{i=1}^{784}w_{ki}x_i+b_k \qquad (k=0,\ldots,9)$$

    $x_i$ は $i$ 番目の画素の値、$w_{ki}$ は数字 $k$ のスコアに対するその画素の重み、$b_k$ はバイアスです。
    10個のスコアをsoftmaxで確率へ変換し、最大のクラスを予測します。

    $$\text{画像}\ \longrightarrow\ \boldsymbol{x}\ (784\text{個})\
    \longrightarrow\ s_0,\ldots,s_9\ \longrightarrow\ \text{確率}\ \longrightarrow\ \text{予測した数字}$$

    **線形モデルでも、数字の分類はできます。** ただし、ある画素を明るくしたときの各スコアへの寄与は、その重みで固定されています。
    例えば「別の場所にも線があるときだけ、この部分が強い手掛かりになる」という組み合わせを、元の画素の重み付き和で直接表すことには制約があります。
    書き方によって線の位置や形が変わる画像では、もっと複雑な入力と出力の関係を扱いたくなります。

    02の多項式のように、非線形に変換した特徴量を線形モデルへ渡す方法もあります。次元を増やすこと自体ではなく、**どんな関係を表せる変換を使うか**がポイントです。

    ### ニューラルネットワークには、こんな利点がある

    - **複雑な非線形の関係を表現できる。** 変換を重ね、入力の組み合わせによって出力が変わる関係を扱います。
    - **モデル全体をデータに合わせて学習できる。** 入力から予測までの途中の変換も、損失を小さくするよう調整します。
    - **画像・音声などで高い性能を実現してきた。** 大量のデータと計算資源を活用でき、実際の認識の課題で成果があります。

    例えばImageNetの画像認識では、深い畳み込みニューラルネットワーク（CNN）が従来の手法を大きく上回る結果を示しました。
    これは画像の構造を利用する大規模なモデルの実績です。この章の小さな説明用NNと同じ性能を意味するわけではありません。
    [ImageNetの研究](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) · [Deep learningの総説](https://doi.org/10.1038/nature14539)。読むのは任意です。

    では、NNはどんな計算をし、どうやって学習するのでしょうか。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 小さな計算を、層にして重ねる

    NNの基本となる計算単位を**ニューロン**と呼びます。一つのニューロンは、入力に重みを掛けて足し、バイアスを加え、**活性化関数**を通します。

    $$z_j=\sum_i W_{ji}x_i+b_j,\qquad h_j=\operatorname{ReLU}(z_j)=\max(0,z_j)$$

    $z_j$ は活性化の**前**の値、$h_j$ は活性化の**後**の値です。$j$ は行き先のニューロン、$i$ は入力元を表します。
    重み $W_{ji}$ とバイアス $b_j$ は、学習で変えるパラメータです。

    例えば、入力が1と2、重みが1と0.5、バイアスが0なら、次の計算になります。
    図では入力を丸、重みを矢印のラベルで表します。バイアスは、上から入る定数1に重みを掛ける形で示します。$b\times1=b$ なので、式のバイアスを足すことと同じです。
    """)
    return


@app.cell(hide_code=True)
def _(mo, neuron_svg):
    mo.Html('<div class="nn-figure">'+neuron_svg()+'</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ReLUは、どんな変換？

    **ReLU（レルー）**は、負の値を0にし、正の値をそのまま返す関数です。

    $$\operatorname{ReLU}(z)=\begin{cases}0 & z\leq 0\\z & z>0\end{cases}$$

    入力が−1なら出力は0、入力が2なら出力は2。0のところでグラフが折れ曲がります。
    """)
    return


@app.cell(hide_code=True)
def _(mo, relu_svg):
    mo.Html('<div class="nn-figure">'+relu_svg()+'</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### なぜ、重み付き和だけを重ねないのか

    活性化関数を入れず、二つの層を重ねると、

    $$\boldsymbol{h}^{(1)}=W^{(1)}\boldsymbol{x}+\boldsymbol{b}^{(1)}$$

    $$\boldsymbol{h}^{(2)}=W^{(2)}\boldsymbol{h}^{(1)}+\boldsymbol{b}^{(2)}
    =(W^{(2)}W^{(1)})\boldsymbol{x}+W^{(2)}\boldsymbol{b}^{(1)}+\boldsymbol{b}^{(2)}$$

    となります。$W^{(2)}W^{(1)}$ を新しい重み、それ以外を新しいバイアスと見れば、**一つの重み付き和にまとめられます**。
    バイアスを含むので厳密にはアフィン変換ですが、ここでは03の線形モデルと同じ種類の計算です。
    ニューロンの数を増やしても、線形な変換だけを重ねる場合は同じです。

    ReLUのような非線形の活性化を間に入れると、一般にはこのように一つの線形変換へまとめられません。
    入力によってどのニューロンが0になるかも変わり、モデル全体として複雑な関係を表せるようになります。

    ### 隠れ層2層、各層3ニューロン

    入力と出力の間にある層を**隠れ層（hidden layer）**と呼びます。この章の動く例では、隠れ層は2層、それぞれ3ニューロンです。
    上付きの $(1),(2)$ は層の番号で、二乗ではありません。

    $$\boldsymbol{x}\ \longrightarrow\ \boldsymbol{h}^{(1)}\ (3\text{個})\
    \longrightarrow\ \boldsymbol{h}^{(2)}\ (3\text{個})\ \longrightarrow\ \hat y$$

    途中の値は、前の層の出力を組み合わせて作ります。各ニューロンに最初から「輪郭」「線」「数字」といった意味を割り当てるわけではありません。

    ここからは計算を手で追えるように、**入力2個、出力1個の回帰NN**へ切り替えます。
    784画素すべての計算と、多クラス分類の微分を同時に追う負担を減らすためです。これは数字認識そのものではありませんが、
    層を順に計算し、微分を逆向きに伝えて重みを更新する仕組みは共通です。05で画像の分類へ戻ります。
    """)
    return


@app.cell(hide_code=True)
def _(forward, initial_parameters, mo, network_svg):
    mo.Html('<div class="nn-figure">'+network_svg(initial_parameters(),forward(initial_parameters()),known=set())+'</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 入力から予測へ：順伝播

    **順伝播（forward propagation）**は、入力側から順に各層の値を計算することです。
    この例の出力層では活性化を加えず、重み付き和をそのまま予測値にします。

    $$\begin{aligned}
    \boldsymbol{z}^{(1)}&=W^{(1)}\boldsymbol{x}+\boldsymbol{b}^{(1)},&
    \boldsymbol{h}^{(1)}&=\operatorname{ReLU}(\boldsymbol{z}^{(1)})\\
    \boldsymbol{z}^{(2)}&=W^{(2)}\boldsymbol{h}^{(1)}+\boldsymbol{b}^{(2)},&
    \boldsymbol{h}^{(2)}&=\operatorname{ReLU}(\boldsymbol{z}^{(2)})\\
    \hat y&=\boldsymbol{v}^{\mathsf T}\boldsymbol{h}^{(2)}+c
    \end{aligned}$$

    入力は $\boldsymbol{x}=[1,2]^{\mathsf T}$ です。以下の重みは、計算を追いやすくするために手で設定した初期値です。
    学習済みの数字認識モデルの重みではありません。

    $$W^{(1)}=\begin{bmatrix}1&0.5\\0.5&0.5\\-1&0.5\end{bmatrix},\quad
    \boldsymbol{b}^{(1)}=\begin{bmatrix}0\\-0.5\\1\end{bmatrix}$$

    $$W^{(2)}=\begin{bmatrix}0.5&0.5&0.5\\0.5&0.5&1\\-0.5&0&0.5\end{bmatrix},\quad
    \boldsymbol{b}^{(2)}=\begin{bmatrix}-0.5\\-2\\1.5\end{bmatrix},\quad
    \boldsymbol{v}=\begin{bmatrix}0.5\\0.5\\0.5\end{bmatrix},\quad c=0$$

    $W^{(1)}$ は**3行2列**で、行が行き先の3ニューロン、列が入力元の2個の値です。$W^{(2)}$ は3行3列です。
    各 $\boldsymbol{b},\boldsymbol{z},\boldsymbol{h},\boldsymbol{v}$ は3個の値を持つ列ベクトル、$c,\hat y$ は一つの数です。

    ### 一つの計算から、全体へ

    第一隠れ層の第一ニューロンは、$z^{(1)}_1=1\times1+0.5\times2+0=2$、$h^{(1)}_1=\max(0,2)=2$。
    同じ計算を残りのニューロンでも行い、次の層へ渡します。この例では、更新前も更新後も隠れ層の値はすべて正になります。

    $$\boldsymbol{z}^{(1)}=[2,1,1]^{\mathsf T}\ \longrightarrow\
    \boldsymbol{h}^{(1)}=[2,1,1]^{\mathsf T}$$

    $$\boldsymbol{z}^{(2)}=[1.5,0.5,1]^{\mathsf T}\ \longrightarrow\
    \boldsymbol{h}^{(2)}=[1.5,0.5,1]^{\mathsf T}$$

    $$\hat y=0.5\times1.5+0.5\times0.5+0.5\times1+0=1.5$$

    次のデモでは、丸の中が活性化後の $h$、丸の下が活性化前の $z$ です。計算中の辺に重みを表示します。
    まず「進む」で、一つのニューロンの和とReLUを追ってください。全体の流れは「自動ループ再生」でも見られます。同じボタンでもう一度押すと停止します。
    """)
    return


@app.cell(hide_code=True)
def _(forward_frames, mo, player_channel, player_html):
    mo.iframe(player_html(forward_frames(), "入力から予測まで", player_channel), height="1000px")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    予測1.5が得られたら、ここで**推論は完了**です。正解、損失、微分は使っていません。
    推論中に変わったのは計算された中間値で、重みは固定されたままです。

    学習済みモデルで新しい画像を認識する場合も、まずは学習済みの重みを固定し、入力から予測を計算します。
    次は、予測を正解と比較し、重みをどう変えるかを考えます。

    ## 4. 損失から重みへ：連鎖律と逆伝播

    ### 予測のずれを、損失にする

    この小例の正解を $y=2$ とします。予測は $\hat y=1.5$ なので、正解より0.5小さい状態です。
    ここでは一つのデータに対する二乗誤差の半分を損失とします。

    $$L=\frac12(\hat y-y)^2=\frac12(1.5-2)^2=0.125$$

    02のMSEは複数点の二乗誤差の平均でした。ここでは**一サンプル**だけを使い、微分の式を簡単にするため係数 $1/2$ を付けています。
    03の分類で使った交差エントロピーとは別の損失です。予測の種類に合った損失を使いますが、微分を伝える基本的な方法は共通です。

    ### 知りたいのは「この重みを少し変えると、損失はどれだけ変わるか」

    例えば入力側の重み $W^{(1)}_{11}$ を変えると、第一隠れ層、第二隠れ層、出力を経由して損失が変わります。
    この依存関係をノードと矢印で表したものが**計算グラフ**です。ニューロン一つも、細かく見ると和とReLUの二つの計算を含みます。

    ほかを固定して一つの値を変えたときの変化率を**偏微分**と呼び、$\partial L/\partial W^{(1)}_{11}$ のように書きます。
    すべてのパラメータに対する偏微分を並べたものが**勾配** $\nabla L$ です。

    ### 連鎖律：途中の変化率を掛ける

    $u$ を変えると $v$ が変わり、その結果 $L$ が変わる場合、

    $$\frac{\partial L}{\partial u}=
    \frac{\partial L}{\partial v}\frac{\partial v}{\partial u}$$

    となります。これを**連鎖律（chain rule）**と呼びます。小さな変化を
    $\Delta v\approx (\partial v/\partial u)\Delta u$、
    $\Delta L\approx(\partial L/\partial v)\Delta v$ とつなぐと、二つの変化率の積になると読めます。

    まず損失から予測への微分は、

    $$\frac{\partial L}{\partial\hat y}=\hat y-y=-0.5$$

    です。予測をほんの少し増やすと、正解2に近づき損失が減るので、ここでは負になります。
    これは損失そのものの値0.125でも、単に誤差をコピーしたものでもありません。**損失の変化率**を計算しています。

    次に、第二隠れ層の第一ニューロンだけを通る経路を追います。

    $$L\leftarrow\hat y\leftarrow h^{(2)}_1\leftarrow z^{(2)}_1
    \leftarrow h^{(1)}_1\leftarrow z^{(1)}_1\leftarrow W^{(1)}_{11}$$

    この経路からの寄与は、途中の局所的な微分を掛けた値です。

    $$\underbrace{(-0.5)}_{\partial L/\partial\hat y}
    \times\underbrace{0.5}_{\partial\hat y/\partial h^{(2)}_1}
    \times\underbrace{1}_{\operatorname{ReLU}'(z^{(2)}_1)}
    \times\underbrace{0.5}_{\partial z^{(2)}_1/\partial h^{(1)}_1}
    \times\underbrace{1}_{\operatorname{ReLU}'(z^{(1)}_1)}
    \times\underbrace{1}_{\partial z^{(1)}_1/\partial W^{(1)}_{11}}
    =-0.125$$

    **これは一本の経路の寄与で、まだ重みの勾配全体ではありません。** 第一隠れ層の値は、第二隠れ層のほかのニューロンにも使われています。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ReLUを通る微分

    ReLUは正側では傾き1、負側では傾き0です。

    $$\operatorname{ReLU}'(z)=\begin{cases}0&z<0\\1&z>0\end{cases}$$

    戻ってきた勾配にこの値を掛けます。負側では局所微分が0なので、勾配に0を掛けることになります。
    「ニューロンの出力」と「勾配」は別の値です。この動く例では隠れ層の値をすべて正にしているので、ReLUを通るときは勾配に1を掛けます。

    $z=0$ では通常の微分は定まりません。計算では0を使う規約にします。今回の例の $z$ は0を避けています。

    ### 分岐して使われた値には、寄与が足し合わされる

    第一隠れ層の値 $h^{(1)}_1$ は、第二隠れ層の3個のニューロンに使われています。
    したがって、戻ってくる3つの経路の寄与を足します。

    $$\frac{\partial L}{\partial h^{(1)}_1}
    =\sum_{j=1}^{3}\frac{\partial L}{\partial z^{(2)}_j}
    \frac{\partial z^{(2)}_j}{\partial h^{(1)}_1}$$

    $$=(-0.25)\times0.5+(-0.25)\times0.5+(-0.25)\times(-0.5)=-0.125$$

    そこから第一隠れ層のReLUと入力を通るので、

    $$\frac{\partial L}{\partial W^{(1)}_{11}}=(-0.125)\times1\times1=-0.125$$

    となります。3経路からの寄与は−0.125、−0.125、＋0.125です。この例では第二・第三経路が打ち消し合いますが、それも全経路を足して初めて分かります。
    **一本の経路では微分を掛け、複数の経路から戻る寄与は足す。** この二つが、連鎖律を計算グラフへ適用する要点です。

    ### 逆伝播：後ろから順に計算し、途中の勾配を再利用する

    **逆伝播（backpropagation）**は、出力側から入力側へ、連鎖律を使って勾配を求める手続きです。
    各重みについて経路を一からたどり直す代わりに、ある値に対する勾配を一度求め、そこから手前へ伝えて使い回します。

    表では $\boldsymbol{\delta}^{(\ell)}=\partial L/\partial\boldsymbol{z}^{(\ell)}$ と書きます。
    $\odot$ は同じ位置の要素同士の掛け算、$\mathsf T$ は転置です。

    | 求めるもの | 計算の形 | この例の値 |
    |---|---|---|
    | 予測への微分 | $\partial L/\partial\hat y=\hat y-y$ | $-0.5$ |
    | 出力の重み・バイアス | $\partial L/\partial\boldsymbol v=(\hat y-y)\boldsymbol h^{(2)}$、$\partial L/\partial c=\hat y-y$ | $[-0.75,-0.25,-0.5]^{\mathsf T}$、$-0.5$ |
    | 第二隠れ層の和 | $\boldsymbol\delta^{(2)}=((\hat y-y)\boldsymbol v)\odot\operatorname{ReLU}'(\boldsymbol z^{(2)})$ | $[-0.25,-0.25,-0.25]^{\mathsf T}$ |
    | 第二隠れ層の重み・バイアス | $\partial L/\partial W^{(2)}=\boldsymbol\delta^{(2)}(\boldsymbol h^{(1)})^{\mathsf T}$、$\partial L/\partial\boldsymbol b^{(2)}=\boldsymbol\delta^{(2)}$ | 下のデモに全成分を表示 |
    | 第一隠れ層の出力 | $\partial L/\partial\boldsymbol h^{(1)}=(W^{(2)})^{\mathsf T}\boldsymbol\delta^{(2)}$ | $[-0.125,-0.25,-0.5]^{\mathsf T}$ |
    | 第一隠れ層の和 | $\boldsymbol\delta^{(1)}=(\partial L/\partial\boldsymbol h^{(1)})\odot\operatorname{ReLU}'(\boldsymbol z^{(1)})$ | $[-0.125,-0.25,-0.5]^{\mathsf T}$ |
    | 入力側の重み・バイアス | $\partial L/\partial W^{(1)}=\boldsymbol\delta^{(1)}\boldsymbol x^{\mathsf T}$、$\partial L/\partial\boldsymbol b^{(1)}=\boldsymbol\delta^{(1)}$ | 下のデモに全成分を表示 |

    行列の式は、一つずつ計算した掛け算と足し算をまとめた表記です。例えば重みの一成分なら、
    $\partial L/\partial W^{(2)}_{11}=\delta^{(2)}_1h^{(1)}_1=(-0.25)\times2=-0.5$ です。
    バイアスは足し算される値なので局所微分が1となり、その場所の $\delta$ がそのままバイアスの勾配になります。

    デモの橙の破線は逆向きに伝える勾配、丸の中と下に残る値は順伝播の結果です。
    **逆伝播の途中では重みを変えません。** 微分の計算に必要な重みと中間値は、予測を作ったときのものを使います。
    """)
    return


@app.cell(hide_code=True)
def _(backward_frames, mo, player_channel, player_html):
    mo.iframe(player_html(backward_frames(), "損失・連鎖律・逆伝播", player_channel), height="1050px")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    最後のコマで、すべてのパラメータの勾配が揃いました。入力 $\boldsymbol{x}$ を更新する必要はありません。
    学習で変えたいのは、重みとバイアスです。

    ## 5. 重みを更新し、同じ入力でもう一度予測する

    逆伝播で求めた勾配を使って、全パラメータ $\theta$ を一回更新します。

    $$\theta_{\mathrm{new}}=\theta_{\mathrm{old}}-\eta\nabla L$$

    $\eta$（イータ）は**学習率**で、一歩の大きさを調整します。この例では $\eta=0.05$ に固定します。
    例えば $W^{(1)}_{11}=1$ の勾配が−0.125なら、

    $$W^{(1)}_{11,\mathrm{new}}=1-0.05\times(-0.125)=1.00625$$

    出力の重み $v_1=0.5$ の勾配は−0.75なので、

    $$v_{1,\mathrm{new}}=0.5-0.05\times(-0.75)=0.5375$$

    となります。勾配が負なら、負の数を引くので重みは増えます。
    **全パラメータの勾配を更新前の同じ状態で求め、その後に一緒に更新します。** 途中で一部の重みだけを変えてから残りの勾配を求めることはしません。
    """)
    return


@app.cell(hide_code=True)
def _(backward, forward, gradient_table, initial_parameters, mo):
    _p = initial_parameters()
    _g, _ = backward(_p, forward(_p))
    mo.Html(gradient_table(_p, _g, with_update=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 更新後は、中間値も計算し直す

    重みを変えたら、古い $\boldsymbol h^{(1)},\boldsymbol h^{(2)}$ をそのまま使うことはできません。
    同じ入力 $\boldsymbol x=[1,2]^{\mathsf T}$ でも、重み付き和が変わるからです。

    更新後の第一ニューロンなら、

    $$z^{(1)}_{1,\mathrm{new}}=1.00625\times1+0.5125\times2+0.00625=2.0375$$

    となり、更新前の2から変わります。第一隠れ層、第二隠れ層、出力の順に計算し直します。
    次のデモは、**左に更新前、右に更新後**を並べています。左の結果を残したまま、右側の計算を進めます。
    画面が狭い場合は、比較図の中を横にスクロールできます。
    """)
    return


@app.cell(hide_code=True)
def _(mo, player_channel, player_html, update_frames):
    mo.iframe(player_html(update_frames(), "更新前と更新後を見比べる", player_channel), height="1050px")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    更新後の結果は、次のようになります。

    | 値 | 更新前 | 更新後 |
    |---|---|---|
    | 入力 | $[1,2]^{\mathsf T}$ | 同じ |
    | 第一隠れ層の出力 | $[2,1,1]^{\mathsf T}$ | $[2.0375,1.075,1.15]^{\mathsf T}$ |
    | 第二隠れ層の出力 | $[1.5,0.5,1]^{\mathsf T}$ | $[1.7225,0.7975,1.1475]^{\mathsf T}$ |
    | 予測 | $1.5$ | $1.962$ |
    | 損失 | $0.125$ | $0.000722$ |

    $$\hat y_{\mathrm{new}}=0.5375\times1.7225+0.5125\times0.7975+0.525\times1.1475+0.025=1.962$$

    $$L_{\mathrm{new}}=\tfrac12(1.962-2)^2=0.000722$$

    予測は正解2に近づきました。ずれの絶対値は0.5から0.038へ小さくなったので、損失は減っています。
    **重みが変わる → 中間値が変わる → 予測と損失が変わる**、という流れを見てください。

    この一回では損失が減りましたが、どんな学習率でも必ず減るわけではありません。
    また、一つのデータに合うようになったことだけでは、未知のデータでの性能は分かりません。
    """)
    return


@app.cell(hide_code=True)
def _(backward, forward, inspect, mo, update):
    mo.accordion({
        "計算の中身を見る：順伝播": mo.md("```python\n" + inspect.getsource(forward) + "\n```"),
        "計算の中身を見る：逆伝播": mo.md("```python\n" + inspect.getsource(backward) + "\n```"),
        "計算の中身を見る：更新": mo.md("```python\n" + inspect.getsource(update) + "\n```"),
    })
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 予測・勾配・更新を区別する

    | 工程 | 何をするか | 重みは変わる？ |
    |---|---|---|
    | 順伝播 | 入力から各層の値と予測を計算する | 固定する |
    | 損失 | 予測を正解と比較する | 固定する |
    | 逆伝播 | 連鎖律で各パラメータの勾配を求める | 固定する |
    | 更新 | 勾配と学習率を使ってパラメータを変える | ここで変える |
    | 更新後の順伝播 | 新しい重みで中間値と予測を計算し直す | 新しい値で固定する |

    NNでは、非線形な変換を重ねることで複雑な関係を表現します。
    その重みを、**順伝播 → 損失 → 逆伝播 → 更新**を繰り返して学習します。
    推論だけなら、正解・損失・逆伝播・更新は不要です。

    **表現力**はどんな関係を表せるか、**学習**はデータを使って重みを求めること、**汎化**は未知のデータでも予測が通用することです。
    この章の一回の更新は学習の仕組みの例です。NNでも、訓練に使っていないデータで性能を評価する必要があります。

    05では手書き数字に戻り、PyTorchでMNISTを学習します。
    この章で追った順伝播はモデルの呼び出し、逆伝播はAutogradによる `backward()`、重みの更新は `optimizer.step()` に対応します。
    自動化された後も、それぞれの役割は同じです。学習曲線を読み、過学習や正則化、評価へつなげます。

    別の章へは左上の **☰** メニューから移動できます。終了は起動したターミナルで `Ctrl+C`、再開は教材ルートで `uv run python app.py` です。
    教材全体の案内は、一つ上のフォルダの `README.md` にあります。
    """)
    return


if __name__ == "__main__":
    app.run()
