import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="02 回帰とモデルの選び方", css_file="notebook.css")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""<div class="lesson hero"><div class="eyebrow">B3 MACHINE LEARNING · 02</div><h1>回帰とモデルの選び方</h1><p>単回帰・重回帰から、過学習と正則化、未知データの評価まで。</p></div>""")
    return


@app.cell(hide_code=True)
def _(escape, mo):
    def control_panel(title, items):
        content=mo.vstack(items,gap=1)
        return mo.Html(f'<section class="experiment"><div class="experiment-title">{escape(title)}</div>{content.text}</section>')

    def control_grid(items, hidden=()):
        # 固定した位置に部品を置く。途中の項の表示切替で隣の部品を再利用させない。
        columns=[]
        for i,item in enumerate(items):
            visibility=' style="display:none"' if i in hidden else ''
            columns.append(f'<div{visibility}>{item.text}</div>')
        return mo.Html('<div class="control-grid">'+''.join(columns)+'</div>')

    def formula_box(text):
        return mo.Html(f'<div class="formula">{mo.md(text).text}</div>')

    return control_grid, control_panel, formula_box


@app.cell(hide_code=True)
def _(np):
    def polynomial_formula(model, standardized=False):
        # 表示だけを丸める。予測・誤差は元の精度で計算する。
        weights=model.weights.copy()
        if not standardized:
            weights[1:]=weights[1:]/model.scale
            weights[0]-=np.dot(weights[1:],model.mean)
        terms=[f'{weights[0]:.4f}']
        for j,w in enumerate(weights[1:],1):
            variable=rf'z_{{{j}}}' if standardized else ('x' if j==1 else rf'x^{{{j}}}')
            terms.append(f'{"+" if w>=0 else "-"}{abs(w):.4f}'+variable)
        per_line=4 if model.degree<=3 else 3
        lines=[''.join(terms[i:i+per_line]) for i in range(0,len(terms),per_line)]
        return r'\begin{aligned}\hat y\approx{}&'+r'\\&'.join(lines)+r'\end{aligned}'

    return (polynomial_formula,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## はじめに：データから、まだ分からない数値を見積もる

    明日、お店で商品がいくつ売れるか分かれば、仕入れる量を考えられます。配送に何分かかるか分かれば、到着の目安を伝えられます。
    このように、**結果が分かる前に数値を見積もり、判断に役立てたい**場面があります。

    そこで手掛かりになるのが、過去の記録です。例えば、これまでの気温と売上の組から関係をつかめれば、明日の予想気温を使って売上を予測できそうです。
    この章では、こうした**入力と結果の関係をデータから学び、新しい入力に対する数値を予測する**方法を、簡単な例で考えます。

    まず一本の直線で予測する式を作り、予測のずれを測ります。次に、使う情報を増やしたり、曲線を使ったりして、表せる関係を広げます。
    最後に、その工夫がまだ見ていないデータの予測にも役立つかを確かめ、モデルを選びます。

    **目安は約１時間**です。①予測と誤差 → ②重回帰と曲線 → ③汎化とモデル選択 → ④正則化と最終評価、と進みます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import importlib.util
    mo.stop(importlib.util.find_spec("numpy") is None, mo.md("NumPyが必要です。教材ルートのターミナルで `uv add numpy` を実行し、教材を開き直してください。"))
    from pathlib import Path
    import inspect
    import numpy as np
    from dataclasses import dataclass
    from html import escape
    import uuid

    return dataclass, escape, inspect, np, uuid


@app.cell(hide_code=True)
def _(np):
    SIMPLE_X = np.array([1., 2., 3.])
    SIMPLE_Y = np.array([2., 3., 5.])
    MULTI_X = np.array([[1., 1.], [1., 3.], [2., 1.], [2., 3.], [3., 2.], [3., 4.]])
    MULTI_Y = np.array([7., 11., 11., 15., 17., 21.])
    LAMBDAS = (0., .00001, .0001, .001, .01, .1, 1., 10.)
    # 9次ではごく小さいλから曲線が変わるため、0に続けて対数間隔で探る。
    FINAL_LAMBDAS = (0., 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
    BLUE, ORANGE, GREEN, INK, MUTED = '#2563a6', '#c66522', '#218269', '#233649', '#65758a'
    CURVE_NOISE = .22
    CURVE_YLIM = (-2.,3.5)
    return (
        BLUE,
        CURVE_NOISE,
        CURVE_YLIM,
        FINAL_LAMBDAS,
        GREEN,
        INK,
        LAMBDAS,
        MULTI_X,
        MULTI_Y,
        MUTED,
        ORANGE,
        SIMPLE_X,
        SIMPLE_Y,
    )


@app.cell(hide_code=True)
def _(np):
    def mse(y, predicted):
        """全ての点の二乗誤差を平均する。"""
        y, predicted = np.asarray(y, dtype=float), np.asarray(predicted, dtype=float)
        if y.shape != predicted.shape or y.size == 0:
            raise ValueError('同じ個数の正解と予測が必要です。')
        return float(np.mean((predicted - y) ** 2))

    return (mse,)


@app.cell(hide_code=True)
def _(CURVE_NOISE, np):
    def curve_data():
        """訓練10点・検証40点。観測時刻に小さな不均等さと測定ノイズを持たせる。"""
        x=np.linspace(0,1,10)
        x[1:-1]+=np.random.default_rng(17).uniform(-.015,.015,8)
        rng=np.random.default_rng(21)
        y=np.sin(2*np.pi*x)+rng.normal(0,CURVE_NOISE,len(x))
        vx=np.linspace(.025,.975,40)
        vy=np.sin(2*np.pi*vx)+rng.normal(0,CURVE_NOISE,len(vx))
        return x,y,vx,vy

    return (curve_data,)


@app.cell(hide_code=True)
def _(CURVE_NOISE, np):
    def heldout_data():
        """最終評価でのみ使用。訓練・検証とは独立した乱数系列。"""
        rng = np.random.default_rng(20260911)
        x = rng.uniform(0, 1, 80)
        return x, np.sin(2 * np.pi * x) + rng.normal(0, CURVE_NOISE, len(x))

    return (heldout_data,)


@app.cell(hide_code=True)
def _(dataclass, np):
    @dataclass(frozen=True)
    class Polynomial:
        degree: int
        strength: float
        weights: np.ndarray
        mean: np.ndarray
        scale: np.ndarray

        def predict(self, x):
            features = np.asarray(x, dtype=float).reshape(-1, 1) ** np.arange(1, self.degree + 1)
            return self.weights[0] + ((features - self.mean) / self.scale) @ self.weights[1:]

        @property
        def penalty(self):
            return float(self.strength * np.sum(self.weights[1:] ** 2))

    return (Polynomial,)


@app.cell(hide_code=True)
def _(Polynomial, np):
    def fit_polynomial(x, y, degree=3, strength=0.):
        """訓練MSE + λΣw²を最小化。切片は罰則から除外する。"""
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
            raise ValueError('少なくとも2点を用意してください。')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('座標は有限の数にしてください。')
        if degree not in range(1, 10) or strength < 0 or not np.isfinite(strength):
            raise ValueError('次数は1〜9、λは0以上にしてください。')
        if np.unique(x).size < 2:
            raise ValueError('xが全て同じです。異なるxの点を追加してください。')
        if strength == 0 and np.unique(x).size < degree + 1:
            raise ValueError(f'λ=0で{degree}次の係数を一意に求めるには、異なるxが{degree+1}個以上必要です。次数を下げるか、点を追加するか、λを正にしてください。')
        features = x[:, None] ** np.arange(1, degree + 1)
        mean, scale = features.mean(axis=0), features.std(axis=0)
        # 定数の特徴量は中心化すると0。除算を安全にし、罰則で係数を0へ。
        scale = np.where(scale > 1e-14, scale, 1.)
        design = np.column_stack([np.ones(len(x)), (features - mean) / scale])
        regularizer = np.diag([0.] + [np.sqrt(len(x) * strength)] * degree)
        matrix = np.vstack([design, regularizer])
        target = np.concatenate([y, np.zeros(degree + 1)])
        weights, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
        if rank < degree + 1 or singular[0] / singular[-1] > 1e12:
            raise ValueError('この点配置では計算が不安定です。xの位置を広げるか、次数を下げるか、λを大きくしてください。')
        return Polynomial(degree, float(strength), weights, mean, scale)

    return (fit_polynomial,)


@app.cell(hide_code=True)
def _(fit_polynomial):
    def fit_line(x, y):
        model = fit_polynomial(x, y, 1)
        w = float(model.weights[1] / model.scale[0])
        return w, float(model.weights[0] - w * model.mean[0])

    return (fit_line,)


@app.cell(hide_code=True)
def _(MULTI_X, MULTI_Y, np):
    def fit_multiple(features=2):
        """同じ6人を、学習時間のみ／学習時間と練習回数で最小二乗フィット。"""
        design = np.column_stack([np.ones(len(MULTI_X)), MULTI_X[:, :features]])
        weights = np.linalg.lstsq(design, MULTI_Y, rcond=None)[0]
        return weights, design @ weights

    return (fit_multiple,)


@app.cell(hide_code=True)
def _(curve_data, fit_polynomial, mse, np):
    def degree_errors(strength=0.):
        """訓練・検証だけで全次数の比較を計算する。"""
        x, y, vx, vy = curve_data()
        models = [fit_polynomial(x, y, d, strength) for d in range(1, 10)]
        return (np.array([mse(y, m.predict(x)) for m in models]),
                np.array([mse(vy, m.predict(vx)) for m in models]))

    return (degree_errors,)


@app.cell(hide_code=True)
def _(BLUE, ORANGE, degree_errors, np, plot):
    def error_plot(strength=0., selected=None, baseline=False):
        train, valid = degree_errors(strength)
        ds = np.arange(1, 10)
        curves = [('訓練MSE', ds, train, BLUE, False), ('検証MSE', ds, valid, ORANGE, False)]
        if baseline:
            bt, bv = degree_errors(0.)
            curves += [('正則化なし：訓練', ds, bt, BLUE, True), ('正則化なし：検証', ds, bv, ORANGE, True)]
        stats = [('λ', f'{strength:g}')]
        if selected is not None:
            stats += [('表示中の次数', str(selected)),
                      ('訓練MSE', number(train[selected-1])), ('検証MSE', number(valid[selected-1]))]
        return plot([], curves, xlim=(1, 9), ylim=(0, .8), xticks=ds, yticks=np.arange(0, .81, .2),
                    xlabel='次数 d', ylabel='MSE（小さいほどよい）',
                    title=f'次数ごとの訓練MSEと検証MSE：λ={strength:g}',
                    markers=True, highlight=selected, stats=stats)

    return (error_plot,)


@app.cell(hide_code=True)
def _(curve_data, fit_polynomial, heldout_data, mse):
    def evaluate_final(degree, strength):
        """選択済みの条件だけを独立したtestで一度評価し、結果を保存する。"""
        x, y, vx, vy = curve_data()
        model = fit_polynomial(x, y, degree, strength)
        tx, ty = heldout_data()
        return dict(degree=degree, strength=strength, model=model,
                    train=mse(y, model.predict(x)), validation=mse(vy, model.predict(vx)),
                    test=mse(ty, model.predict(tx)), count=len(tx), test_x=tx, test_y=ty)

    return (evaluate_final,)


@app.function(hide_code=True)
def number(value, digits=4):
    if abs(value) < 1e-11:
        value = 0.
    return f'{value:,.{digits}f}'


@app.cell(hide_code=True)
def _(escape):
    def table(headers, rows, selected=None):
        head = ''.join(f'<th>{escape(str(v))}</th>' for v in headers)
        body = ''.join('<tr' + (' class="chosen"' if i == selected else '') + '>' +
                       ''.join(f'<td>{escape(str(v))}</td>' for v in row) + '</tr>'
                       for i, row in enumerate(rows))
        return f'<div class="lesson table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    return (table,)


@app.cell(hide_code=True)
def _(escape):
    def metrics(items):
        return '<div class="lesson metrics">' + ''.join(
            f'<div class="metric"><span>{escape(label)}</span><strong>{escape(str(value))}</strong></div>'
            for label, value in items) + '</div>'

    return


@app.cell(hide_code=True)
def _(INK, MUTED, ORANGE, escape, np, uuid):
    def plot(groups, curves=(), *, xlim=(0, 1), ylim=(-1.8, 1.8), xlabel='入力 x', ylabel='正解・予測 y',
             title='', residual=None, selected=None, stats=(), xticks=None, yticks=None, markers=False, highlight=None):
        """固定軸のSVG。groups=(ラベル,x,y,色,形)、curves=(ラベル,x,y,色,破線)。"""
        width, left, right, top = 760, 66, 18, 28
        pw, ph = width-left-right, 255
        legend_count = len(groups) + len(curves) + int(residual is not None)
        legend_rows = (legend_count + 1) // 2
        height = 345 + 28 * legend_rows + 28 * ((len(stats) + 1) // 2)
        sx = lambda v: left + (float(v)-xlim[0])/(xlim[1]-xlim[0])*pw
        sy = lambda v: top + (ylim[1]-float(v))/(ylim[1]-ylim[0])*ph
        clip = 'c'+uuid.uuid4().hex
        bits=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}" style="width:100%;height:auto;display:block;background:#fff;border-radius:12px"><title>{escape(title)}</title>',
              f'<defs><clipPath id="{clip}"><rect x="{left}" y="{top}" width="{pw}" height="{ph}"/></clipPath></defs>']
        for y in (np.linspace(*ylim,5) if yticks is None else yticks):
            yy=sy(y)
            bits.append(f'<path d="M{left},{yy}H{width-right}" stroke="#e5ebf1"/><text x="{left-10}" y="{yy+4}" text-anchor="end">{y:g}</text>')
        for x in (np.linspace(*xlim,5) if xticks is None else xticks):
            xx=sx(x)
            bits.append(f'<path d="M{xx},{top}V{top+ph}" stroke="#edf1f5"/><text x="{xx}" y="{top+ph+22}" text-anchor="middle">{x:g}</text>')
        bits.append(f'<text x="{left}" y="17">{escape(ylabel)}</text><text x="{width-right}" y="{top+ph+43}" text-anchor="end">{escape(xlabel)}</text>')
        bits.append(f'<g clip-path="url(#{clip})">')
        if highlight is not None:
            bits.append(f'<path d="M{sx(highlight)},{top}V{top+ph}" stroke="#233649" stroke-opacity=".22" stroke-width="12"/>')
        if residual is not None:
            rx,ry,rp = residual
            for i,(x,y,p) in enumerate(zip(rx,ry,rp)):
                bits.append(f'<path d="M{sx(x)},{sy(y)}V{sy(p)}" stroke="{ORANGE if i==selected else MUTED}" stroke-width="{3 if i==selected else 1.5}" stroke-dasharray="5 4"/>')
        clipped=False
        for label,xs,ys,color,dashed in curves:
            clipped |= bool(np.any(np.asarray(ys)<ylim[0]) or np.any(np.asarray(ys)>ylim[1]))
            coords=' '.join(f'{sx(x):.2f},{sy(np.clip(y,ylim[0]-20*(ylim[1]-ylim[0]),ylim[1]+20*(ylim[1]-ylim[0]))):.2f}' for x,y in zip(xs,ys))
            bits.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.6"'+(' stroke-dasharray="6 4"' if dashed else '')+'/>')
            if markers and not dashed:
                for x, y in zip(xs, ys):
                    bits.append(f'<circle cx="{sx(x)}" cy="{sy(y)}" r="5" fill="{color}"><title>{escape(label)}：次数 {float(x):g}、MSE {float(y):.4f}</title></circle>')
        for label,xs,ys,color,shape in groups:
            for i,(x,y) in enumerate(zip(xs,ys)):
                xx,yy=sx(x),sy(y)
                clipped |= bool(x<xlim[0] or x>xlim[1] or y<ylim[0] or y>ylim[1])
                if shape=='diamond':
                    bits.append(f'<path d="M{xx},{yy-4}l4,4 -4,4 -4,-4Z" fill="{color}" fill-opacity=".7"/>')
                elif shape=='square':
                    bits.append(f'<rect x="{xx-4}" y="{yy-4}" width="8" height="8" fill="{color}"/>')
                else:
                    bits.append(f'<circle cx="{xx}" cy="{yy}" r="5" fill="{color}" stroke="white" stroke-width="1.5"/>')
                if selected is not None and i==selected and shape=='circle':
                    bits.append(f'<circle cx="{xx}" cy="{yy}" r="9" fill="none" stroke="{ORANGE}" stroke-width="2"/>')
        bits.append('</g>')
        legend=[]
        for label,_,_,color,shape in groups:
            legend.append((label,color,shape,False))
        for label,_,_,color,dashed in curves:
            legend.append((label,color,'line',dashed))
        if residual is not None:
            legend.append(('残差（予測 − 正解）',MUTED,'line',True))
        for i,(label,color,shape,dashed) in enumerate(legend):
            lx, ly = left+(i%2)*pw/2, 351+(i//2)*28
            if shape=='line':
                bits.append(f'<path d="M{lx},{ly-5}h22" stroke="{color}" stroke-width="3"'+(' stroke-dasharray="5 4"' if dashed else '')+'/>')
            elif shape=='diamond':
                bits.append(f'<path d="M{lx+10},{ly-11}l6,6 -6,6 -6,-6Z" fill="{color}"/>')
            elif shape=='square':
                bits.append(f'<rect x="{lx+5}" y="{ly-10}" width="10" height="10" fill="{color}"/>')
            else:
                bits.append(f'<circle cx="{lx+10}" cy="{ly-5}" r="5" fill="{color}"/>')
            bits.append(f'<text x="{lx+30}" y="{ly}" fill="{color}">{escape(label)}</text>')
        for i,(label,value) in enumerate(stats):
            lx,ly=left+(i%2)*pw/2,351+legend_rows*28+(i//2)*28
            bits.append(f'<text x="{lx}" y="{ly}" fill="{INK}">{escape(label)}：{escape(str(value))}</text>')
        bits.append('</svg>')
        note='<p class="small">図の範囲を超える値があります。比較のため軸は固定しています。MSEには範囲外の点への誤差も含みます。</p>' if clipped else ''
        return '<div class="lesson figure">'+''.join(bits)+note+'</div>'

    return (plot,)


@app.cell(hide_code=True)
def _(BLUE, GREEN, ORANGE, escape):
    def contributions(values, labels):
        total=sum(values)
        extent=max(total,1.)
        x=20.
        bits=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 740 100" role="img" aria-label="特徴量ごとの寄与"><title>特徴量ごとの寄与を足して予測する</title>']
        for value,label,color in zip(values,labels,[BLUE,GREEN,ORANGE]):
            w=690*value/extent
            if w>0:
                bits.append(f'<rect x="{x}" y="15" width="{w}" height="28" rx="3" fill="{color}"/>')
            x+=w
        for i,(value,label,color) in enumerate(zip(values,labels,[BLUE,GREEN,ORANGE])):
            bits.append(f'<text x="{20+i*240}" y="75" fill="{color}" font-size="16">{escape(label)}：{value:g}</text>')
        return '<div class="lesson figure">'+''.join(bits)+'</svg></div>'

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 単回帰と予測

    ここでは、**「2.5時間勉強した人は、何点くらい取れそうか」**を、得点が分かる前に見積もりたいとします。
    手元にある過去の記録は、1時間で2点、2時間で3点、3時間で5点という3組です。2.5時間の記録はないので、その答えを表から直接読むことはできません。
    この例では小さな数を使います。**説明用の合成データ**であり、現実の因果関係や得点を保証するものではありません。

    学習時間を **入力 $x$**、観測した得点を **正解 $y$** と呼びます。これらの組を手掛かりに予測する式を作るのが、教師あり学習です。
    そのうち、得点や気温のような数値を予測する問題を **回帰（regression）** と呼びます。

    ### 単回帰：一つの入力から予測する式を作る

    記録を見ると、学習時間が長い人ほど得点が高くなっています。この傾向を一本の直線で表せれば、記録にない学習時間についても得点の目安を出せそうです。
    まずは学習時間という**一つの入力を使う単回帰**から始めます。直線を使うのは、「1時間増えると何点くらい増えるか」という単純な関係から試すためです。
    学習時間に傾き $w$ を掛け、切片 $b$ を足して予測します。

    $$\hat y = wx+b$$

    $\hat y$（ワイハット）は**予測**です。$w$ と $b$ はデータから求めたい数で、モデルの**パラメータ**と呼びます。
    最初は $w=1, b=0$、つまり入力と同じ数を予測する直線から始めます。

    ### 予測：作った式を新しい入力に使う

    式を作る目的は、**まだ得点を知らない人にも、学習時間から予測を出せるようにすること**です。
    例えば2.5時間なら、式の $x$ に2.5を入れます。今の仮の式では $1\times2.5+0=2.5$ 点となり、図の四角がその予測を表します。
    このとき正解はまだ分かりません。今の式がよさそうかは、まず得点が分かっている3人への予測がどれくらいずれるかで確かめます。
    **予測を出すためには正解を使いません。** 学習に使わなかったデータにも予測が通用することを、**汎化（generalization）**といいます。
    """)
    return


@app.cell(hide_code=True)
def _(BLUE, GREEN, INK, SIMPLE_X, SIMPLE_Y, mo, plot):
    mo.Html(plot([('観測した正解',SIMPLE_X,SIMPLE_Y,BLUE,'circle'),('新しい入力の予測',[2.5],[2.5],GREEN,'square')],
     [('予測値 = x',[0,4],[0,4],INK,False)],xlim=(0,4),ylim=(-2,8),xlabel='学習時間 x（時間）',ylabel='得点 y（点）',title='3点の正解と、新しい入力2.5時間への予測'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 平均二乗誤差（MSE）

    今の直線では3点とも正解より低く予測しています。どの直線がよいか比べるために、まず各点のずれを測ります。
    ここでは **残差（residual）を「予測 − 正解」** と決めます。

    $$e_i=\hat y_i-y_i$$

    $i$ は点の番号です。3点目なら $e_3=3-5=-2$。負の値は低く予測したことを表します。
    符号付きのまま足すと、上へのずれと下へのずれが打ち消し合うので、各残差を二乗します。

    $$L=\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2$$

    $\sum$ は「各点について足し合わせる」、$n$ は点の数です。二乗誤差を全部足して点の数で割った値を、
    **平均二乗誤差（MSE: mean squared error）**と呼びます。予測のずれを測る**損失（loss）**の一つです。

    初期例では、予測は $[1,2,3]$、残差は $[-1,-1,-2]$、二乗誤差は $[1,1,4]$ なので、

    $$L=\frac{1+1+4}{3}=2$$

    二乗すると符号が消え、ずれが2倍なら寄与は4倍になります。MSEが小さいほど、**この点の集まりに対する**予測のずれが小さいと読めます。
    得点の単位が「点」ならMSEの単位は「点²」で、平均何点外したかとは異なります。

    ### 傾き・切片とMSE

    最初は切片を0に保ち、傾きだけを増やすと3点のずれがどう変わるか予想してください。
    次に切片を変え、直線が平行に移動することを確かめます。図の破線が各点の残差です。図の下部にある全体のMSEと、表の各点の誤差を見比べてください。
    「MSEを最小にする直線に合わせる」を押すと、傾きと切片の入力値も最小二乗解に切り替わります。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    get_line,set_line=mo.state((1.,0.))
    return get_line, set_line


@app.cell(hide_code=True)
def _(SIMPLE_X, SIMPLE_Y, fit_line, get_line, mo, np):
    _lw,_lb=get_line()
    _bw,_bb=[round(v,6) for v in fit_line(SIMPLE_X,SIMPLE_Y)]
    line_w=mo.ui.slider(-1.,3.,step=.1,value=round(_lw,12),include_input=True,full_width=True,label='傾き w')
    line_b=mo.ui.slider(steps=sorted(set(np.round(np.arange(-2,3.01,.1),8).tolist()+[_bb])),value=_lb,show_value=True,full_width=True,label='切片 b')
    return line_b, line_w


@app.cell(hide_code=True)
def _(SIMPLE_X, SIMPLE_Y, fit_line, mo, set_line):
    line_best=mo.ui.button(on_click=lambda _:set_line(tuple(round(v,6) for v in fit_line(SIMPLE_X,SIMPLE_Y))),label='MSEを最小にする直線に合わせる')
    line_reset=mo.ui.button(on_click=lambda _:set_line((1.,0.)),label='直線を初期値に戻す')
    return line_best, line_reset


@app.cell(hide_code=True)
def _(control_grid, control_panel, formula_box, line_b, line_best, line_reset, line_w, mo):
    control_panel('手で係数を変える → 3点のずれを確かめる',[
        control_grid([line_w,line_b]),
        formula_box(rf'現在の式：$\hat y={line_w.value:.8g}x{line_b.value:+.8g}$'),
        mo.hstack([line_best,line_reset],wrap=True,justify='start'),
    ])
    return


@app.cell(hide_code=True)
def _(BLUE, INK, SIMPLE_X, SIMPLE_Y, line_b, line_w, mo, mse, np, plot, table):
    _pred=line_w.value*SIMPLE_X+line_b.value
    _xs=np.linspace(0,4,120)
    mo.vstack([
     mo.Html(plot([('正解',SIMPLE_X,SIMPLE_Y,BLUE,'circle')],
     [('予測直線',_xs,line_w.value*_xs+line_b.value,INK,False)],xlim=(0,4),ylim=(-2,8),yticks=np.arange(-2,9,2),xlabel='学習時間 x（時間）',ylabel='得点（点）',title='単回帰の予測とMSE',residual=(SIMPLE_X,SIMPLE_Y,_pred),
     stats=[('3点全体のMSE',number(mse(SIMPLE_Y,_pred))),('予測値',f'{line_w.value:.8g} × x + ({line_b.value:.4g})')])),
     mo.Html(table(['点','入力 x','正解 y','予測値','残差','二乗誤差'],[(i+1,x,y,f'{p:.3f}',f'{p-y:.3f}',f'{(p-y)**2:.3f}') for i,(x,y,p) in enumerate(zip(SIMPLE_X,SIMPLE_Y,_pred))])),
     mo.md('ある点の誤差が減っても、別の点の誤差が増えることがあります。全点をまとめたMSEが最小になる傾きと切片を求める方法が**最小二乗法**です。最小MSEでも全ての点を通るとは限りません。次の節で、入力が増えた場合の求め方も説明します。')
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 重回帰と多項式回帰

    学習時間だけを使う式では、同じ時間勉強した人には同じ得点を予測します。でも、問題を解く練習の回数など、ほかの情報も予測の手掛かりになるかもしれません。
    ここでは、**一つの入力だけでは捉えられない違いを、複数の情報を使って予測に反映したい**と考えます。

    ### 複数の特徴量を使う

    次の6人について、学習時間、練習回数、得点が分かっているとします。計算を追いやすくした説明用のデータです。
    **同じ学習時間でも得点が違います。** 例えばAさんとBさんはどちらも1時間ですが、練習回数と得点が異なります。

    | 人 | 学習時間 $x_1$（時間） | 練習回数 $x_2$（回） | 得点 $y$（点） |
    |---|---:|---:|---:|
    | A | 1 | 1 | 7 |
    | B | 1 | 3 | 11 |
    | C | 2 | 1 | 11 |
    | D | 2 | 3 | 15 |
    | E | 3 | 2 | 17 |
    | F | 3 | 4 | 21 |

    AさんとBさんの違いを予測にも反映できるよう、練習回数を入力に加えてみましょう。
    予測に使う入力の項目を**特徴量**と呼びます。学習時間に加えて練習回数も使うと、予測は次の式になります。
    複数の特徴量を使う線形回帰が**重回帰**です。

    $$\hat y=w_1x_1+w_2x_2+b$$

    例えば $w_1=4,w_2=2,b=1$ なら、Dさんの予測は $4\times2+2\times3+1=15$ 点です。
    各特徴量に重みを掛けて足すという仕組みは、単回帰と同じです。

    ### 最小二乗法で係数を求める

    重回帰でも、目標は**全員の予測と正解のMSEを小さくすること**です。

    $$L(w_1,w_2,b)=\frac{1}{n}\sum_i(w_1x_{i1}+w_2x_{i2}+b-y_i)^2$$

    $w_1,w_2,b$ を変えれば予測が変わり、MSEも変わります。下ではまず手で重みを動かして、その関係を見られます。
    ただし、多数の重みを手探りで決めるのは大変です。
    このMSEは係数についての二次式なので、**各係数についてMSEの傾きが0になる条件を連立方程式にし、まとめて解く**ことで最小値を求められます。
    この条件は、残差を $e_i=\hat y_i-y_i$ とすると次の形です。

    $$\sum_i e_i=0,\qquad \sum_i x_{i1}e_i=0,\qquad \sum_i x_{i2}e_i=0$$

    左から切片、重み1、重み2についての条件です。残差の式を代入すると、未知数が $b,w_1,w_2$ の一次方程式になります。
    教材の「MSEを最小にする係数に合わせる」は、この最小二乗問題を数値計算で解きます。
    損失を少しずつ下げる**勾配法**もあり、詳しくは03で扱います。

    ### 特徴量と予測誤差の比較

    まず「学習時間だけ」で最小MSEの係数に合わせてください。A・Bの予測は同じになり、両方の正解には一致しません。
    次に「学習時間と練習回数」へ切り替えて再び係数を合わせ、図の予測とMSEを比較してください。
    手で係数を動かすと、そこから誤差がどう増えるかも確かめられます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    get_multi,set_multi=mo.state((2.,0.,1.))
    multi_features=mo.ui.radio({'学習時間だけ':1,'学習時間と練習回数':2},value='学習時間だけ',inline=True,label='重回帰の比較：使う特徴量')
    return get_multi, multi_features, set_multi


@app.cell(hide_code=True)
def _(get_multi, mo, np):
    _mw1,_mw2,_mb=get_multi()
    multi_w1=mo.ui.slider(-5.,10.,step=.5,value=_mw1,include_input=True,full_width=True,label='学習時間の重み w₁')
    multi_w2=mo.ui.slider(-5.,10.,step=.5,value=_mw2,include_input=True,full_width=True,label='練習回数の重み w₂')
    multi_b=mo.ui.slider(steps=sorted(set(np.round(np.arange(-10,10.01,.1),6).tolist()+[_mb])),value=_mb,show_value=True,full_width=True,label='切片 b')
    return multi_b, multi_w1, multi_w2


@app.cell(hide_code=True)
def _(fit_multiple, mo, multi_features, set_multi):
    def fit_multi_action(_):
        _weights,_=fit_multiple(multi_features.value)
        set_multi((round(float(_weights[1]),6),round(float(_weights[2]),6) if len(_weights)==3 else 0.,round(float(_weights[0]),6)))
    multi_best=mo.ui.button(on_click=fit_multi_action,label='MSEを最小にする係数に合わせる')
    multi_reset=mo.ui.button(on_click=lambda _:set_multi((2.,0.,1.)),label='重回帰の係数を初期値に戻す')
    return multi_best, multi_reset


@app.cell(hide_code=True)
def _(control_grid, control_panel, formula_box, mo, multi_b, multi_best, multi_features, multi_reset, multi_w1, multi_w2):
    _two=multi_features.value==2
    _terms=rf'{multi_w1.value:.8g}x_1'
    if _two:
        _terms+=rf'{multi_w2.value:+.8g}x_2'
    _terms+=f'{multi_b.value:+.8g}'
    control_panel('同じ学習時間の2人を、区別して予測できるか',[
        multi_features,
        control_grid([multi_w1,multi_w2,multi_b],hidden=() if _two else (1,)),
        formula_box(rf'現在の式：$\hat y={_terms}$'),
        mo.md('学習時間と練習回数を使います。まず下のボタンで係数を求め、A・Bの予測が分かれるか確かめましょう。' if _two else '学習時間だけを使います。係数をどう変えても、同じ1時間のA・Bには同じ得点を予測します。'),
        mo.hstack([multi_best,multi_reset],wrap=True,justify='start'),
        mo.Html('<div class="hint">スライダーは係数を手で変える操作です。特徴量を切り替えただけでは学習しません。ボタンを押すと、今使っている特徴量で6人全体のMSEが最小になる係数を求めます。</div>'),
    ])
    return


@app.cell(hide_code=True)
def _(
    BLUE,
    GREEN,
    MULTI_X,
    MULTI_Y,
    mo,
    mse,
    multi_b,
    multi_features,
    multi_w1,
    multi_w2,
    np,
    plot,
    table,
):
    _w2=multi_w2.value if multi_features.value==2 else 0.
    _pred=MULTI_X[:,0]*multi_w1.value+MULTI_X[:,1]*_w2+multi_b.value
    mo.vstack([
     mo.Html(plot([('正解',np.arange(1,7),MULTI_Y,BLUE,'circle'),('予測',np.arange(1,7),_pred,GREEN,'square')],xlim=(.5,6.5),ylim=(0,25),yticks=np.arange(0,26,5),xticks=np.arange(1,7),xlabel='人（1=A、2=B、…、6=F）',ylabel='得点（点）',title='6人の正解と重回帰の予測',residual=(np.arange(1,7),MULTI_Y,_pred),stats=[('6人全体のMSE',number(mse(MULTI_Y,_pred))),('使う特徴量',multi_features.value)])),
     mo.Html(table(['人','学習時間','練習回数','正解','予測','二乗誤差'],[(name,*row,y,f'{p:.3f}',f'{(p-y)**2:.3f}') for name,row,y,p in zip('ABCDEF',MULTI_X,MULTI_Y,_pred)])),
     mo.md('この例は2特徴量で誤差を0にできるように作っています。実際の観測にはノイズや未観測の要因があり、全点には合わないこともあります。**訓練MSEが下がっても、未知データの予測がよくなるとは限りません。** また、この予測式だけで学習時間や練習の因果効果は分かりません。')
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({'行列による表現（任意）':mo.md(r"""
    入力を一人一行に並べた行列を $X$ とすると、全員の予測を $\hat{\boldsymbol y}=Xw+b\boldsymbol1$ と書けます。
    この例では $X$ は6行2列、$w$ は2成分、予測は6成分です。各行で行う掛け算と足し算をまとめた表現です。
    切片用の1の列を $X$ に加えると、最小二乗法の条件を一つの行列の方程式にまとめられます。
    実装では、数値の丸め誤差を抑えるため、逆行列を直接作らず行列を分解して解きます。
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 多項式回帰

    入力を増やす別の使い方を考えます。ある装置の出力を、時刻を変えて10回測ったとします。
    次の図では、前半に出力が上がり、その後下がり、最後に戻ってきます。**一本の直線では、この変化を表せません。**
    まず点と直線を見比べてください。時刻と出力は比較しやすい尺度にした、説明用の合成データです。
    """)
    return


@app.cell(hide_code=True)
def _(curve_data, mo, np):
    train_x,train_y,valid_x,valid_y=curve_data()
    grid_x=np.linspace(0,1,401)
    get_evaluation,set_evaluation=mo.state({'locked':False,'ever':False,'result':None})
    return (
        get_evaluation,
        grid_x,
        set_evaluation,
        train_x,
        train_y,
        valid_x,
        valid_y,
    )


@app.cell(hide_code=True)
def _(BLUE, INK, fit_polynomial, grid_x, mo, plot, train_x, train_y):
    _intro=fit_polynomial(train_x,train_y,1)
    mo.Html(plot([('観測した出力',train_x,train_y,BLUE,'circle')],[('最小二乗法で求めた直線',grid_x,_intro.predict(grid_x),INK,False)],xlabel='時刻 x（0〜1に換算）',ylabel='出力 y（相対値）',title='直線では表しきれない出力の変化'))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    直線では、$x$ が同じ量だけ増えるたびに予測も一定量ずつ変わります。
    そこで $x$ に加え、$x^2,x^3$ を特徴量にすると、曲がる予測式を作れます。

    $$\hat y=b+w_1x+w_2x^2+w_3x^3$$

    これが3次の**多項式回帰**です。例えば $x=0.5$ から $[x,x^2,x^3]=[0.5,0.25,0.125]$ を作り、それぞれに重みを掛けて足します。
    新しい測定項目を加える代わりに、一つの入力から複数の特徴量を作っています。

    係数は先ほどの**最小二乗法**で求めます。重回帰の $x_1,x_2$ を $x,x^2,x^3$ に置き換え、訓練MSEが最小になる $b,w_1,w_2,w_3$ を解くだけです。
    **次数を決めるのは私たち、係数を求めるのは訓練データを使った学習**です。

    下で1次から3次へ切り替え、曲線とMSEを比べてください。次数 $d$ のとき、元の入力は1個、作る特徴量は $d$ 個、切片を含む係数は $d+1$ 個です。
    点には測定のばらつきに相当する**ノイズ**も加えているので、全点を通ることが本来の関係を捉えることとは限りません。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    bridge_degree=mo.ui.radio({'1次：直線':1,'3次：曲線':3},value='1次：直線',inline=True,label='多項式の項を増やす')
    return (bridge_degree,)


@app.cell(hide_code=True)
def _(
    BLUE,
    INK,
    bridge_degree,
    control_panel,
    formula_box,
    polynomial_formula,
    fit_polynomial,
    grid_x,
    mo,
    mse,
    plot,
    train_x,
    train_y,
):
    _d=bridge_degree.value
    _model=fit_polynomial(train_x,train_y,_d)
    mo.vstack([
     control_panel('次数を選ぶ → 係数は訓練10点から自動で求める',[bridge_degree,mo.md(f'**{_d}次・特徴量{_d}個・係数{_d+1}個**で、訓練MSEを最小にしました。'),formula_box('学習してできた式（表示は丸めています）：\n\n$$'+polynomial_formula(_model)+'$$')]),
     mo.Html(plot([('訓練点',train_x,train_y,BLUE,'circle')],[('最小二乗法による予測',grid_x,_model.predict(grid_x),INK,False)],title=f'{_d}次の多項式回帰',stats=[('次数',_d),('係数の数',_d+1),('訓練MSE',number(mse(train_y,_model.predict(train_x))))])),
     mo.md('3次では、直線が捉えられなかった曲がりを表せます。では、次数をさらに増やすと予測はよくなるのでしょうか。次は、学習に使わなかった点への誤差も調べます。')
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 汎化と過学習

    目指しているのは、手元の点をなぞることだけではありません。同じ生成の仕組みから来る、**学習に使わなかったデータ**にも予測が通用することです。
    汎化を確かめるには、係数を求めるための点と、評価するための点を分けておきます。

    | データ | 役割 | この教材では |
    |---|---|---|
    | 訓練（train） | 重み・切片を求める | 青い丸の10点 |
    | 検証（validation） | 次数や正則化の強さを選ぶ | 橙色のひし形の40点 |
    | テスト（test） | 設定を決めた後、最後に評価する | 選択が終わるまで点も誤差も見ない |

    **学習とモデル選択は別の段階です。** 次数を決めたら、その次数の係数を訓練データで求めます。
    次数のように学習前に決める設定を**ハイパーパラメータ**と呼びます。どの設定がよいかは、検証データの誤差で比べます。

    ### 次数と過学習

    1次では、訓練点でも検証点でも曲がった関係を十分に捉えられません。このような当てはまり不足を**過少適合（underfitting）**と呼びます。
    3次では、曲がりを捉えて両方の誤差が小さくなります。
    この例の9次は係数が10個あり、異なる時刻の訓練10点すべてを、ノイズごと通れてしまいます。
    訓練MSEはほぼ0になりますが、点の間で大きく曲がり、検証MSEは3次より悪化します。
    訓練データの細かな揺れに合わせすぎて、未知データではうまく予測できない状態が**過学習（overfitting）**です。
    """)
    return


@app.cell(hide_code=True)
def _(
    BLUE,
    CURVE_YLIM,
    GREEN,
    MUTED,
    ORANGE,
    fit_polynomial,
    grid_x,
    mo,
    mse,
    np,
    plot,
    train_x,
    train_y,
    valid_x,
    valid_y,
):
    _curves=[]
    for _d,_color in [(1,MUTED),(3,GREEN),(9,'#9663b0')]:
        _m=fit_polynomial(train_x,train_y,_d)
        _curves.append((f'{_d}次',grid_x,_m.predict(grid_x),_color,_d==1))
    mo.Html(plot([('訓練点',train_x,train_y,BLUE,'circle'),('検証点',valid_x,valid_y,ORANGE,'diamond')],_curves,title='次数1・3・9の予測曲線',ylim=CURVE_YLIM,yticks=np.arange(-2,4)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    下のMSEのグラフは**横軸が次数、縦軸が誤差**です。青は訓練、橙は検証です。
    次数を増やすと訓練MSEは下がり続けますが、検証MSEは下がった後に増えます。この食い違いが過学習の手掛かりです。

    各次数の係数は、同じ訓練10点のMSEが最小になるよう、それぞれ自動で求めています。
    上の1・3・9次の曲線と、下の同じ次数のMSEを見比べてください。この節では正則化を使いません（λ=0）。
    """)
    return


@app.cell(hide_code=True)
def _(error_plot, mo):
    mo.vstack([
        mo.Html(error_plot(0.)),
        mo.md('**訓練MSEが最小の9次を、そのまま選んでよいでしょうか。** 上の曲線と見比べると、訓練点を通ることと、未知の点をよく予測することの違いが分かります。次数が高ければ常に悪いという意味ではなく、検証MSEで判断します。'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    検証誤差は汎化性能を知る手掛かりですが、有限個の点で測った**推定**です。将来の全データへの保証ではありません。
    検証結果を何度も見て設定を選ぶと、その検証データにも合わせていくことになるので、最後の評価用にtestを残します。
    また、将来の入力範囲や生成の仕組みが変われば、同じ性能が出るとは限りません。ここでは入力 $0\le x\le1$ の同じ仕組みを想定しています。

    ## 4. L2正則化とモデル選択

    曲線が点に合わせすぎるとき、次数を下げる以外の方法もあります。
    使う項はそのままにして、係数が大きくなることに罰則を加える方法が**正則化（regularization）**です。

    ここでは、係数の二乗を足したものに、強さ $\lambda$（ラムダ）を掛けます。これを **L2正則化**と呼びます。

    $$J=\underbrace{L_{\mathrm{train}}}_{\text{訓練点へのずれ}}+
    \underbrace{\lambda\sum_{j=1}^{d}w_j^2}_{\text{重みの大きさへの罰則}}$$

    **切片は罰則に含めません。** 当てはまりだけでなく、重みの大きさも考えた **目的関数 $J$** を最小にする係数を求めます。
    **L2正則化では次数も項数も変えません。** 係数が小さくなるように最小化の基準を変えます。
    $\lambda=0$ なら通常の当てはめ、$\lambda$ が大きいほど重みを小さくする働きが強まります。

    $x,x^2,\ldots$ は値の尺度が違うので、各特徴量から**訓練データの平均を引き、訓練データの標準偏差で割って**尺度を揃えます。これを標準化と呼びます。
    検証・testにも同じ平均と標準偏差を使います。以下で表示する係数は、**標準化後の特徴量**に掛かる値です。生の $x^j$ の係数ではありません。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 正則化で過学習を抑える

    ここで確かめたいのは、**訓練点への当てはまりを少し譲ると、未知の点への予測がよくなる場合がある**ということです。
    下の①〜④を順に選んでください。点は一切変わりません。実線が今のモデル、灰色の破線が一つ前のモデルです。

    ①→②では**次数だけ**を増やします。②→③では**9次のまま正則化**します。③→④では**正則化を強めすぎた場合**を見ます。
    曲線が訓練点を通るかだけでなく、橙の検証点からどれくらい外れるかを見てください。
    """)
    return


@app.cell(hide_code=True)
def _(control_panel, mo):
    ridge_step=mo.ui.radio({'① 3次':0,'② 9次':1,'③ 9次＋正則化':2,'④ 正則化を強く':3},value='① 3次',inline=True,label='正則化を理解する4段階')
    control_panel('一つだけ条件を変えて、予測の違いを見る',[ridge_step])
    return (ridge_step,)


@app.cell(hide_code=True)
def _(
    BLUE,
    CURVE_YLIM,
    GREEN,
    MUTED,
    ORANGE,
    fit_polynomial,
    grid_x,
    mo,
    mse,
    np,
    plot,
    ridge_step,
    train_x,
    train_y,
    valid_x,
    valid_y,
):
    _settings=[(3,0.),(9,0.),(9,.0001),(9,10.)]
    _step=ridge_step.value
    _d,_lam=_settings[_step]
    _model=fit_polynomial(train_x,train_y,_d,_lam)
    _te=mse(train_y,_model.predict(train_x));_ve=mse(valid_y,_model.predict(valid_x))
    _titles=['① 3次：大まかな曲がりを捉える','② 9次：訓練点に合わせすぎる','③ 9次＋正則化：点の間の予測を改善する','④ 正則化が強すぎる：曲がりまで失う']
    _notes=[
     '3次は係数が4個です。10点すべてを通りませんが、全体の曲がりを捉えています。次の②では次数だけを増やします。',
     '係数を10個に増やすと、10個の訓練点をほぼ完全に通ります。しかし点の間で曲線が大きく膨らみ、検証MSEは増えました。訓練MSEがほぼ0でも、よい予測とは限りません。',
     '②と同じ9次・同じ10点です。係数を抑えると、全訓練点を通ることをやめ、点の間の大きな膨らみが抑えられます。訓練MSEは増えましたが、検証MSEは大きく下がりました。これがこの例での正則化の効果です。',
     '③からλだけを強めました。重みを抑えすぎて、捉えたかった曲がりまで失っています。訓練・検証MSEの両方が増えました。正則化は強いほどよいわけではありません。'
    ]
    _curves=[]
    _stats=[('次数／係数数',f'{_d}次／{_d+1}個'),('λ',f'{_lam:g}')]
    if _step:
        _bd,_blam=_settings[_step-1]
        _before=fit_polynomial(train_x,train_y,_bd,_blam)
        _bt=mse(train_y,_before.predict(train_x));_bv=mse(valid_y,_before.predict(valid_x))
        _curves.append(('一つ前のモデル',grid_x,_before.predict(grid_x),MUTED,True))
        _stats.extend([('訓練MSE',f'{_bt:.4f} → {_te:.4f}'),('検証MSE',f'{_bv:.4f} → {_ve:.4f}')])
    else:
        _stats.extend([('訓練MSE',number(_te)),('検証MSE',number(_ve))])
    _curves.append(('今のモデル',grid_x,_model.predict(grid_x),GREEN,False))
    mo.vstack([
     mo.md(f'**{_titles[_step]}**\n\n{_notes[_step]}'),
     mo.Html(plot([('訓練点：10点',train_x,train_y,BLUE,'circle'),('検証点：40点',valid_x,valid_y,ORANGE,'diamond')],_curves,title=_titles[_step],ylim=CURVE_YLIM,yticks=np.arange(-2,4),stats=_stats)),
     mo.md('**検証MSEで選ぶのは、点の間を含む未知の入力にも予測が通用するモデルです。** この例では、次数を抑える方法と、9次に正則化を加える方法のどちらも有効です。正則化した高次数が低次数より必ず優れる、という意味ではありません。')
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    show_ridge_free=mo.ui.checkbox(value=False,label='λを変えて、次数ごとのMSEを見る（任意）')
    show_ridge_free
    return (show_ridge_free,)


@app.cell(hide_code=True)
def _(LAMBDAS, mo):
    ridge_lambda=mo.ui.slider(steps=list(LAMBDAS),value=.0001,show_value=True,full_width=True,label='正則化の強さ λ')
    return (ridge_lambda,)


@app.cell(hide_code=True)
def _(control_panel, mo, ridge_lambda, show_ridge_free):
    mo.stop(not show_ridge_free.value)
    control_panel('同じ正則化を、すべての次数にかける',[ridge_lambda,mo.md('λを変えるたびに、**各次数で「訓練MSE＋罰則」が最小になる係数を自動で求め直します。** その後、訓練・検証MSEを描きます。グラフのMSEには罰則を含めません。破線は正則化なしの基準です。')])
    return


@app.cell(hide_code=True)
def _(error_plot, mo, ridge_lambda, show_ridge_free):
    mo.stop(not show_ridge_free.value)
    mo.Html(error_plot(ridge_lambda.value,baseline=True))
    return


@app.cell(hide_code=True)
def _(fit_polynomial, inspect, mo):
    mo.accordion({"計算の中身を見る：標準化とL2正則化":mo.md("訓練データだけで標準化し、二乗誤差の平均に対応する罰則を使う、実際の計算です。導出やコードの読解は任意です。\n\n```python\n"+inspect.getsource(fit_polynomial)+"\n```")})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### テストデータで汎化性能を評価する

    これまでの比較をもとに、**最後に評価するモデルの次数とλを、この下で選んでください。** 初期候補は3次・λ=0です。

    ここで選ぶのは係数そのものではなく、**使う多項式の次数 $d$ と、正則化の強さ $\lambda$** です。
    次数を変えると $x,x^2,\ldots,x^d$ という $d$ 個の特徴量を作り、**その設定で最適な係数を、訓練10点から毎回自動で求め直します。**
    $\lambda=0$ なら訓練MSEを、$\lambda>0$ なら「訓練MSE＋重みへの罰則」を最小にします。検証点は係数の計算に使いません。

    **まずλ=0のまま3次と9次を比較し、次に9次のままλを少しずつ増やしてください。**
    学習してできる式・曲線・訓練MSEが変わることを確かめ、**検証MSEを根拠に**候補を選びます。

    決まったら「このモデルをテストデータで評価」を押します。選んだモデルを固定し、学習にも設定選びにも使っていない80点を重ねて、予測のずれを表示します。
    これが、設定選びで繰り返し見た検証データとは別に、テストデータを残しておく理由です。

    テスト結果を見て選び直すと、そのテストも設定選びに使ったことになります。
    操作をやり直したり再読み込みしたりしても、教材では同じテスト点を使います。再評価は練習として扱ってください。
    """)
    return


@app.cell(hide_code=True)
def _(FINAL_LAMBDAS, get_evaluation, mo):
    _e=get_evaluation()
    _saved=_e['result']
    final_degree=mo.ui.radio({str(d):d for d in range(1,10)},value=str(_saved['degree'] if _saved else 3),inline=True,disabled=_e['locked'],label='多項式の次数 d')
    final_lambda=mo.ui.slider(steps=list(FINAL_LAMBDAS),value=_saved['strength'] if _saved else 0.,show_value=True,full_width=True,disabled=_e['locked'],label='正則化の強さ λ')
    return final_degree, final_lambda


@app.cell(hide_code=True)
def _(final_degree, final_lambda, fit_polynomial, get_evaluation, train_x, train_y):
    _result=get_evaluation()['result']
    final_model=_result['model'] if _result else fit_polynomial(train_x,train_y,final_degree.value,final_lambda.value)
    return (final_model,)


@app.cell(hide_code=True)
def _(control_panel, final_degree, final_lambda, final_model, formula_box, get_evaluation, mo, polynomial_formula):
    _d=final_model.degree
    _power=int(f'{final_model.strength:.0e}'.split('e')[1]) if final_model.strength else 0
    _objective=r'L_{\mathrm{train}}' if final_model.strength==0 else rf'L_{{\mathrm{{train}}}}+10^{{{_power}}}\sum_{{j=1}}^{{{_d}}}w_j^2'
    _status='固定したモデル' if get_evaluation()['locked'] else '自動で学習した候補'
    control_panel('最終評価の候補：設定を選ぶ → 自動で学習 → 検証MSEで判断',[
        final_degree,
        final_lambda,
        mo.Html('<div class="hint">左端は0（正則化なし）。次は10⁻¹⁴で、以後は1目盛りごとに10倍、右端は10⁻⁴です。小さいλでの曲線の変化を見やすくしています。</div>'),
        mo.md(f'**{_status}：{_d}次・特徴量{_d}個・係数{_d+1}個**\n\n'+rf'訓練10点で $J={_objective}$ が最小になるよう、切片と重みを求めました。'),
        formula_box('学習してできた式：\n\n$$'+polynomial_formula(final_model,standardized=True)+'$$'),
        mo.md(r'$z_j=(x^j-\mu_j)/s_j$ は標準化した特徴量です。$\mu_j,s_j$ は訓練点での $x^j$ の平均と標準偏差です。係数は表示だけ丸めています。'),
    ])
    return


@app.cell(hide_code=True)
def _(
    BLUE,
    CURVE_YLIM,
    GREEN,
    INK,
    ORANGE,
    final_model,
    get_evaluation,
    grid_x,
    mo,
    mse,
    np,
    plot,
    train_x,
    train_y,
    valid_x,
    valid_y,
):
    _e=get_evaluation()
    _r=_e['result']
    _model=final_model
    _groups=[('訓練点',train_x,train_y,BLUE,'circle'),('検証点',valid_x,valid_y,ORANGE,'diamond')]
    _stats=[('次数',_model.degree),('λ',f'{_model.strength:g}'),('訓練MSE',number(mse(train_y,_model.predict(train_x)))),('検証MSE',number(mse(valid_y,_model.predict(valid_x))))]
    if _r:
        _groups.append(('テスト点（80点）',_r['test_x'],_r['test_y'],GREEN,'square'))
        _stats.append(('テストMSE',number(_r['test'])))
        _note='モデルを固定し、緑の四角のテスト点を追加しました。曲線が変わらないのは、テスト点を使って学習し直していないからです。検証MSEとテストMSEは別の点で測った推定値なので、同じ値にはなりません。'
    else:
        _note='図のMSEには正則化の罰則を含めず、予測のずれだけを表示しています。検証MSEを根拠に候補が決まったら、下のボタンでモデルを固定してテスト点を表示してください。'
        if _e['ever']:
            _note+=' このテストはすでに見たデータです。今回は操作の練習です。'
    mo.vstack([mo.Html(plot(_groups,[('評価するモデル',grid_x,_model.predict(grid_x),INK,False)],title='最終評価：選んだモデルと未知データ',ylim=CURVE_YLIM,yticks=np.arange(-2,4),stats=_stats)),mo.md(_note)])
    return


@app.cell(hide_code=True)
def _(
    evaluate_final,
    final_degree,
    final_lambda,
    get_evaluation,
    mo,
    set_evaluation,
):
    _e=get_evaluation()
    def finalize_choice(_):
        if not get_evaluation()['locked']:
            set_evaluation({'locked':True,'ever':True,'result':evaluate_final(final_degree.value,final_lambda.value)})
    final_button=mo.ui.button(on_click=finalize_choice,label='このモデルをテストデータで評価',disabled=_e['locked'],kind='success')
    restart_choice=mo.ui.button(on_click=lambda _:set_evaluation({'locked':False,'ever':get_evaluation()['ever'],'result':None}),label='最終評価をやり直す（練習）',disabled=not _e['locked'])
    mo.hstack([final_button,restart_choice],wrap=True,justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 回帰とモデル選択のまとめ

    - **回帰は数値を予測する。** 重みと切片で式を作り、特徴量を増やすと曲線も表せます。
    - **学習では訓練データの目的関数を小さくする。** MSEでずれを測り、L2正則化では重みへの罰則も考えます。
    - **目指すのは汎化。** 訓練点への当てはまりだけでは選べません。検証で次数・λを選び、最後にtestで評価します。

    自分が選んだ候補について「何を変えたか」「どの数値を根拠に選んだか」を一言で振り返ってください。
    コードを完成させる課題や提出物はありません。

    次の03では、**重みをどちらへ、どれだけ動かせばよいか**を損失の傾きから考えます。
    線形二値分類の予測から始め、尤度・損失を通して、勾配更新・学習率・SGDへ進みます。左上の **☰** メニューから「03 線形分類と勾配降下法」を選べます。
    05では、ここで学んだ汎化・過学習・正則化を実際の学習曲線と評価へつなげます。

    ### もっと詳しく読むとき

    [機械学習帳](https://chokkan.github.io/mlnote/)は、式をじっくり読みたいときの資料です。
    **1週間の本編では通読・コード演習は不要**です。

    - [単回帰](https://chokkan.github.io/mlnote/regression/01sra.html)：残差と目的関数を復習し、直線を求める式の導出を読む。
    - [重回帰](https://chokkan.github.io/mlnote/regression/02mra.html)の2.1〜2.3：多項式との接続と、行列による表現を深める。
    - [モデル選択と正則化](https://chokkan.github.io/mlnote/regression/03regularization.html)の3.1〜3.3：汎化・過学習と、検証による選択を復習する。
    - [勾配法によるパラメータ推定](https://chokkan.github.io/mlnote/regression/04sgd.html)：03で扱う、重みを求める方法を詳しく読む。

    本教材は残差を予測−正解、損失を二乗誤差の平均としています。参照先には逆符号の残差や二乗和による表記もあります。
    MSE自体は残差の符号によらず、平均を省いた式とは係数が変わります。正則化の数値もそのまま移さず、式を確認してください。

    終えるときは、marimoを起動したターミナルで `Ctrl+C` を押します。再開するときは教材ルートで `uv run python app.py` を実行します。
    """)
    return


if __name__ == "__main__":
    app.run()
