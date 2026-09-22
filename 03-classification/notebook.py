import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="03 線形分類と勾配降下法")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np

    return mo, np


@app.cell(hide_code=True)
def classification_core(np):
    from pathlib import Path
    from html import escape
    import json

    X = np.array([-3., -2., -1., 1., 2., 3.])[:, None]
    Y = np.array([0., 1., 0., 1., 0., 1.])
    BLUE, ORANGE, PURPLE = '#2563a6', '#bb6025', '#7856a3'

    def sigmoid(s):
        return np.exp(-np.logaddexp(0., -np.asarray(s, dtype=float)))

    def loss_gradient(x, y, theta):
        x, y, theta = np.asarray(x), np.asarray(y), np.asarray(theta)
        scores = x @ theta[:-1] + theta[-1]
        error = sigmoid(scores) - y
        return float(np.mean(np.logaddexp(0., scores)-y*scores)), np.r_[x.T @ error / len(y), error.mean()]

    def softmax_loss(scores, label):
        scores = np.asarray(scores, dtype=float)
        shifted = scores - scores.max()
        logsum = np.log(np.exp(shifted).sum())
        return np.exp(shifted-logsum), float(logsum-shifted[label])

    def history(eta=.3, epochs=9, batch_size=6, fixed_bias=True):
        theta = np.array([-1., 0.])
        rng = np.random.default_rng(31)
        states = [dict(theta=theta.tolist(), loss=loss_gradient(X,Y,theta)[0])]
        for _epoch in range(epochs):
            order = rng.permutation(len(Y))
            for ids in np.array_split(order, len(Y)//batch_size):
                _, g = loss_gradient(X[ids],Y[ids],theta)
                if fixed_bias:
                    g[-1] = 0.
                theta = theta-eta*g
                states.append(dict(theta=theta.tolist(), loss=loss_gradient(X,Y,theta)[0]))
        return states

    def chart(series, *, xlim, ylim, xlabel, ylabel, xticks, yticks, width=720, height=280,
              dots=(), rings=(), background='', annotation=None):
        left, right, top, bottom = 52, 18, 34, 46
        sx = lambda x: left+(float(x)-xlim[0])/(xlim[1]-xlim[0])*(width-left-right)
        sy = lambda y: top+(ylim[1]-float(y))/(ylim[1]-ylim[0])*(height-top-bottom)
        bits=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(ylabel)}と{escape(xlabel)}の関係">']
        for y in yticks:
            bits.append(f'<path d="M{left},{sy(y)}H{width-right}" stroke="#edf0f3"/><text x="{left-9}" y="{sy(y)+4}" text-anchor="end">{y:g}</text>')
        for x in xticks:
            bits.append(f'<text x="{sx(x)}" y="{height-bottom+22}" text-anchor="middle">{x:g}</text>')
        bits.append(f'<path d="M{left},{top}V{height-bottom}H{width-right}" fill="none" stroke="#a7b4bf"/><text x="{left}" y="18">{escape(ylabel)}</text><text x="{width-right}" y="{height-5}" text-anchor="end">{escape(xlabel)}</text>')
        bits.append(background)
        for points,color,dash in series:
            coords=' '.join(f'{sx(x):.2f},{sy(y):.2f}' for x,y in points)
            bits.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.7" stroke-dasharray="{dash}" stroke-linejoin="round"/>')
        for x,y,color in rings:
            bits.append(f'<circle cx="{sx(x)}" cy="{sy(y)}" r="5" fill="white" stroke="{color}" stroke-width="2"/>')
        for x,y,color in dots:
            bits.append(f'<circle cx="{sx(x)}" cy="{sy(y)}" r="5" fill="{color}" stroke="white" stroke-width="1.4"/>')
        if annotation:
            x,y,label=annotation
            bits.append(f'<text x="{sx(x)}" y="{sy(y)}" text-anchor="middle">{escape(label)}</text>')
        return ''.join(bits)+'</svg>'

    def figure(content, caption):
        return f'<figure class="textbook-figure">{content}<figcaption>{caption}</figcaption></figure>'

    def sigmoid_figure():
        xs=np.linspace(-5,5,150)
        graphic=chart([(list(zip(xs,sigmoid(xs))),BLUE,'')],xlim=(-5,5),ylim=(0,1),
            xlabel='スコア s',ylabel='クラス1の確率 p',xticks=[-4,0,4],yticks=[0,.5,1],
            dots=[(-1,float(sigmoid(-1)),ORANGE),(0,.5,BLUE),(1,float(sigmoid(1)),ORANGE)])
        return figure(graphic,'図1　sigmoid関数。スコア−1・0・1は、それぞれ確率約0.27・0.50・0.73に対応する。')

    def boundary_figure():
        # A small fixed example; the boundary is x₁=x₂, not a training-result dashboard.
        graphic=chart([([(-2,-2),(2,2)],'#526575','')],xlim=(-2.2,2.2),ylim=(-2.2,2.2),
            xlabel='特徴量 x₁',ylabel='特徴量 x₂',xticks=[-2,0,2],yticks=[-2,0,2],width=500,height=300,
            dots=[(-1,1,ORANGE),(-1.5,.5,ORANGE),(1,-1,BLUE),(1.5,-.5,BLUE)],
            annotation=(-.9,1.75,'クラス0の側'))
        return figure('<div class="small-figure">'+graphic+'</div>','図2　w₁=1、w₂=−1、b=0の決定境界。直線上はp=0.5、x₁がx₂より大きい側ではp&gt;0.5となる。青い点はクラス1、橙の点はクラス0の例。')

    def loss_figure():
        qs=np.linspace(.03,1,130)
        graphic=chart([(list(zip(qs,-np.log(qs))),BLUE,'')],xlim=(0,1),ylim=(0,3.6),
            xlabel='正解に与えた確率 q',ylabel='一点の損失 −log q',xticks=[0,.5,1],yticks=[0,1,2,3],
            dots=[(q,float(-np.log(q)),ORANGE) for q in [.2,.5,.8]])
        return figure(graphic,'図3　正解への確率が低いほど損失は大きい。自信を持って外した予測には、大きな損失が与えられる。')

    def rate_figure():
        items=[]
        for eta,color,label in [(.1,BLUE,'小さい：η=0.1'),(1,ORANGE,'適度：η=1'),(3,PURPLE,'大きすぎる：η=3')]:
            states=history(eta=eta)
            points=[(i,s['loss']) for i,s in enumerate(states)]
            graphic=chart([(points,color,'')],xlim=(0,9),ylim=(.5,1.7),xlabel='更新回数',ylabel='損失 L',
                          xticks=[0,3,6,9],yticks=[.5,1,1.5],width=340,height=270,
                          dots=[(*points[-1],color)])
            items.append(f'<div><h4>{label}</h4>{graphic}</div>')
        return figure('<div class="three-figures">'+''.join(items)+'</div>',
                       '図5　同じ6点、同じ初期値から9回更新した結果。縦軸・横軸の範囲は共通。学習率だけを変えている。')

    def method_figure():
        items=[]
        for size,color,name,comment,dash in [(6,BLUE,'バッチ','全体の勾配に沿って進む',''),(2,ORANGE,'ミニバッチ','向きを変えながら進む','6 3'),(1,PURPLE,'SGD','一点の影響で細かく揺れる','2 3')]:
            states=history(eta=.3,epochs=6,batch_size=size,fixed_bias=False)
            points=[s['theta'] for s in states]
            graphic=chart([(points,color,dash)],xlim=(-1.2,1.2),ylim=(-.5,.5),xlabel='重み w',ylabel='バイアス b',
                xticks=[-1,0,1],yticks=[-.5,0,.5],width=340,height=290,
                rings=[(*points[0],color)],dots=[(*points[-1],color)])
            items.append(f'<div><h4>{name}</h4>{graphic}<p>{comment}</p></div>')
        return figure('<div class="three-figures">'+''.join(items)+'</div>',
            '図6　パラメータ(w, b)が動いた軌跡。○が初期値、●が6巡後。同じデータ・初期値・学習率0.3、同じ座標軸で比較した。バッチは6回、ミニバッチは18回、SGDは36回更新している。')

    def softmax_figure():
        p,_=softmax_loss([2,1,0],0)
        bits=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 190" role="img" aria-label="スコア2、1、0に対応するクラス確率">']
        for i,v in enumerate(p):
            bits.append(f'<text x="20" y="{36+52*i}">クラス{i}</text><rect x="110" y="{15+52*i}" width="{v*450}" height="29" rx="3" fill="{BLUE}"/><text x="{120+v*450}" y="{36+52*i}">{v:.3f}</text>')
        return figure(''.join(bits)+'</svg>','図7　スコア[2, 1, 0]をsoftmaxで確率に変換した結果。3クラスの確率の合計は1になる。')

    def demo_data():
        states=history()
        for state in states:
            state['gradient']=float(loss_gradient(X,Y,state['theta'])[1][0])
        ws=np.linspace(-1.15,1.15,140)
        return dict(states=states,curve=[[float(w),loss_gradient(X,Y,[w,0])[0]] for w in ws],eta=.3)

    def learning_demo():
        root=Path(__file__).resolve().parent
        config=json.dumps(demo_data(),ensure_ascii=False,allow_nan=False).replace('</','<\\/')
        return (root/'player.html').read_text().replace('/* DATA */','const DATA = '+config+';').replace('/* PLAYER */',(root/'player.js').read_text())

    return (
        boundary_figure,
        learning_demo,
        loss_figure,
        method_figure,
        rate_figure,
        sigmoid_figure,
        softmax_figure,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.Html('''<style>
    .output > .markdown.prose{display:block}.markdown.prose{font-size:16px;line-height:1.95;max-width:860px;margin-inline:auto}
    .markdown h1{font-size:29px;line-height:1.5}.markdown h2{font-size:24px;margin-top:1.6em}.markdown h3{font-size:19px}
    .textbook-figure{max-width:860px;margin:18px auto 28px;color:#283b4a;background:#fff}
    .textbook-figure svg{width:100%;height:auto;display:block}.textbook-figure svg text{font:14px system-ui,sans-serif;fill:#526575}
    .textbook-figure figcaption{font-size:14px;line-height:1.8;color:#526575;margin:10px 0 0}
    .three-figures{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.three-figures h4{font:600 15px/1.6 system-ui;margin:0 0 7px}.three-figures p{font-size:14px;line-height:1.7;margin:6px 0}
    .small-figure{max-width:500px;margin:auto}iframe{display:block;border:0;width:100%;max-width:860px;margin-inline:auto}
    @media(max-width:800px){.three-figures{grid-template-columns:1fr}.three-figures>div{max-width:440px;width:100%;margin:0 auto}.markdown.prose{font-size:16px}}
    @media(max-width:600px){.textbook-figure svg text{font-size:20px}.three-figures svg text{font-size:14px}}
    </style>''')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 線形分類と勾配降下法

    この章では、データを二つのクラスに分けるモデルを作り、その重みを学習する方法を説明します。
    まず予測の計算を確かめ、次に「重みをどう変えれば、正解に近づくか」を考えます。
    尤度、損失、勾配降下法を順に結びつけ、最後に多クラスの分類へ広げます。

    読了の目安は約60分です。ベクトルの内積と、関数のグラフを知っていれば読み進められます。
    微分は傾きから説明します。途中の図を一度ずつ確かめながら読んでください。

    ## 1. 線形二値分類

    **分類**とは、入力がどのクラスに属するかを予測することです。
    たとえば、メールが迷惑メールかどうかを、文中の単語などの特徴から判断します。
    二つのクラスを扱うとき、正解のラベルを $y=0$ または $y=1$ と表します。
    この0と1はクラスを区別する記号です。

    入力を特徴量のベクトル $\boldsymbol{x}$ として表し、重み $\boldsymbol{w}$ と
    バイアス $b$ を使って、まず**スコア**を計算します。

    \[
    s=\boldsymbol{w}^{\top}\boldsymbol{x}+b
    \]

    特徴量が一つなら $s=wx+b$、二つなら $s=w_1x_1+w_2x_2+b$ です。
    重みは各特徴がスコアに与える影響、バイアスはスコア全体のずれを決めます。
    この段階のスコアは、負にも、1より大きくもなるため、そのまま確率とは呼べません。

    そこで、**sigmoid（シグモイド）関数**を通して、0から1の間の値に変換します。

    \[
    p=P(y=1\mid\boldsymbol{x})=\sigma(s)=\frac{1}{1+e^{-s}}
    \]

    $e$ は約2.718の定数で、指数関数の底です。$p$ はモデルが予測したクラス1の確率です。クラス0の確率は $1-p$ になります。
    スコアが大きいほどクラス1の確率が高くなり、スコア0では両クラスの確率が0.5になります。
    このモデルを**ロジスティック回帰**と呼びます。名前に「回帰」が入っていますが、ここでは分類に使います。
    """)
    return


@app.cell(hide_code=True)
def _(mo, sigmoid_figure):
    mo.Html(sigmoid_figure())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    確率から一つのクラスを選ぶときは、ここでは $p\geq0.5$ ならクラス1、
    $p<0.5$ ならクラス0と判定します。予測ラベルは $\hat y$ と書き、正解ラベル $y$ と区別します。

    **計算例。** 特徴量が $x=1$、重みが $w=-1$、バイアスが $b=0$ なら、
    $s=-1$、$p=\sigma(-1)\approx0.269$ なので、予測はクラス0です。
    正解が $y=1$ だったとすると、この予測は外れています。
    $w$ を少し大きくすれば、この点のスコアも $p$ も大きくなり、正解に与える確率が上がります。

    特徴量が二つの場合、判定が切り替わる場所は $w_1x_1+w_2x_2+b=0$ という直線になります。
    これを**決定境界**と呼びます。sigmoidのグラフは曲線ですが、入力の空間での境界は直線です。
    そのため、このモデルは線形分類器の一つに数えられます。
    """)
    return


@app.cell(hide_code=True)
def _(boundary_figure, mo):
    mo.Html(boundary_figure())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    一点だけを見れば、正解の確率を上げる方向を考えられそうです。
    しかし、同じ重みを多くの点に使うので、一点に都合のよい変更が別の点には不利になることもあります。
    **データ全体にとってよい重み**を選ぶために、まず予測のよさを一つの数で測ります。

    ## 2. 尤度と最尤推定

    ある入力に対し、モデルがクラス1の確率を $p=0.8$ と予測したとします。
    正解がクラス1なら、モデルは正解に0.8の確率を与えています。
    正解がクラス0なら、正解に与えた確率は $1-0.8=0.2$ です。

    データの $i$ 番目について、この「正解に与えた確率」を $q_i$ と書くと、
    二つの場合を次の式でまとめられます。

    \[
    q_i=p_i^{y_i}(1-p_i)^{1-y_i}
    \]

    $y_i=1$ を代入すると $q_i=p_i$、$y_i=0$ なら $q_i=1-p_i$ です。
    $p_i$ はいつもクラス1への確率、$q_i$ はその点の正解クラスへの確率、という違いがあります。

    入力とモデルの重みを与えたとき、各ラベルが独立に生じると仮定すると、
    観測された正解ラベル全体への確率は、それぞれの $q_i$ の積になります。
    この積を、重みの関数として見たものが**尤度（ゆうど）**です。
    $\theta=(\boldsymbol{w},b)$ とまとめて書けば、

    \[
    \mathcal{L}(\theta)=\prod_{i=1}^{N}q_i
    \]

    となります。$N$ はデータの点数、$\prod$ は全点について掛け合わせる記号です。
    データを固定して重みを変えると、予測する確率が変わり、尤度も変わります。
    尤度は「重みが正しい確率」ではなく、**その重みで観測済みの正解をどれだけ説明できるか**を表します。

    **計算例。** 二つの点の正解に、あるモデルは確率0.8と0.6を、
    別のモデルは0.9と0.7を与えたとします。

    | モデル | 一点目の $q_1$ | 二点目の $q_2$ | 尤度 $q_1q_2$ |
    |:--|--:|--:|--:|
    | A | 0.8 | 0.6 | 0.48 |
    | B | 0.9 | 0.7 | 0.63 |

    どちらも判定は二点とも正解ですが、Bの方が正解ラベルに高い確率を与えています。
    この基準で、尤度をできるだけ大きくする重みを探すことを**最尤推定**と呼びます。

    ## 3. 交差エントロピー

    尤度は多くの確率の積なので、データが増えると非常に小さな値になります。
    計算と微分を扱いやすくするため、積の対数を取ります。
    対数は単調に増える関数なので、尤度を最大にする重みは、対数尤度も最大にします。

    \[
    \log\mathcal{L}(\theta)=\sum_{i=1}^{N}\log q_i
    \]

    さらに符号を反転して「小さいほどよい」量にし、データ数 $N$ で割って平均を取ります。
    得られる**損失**が、二値分類の交差エントロピーです。以下の $\log$ は自然対数です。

    \[
    \ell_i=-y_i\log p_i-(1-y_i)\log(1-p_i)=-\log q_i
    \]

    \[
    L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell_i
    \]

    $\ell_i$ は一点の損失、$L$ はデータ全体の平均損失です。
    尤度を最大にすることと、この平均損失を最小にすることは同じ重みの選び方になります。
    """)
    return


@app.cell(hide_code=True)
def _(loss_figure, mo):
    mo.Html(loss_figure())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    正解に与える確率 $q$ が0.2、0.5、0.8のとき、損失はそれぞれ約1.609、0.693、0.223です。
    判定が同じでも、正解への確率が上がれば損失は下がります。
    正解率だけでは区別できなかった予測の変化を、連続的な数値として測れるようになりました。

    ここからは、次の6点を共通の例として使います。特徴量は一つです。

    | 特徴量 $x$ | −3 | −2 | −1 | 1 | 2 | 3 |
    |:--|--:|--:|--:|--:|--:|--:|
    | 正解 $y$ | 0 | 1 | 0 | 1 | 0 | 1 |

    正解ラベルは途中で入れ替わっているため、一つの境界ですべての点を正しく分類することはできません。
    それでも、6点への予測をまとめた平均損失は計算できます。
    この損失が小さくなるように、重みを調整していきます。

    ## 4. 微分と勾配降下法

    まず、バイアスを $b=0$ に固定して、重み $w$ 一つだけを変えます。
    すると損失は $L(w)$ という一変数の関数になります。
    図4の横軸は重み $w$、縦軸は6点の平均損失です。横軸は入力 $x$ ではありません。

    関数の**微分** $L'(w)$ は、その場所でのグラフの傾きです。
    重みを少しだけ $\Delta w$ 動かしたときの損失の変化は、

    \[
    L(w+\Delta w)-L(w)\approx L'(w)\Delta w
    \]

    と近似できます。傾きが負なら、右へ動くと損失が下がります。
    傾きが正なら、左へ動くと下がります。
    このように、**傾きと逆の向きに重みを動かす**のが勾配降下法の基本です。

    \[
    w_{t+1}=w_t-\eta L'(w_t)
    \]

    $t$ は更新回数、$\eta>0$ は一回に動かす量を調節する**学習率**です。
    一回動かしたら、新しい位置で傾きを計算し直し、同じ操作を繰り返します。

    このモデルの一点の損失を微分すると、次の簡単な形になります。

    \[
    \frac{\partial\ell_i}{\partial w}=(p_i-y_i)x_i
    \]

    これは、$\partial\ell_i/\partial s_i=p_i-y_i$ と
    $\partial s_i/\partial w=x_i$ を掛け合わせたものです。
    全体の平均損失の微分は、各点から得た値の平均になります。

    \[
    L'(w)=\frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)x_i
    \]

    **最初の更新。** 6点の例で $w=-1$ とすると、平均損失は約1.496、傾きは約−1.117です。
    学習率を0.3にすると、

    \[
    w_{1}=-1-0.3\times(-1.117)\approx-0.665
    \]

    となり、更新後の平均損失は約1.146に下がります。
    次の図で「進む」を押すと、この更新を一回ずつ行います。
    **点が右へ進むにつれて、傾きと一回の移動量が小さくなること**を確かめてください。
    """)
    return


@app.cell
def _(learning_demo, mo):
    mo.iframe(learning_demo(), height="430px")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    図4では、初期状態から9回の更新で損失が約1.496から0.645に下がります。
    損失が0にならないのは、すべての正解に確率1を与えられるデータとモデルではないためです。
    学習は「すべてを正解にするまで動かすこと」とは限りません。

    重みが複数ある場合も、考え方は同じです。
    ほかの重みを固定して一つの重みに関する傾きを求める計算を**偏微分**と呼びます。
    パラメータ $\theta=(w_1,\ldots,w_d,b)$ の偏微分を並べたベクトルが**勾配** $\nabla_\theta L$ です。

    \[
    \theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)
    \]

    二値分類では、$w_j$ に対する偏微分は $(p_i-y_i)x_{ij}$ の平均、
    $b$ に対する偏微分は $p_i-y_i$ の平均になります。
    すべての成分を更新前の同じパラメータから計算し、まとめて更新します。
    一つの傾きを使った図4は、このベクトルの更新を一方向だけに限定した例です。

    ## 5. 学習率

    勾配から分かるのは、現在の位置の近くで損失が増える向きと、その変化の大きさです。
    遠くまで同じ傾きが続くとは限らないため、逆向きに動かせば必ず損失が下がるわけではありません。
    どこまで動かすかは、学習率 $\eta$ によって変わります。

    図5は、図4と同じデータ・初期値で、学習率だけを変えて9回更新した結果です。
    ここでもバイアスは0に固定しています。
    """)
    return


@app.cell(hide_code=True)
def _(mo, rate_figure):
    mo.Html(rate_figure())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $\eta=0.1$ では、一回の変化が小さく、損失がゆっくり下がっています。
    $\eta=1$ では、この例では少ない更新で低い損失に近づきます。
    $\eta=3$ では、最初に谷を大きく飛び越え、損失が約1.496から1.601へ増えます。
    その後も損失が上下し、更新が安定しません。

    どの値が適切かは、データの尺度や損失の形によって変わります。
    特徴量の値を大きくすれば、同じ重みでもスコアや勾配が変わります。
    したがって「学習率は1にする」という決まりはありません。
    損失が下がる様子を見ながら調整し、学習が進むにつれて学習率を小さくする方法もあります。

    ## 6. バッチ・ミニバッチ・SGD

    ここまでは、一回の更新のたびに、6点すべてから勾配を計算しました。
    この方法を**バッチ勾配降下法**と呼びます。
    データ数が多いと、たった一回の更新にも全データを処理する必要があります。

    全体の一部だけから勾配を計算して更新するのが**ミニバッチ勾配降下法**です。
    さらに、一点ずつ選んで更新する方法を**確率的勾配降下法（SGD）**と呼びます。
    この章では三つを区別するため、SGDを一点ずつ更新する意味で使います。
    実装や文献では、ミニバッチを使う方法も広くSGDと呼ばれます。

    | 方法 | 一回の更新に使うデータ | 6点を一巡するときの更新回数 |
    |:--|:--|--:|
    | バッチ勾配降下法 | 全6点 | 1回 |
    | ミニバッチ勾配降下法 | 2点ずつ | 3回 |
    | 確率的勾配降下法（SGD） | 1点ずつ | 6回 |

    一回に選んだデータの集合を $B$ とすると、更新に使う勾配は次の平均です。

    \[
    g_B(\theta)=\frac{1}{|B|}\sum_{i\in B}\nabla_\theta\ell_i(\theta)
    \]

    \[
    \theta_{t+1}=\theta_t-\eta g_B(\theta_t)
    \]

    更新式の形は共通で、勾配を計算するデータの数が異なります。
    ミニバッチやSGDでは、データをランダムに並べ替え、順に使う方法がよく使われます。
    全データを一巡することを**1エポック**と呼びます。

    図6では、バイアスも動かし、横軸を $w$、縦軸を $b$ として、その軌跡を描いています。
    図4・5の損失のグラフとは軸が異なります。
    初期値はどれも $(w,b)=(-1,0)$、学習率は0.3です。各エポックの並べ替え順もそろえました。
    """)
    return


@app.cell(hide_code=True)
def _(method_figure, mo):
    mo.Html(method_figure())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    バッチでは全体の勾配を使うので、この例ではまっすぐ進みます。
    データが左右対称に配置されているため、バイアスの勾配が打ち消し合って0になるからです。
    バッチの軌跡がいつでも直線になる、という意味ではありません。

    ミニバッチは、選ばれた2点によって勾配の向きが変わり、くねくねと進みます。
    SGDは一点の影響を直接受けるので、この例ではさらに細かく揺れています。
    選んだデータにとって損失を下げる更新でも、全データの損失が毎回下がるとは限りません。

    この揺れは、勾配をデータの一部で近似することから生じます。
    一回に使う点を増やすと、一般に全体の勾配に近い値が得られ、揺れは小さくなります。
    ただし、実際の軌跡はデータと並び順にも依存します。

    図は同じ6エポックで比べていますが、更新回数は6回・18回・36回と異なります。
    軌跡の長さや終点だけから、実行時間の速さや最終的な性能を順位づけることはできません。
    実際の学習では、一回の計算量、利用できるメモリ、勾配の揺れを考慮してミニバッチの大きさを選びます。

    ## 7. 多クラス分類

    クラスが三つ以上でも、スコアを確率に変え、正解への確率から損失を計算して、
    勾配で重みを更新する流れは変わりません。
    クラスが $K$ 個あるときは、クラスごとに重みとバイアスを持ち、$K$ 個のスコアを計算します。

    \[
    s_k=\boldsymbol{w}_k^{\top}\boldsymbol{x}+b_k
    \]

    先ほどまでの二値分類と同様に考えて各スコアをばらばらにsigmoidに通すと、確率の合計が1になるとは限りません。
    一つの正解クラスを選ぶ分類では、**softmax（ソフトマックス）関数**を使います。

    \[
    p_k=\frac{e^{s_k}}{\sum_{j=0}^{K-1}e^{s_j}}
    \]

    それぞれの$k$に対応するスコアの指数$e^{s_k}$を
    スコアの指数の合計$\sum_{j=0}^{K-1}e^{s_j}で割った$p_k$ は正となり、合計は1になります。
    最も大きいスコアのクラスが、最も大きい確率のクラスになります。
    """)
    return


@app.cell(hide_code=True)
def _(mo, softmax_figure):
    mo.Html(softmax_figure())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    図7の例では、スコア $[2,1,0]$ から、確率約 $[0.665,0.245,0.090]$ が得られます。
    予測するラベルはクラス0です。
    正解がクラス0なら一点の損失は $-\log0.665\approx0.408$、
    正解がクラス2なら $-\log0.090\approx2.408$ になります。

    つまり、多クラスでも損失は**正解クラスに与えた確率の負の対数**です。

    \[
    \ell_i=-\log p_{i,y_i}
    \]

    データ全体でこの損失を平均し、勾配を求め、重みとバイアスを更新します。
    基本的にやることは二値分類と変わらず、
    バッチ・ミニバッチ・SGDの区別も、学習率の役割も、そのまま当てはまります。

    ## 8. まとめ

    線形分類器の学習は、次の計算を繰り返すことです。

    1. 入力と現在の重みからスコアを計算し、確率に変換する。
    2. 正解に与えた確率から、交差エントロピーを計算する。
    3. 損失の勾配を求め、学習率を掛けて、逆向きに重みを更新する。

    尤度は正解ラベル全体の説明のよさを表し、最尤推定が重みを選ぶ基準になります。
    その計算を負の対数で書き換えたものが交差エントロピーです。
    勾配降下法は、この損失を小さくするための更新方法です。

    ここで小さくしてきたのは、重みの更新に使った**訓練データ**の損失です。
    未知の入力にもよい予測ができるかは、更新に使わなかった**検証データ**で確かめる必要があります。
    訓練での損失の減少と、未知のデータでの正解率の向上は、同じことではありません。
    次の章では、手書き数字を題材に、多クラス分類器の学習と評価を扱います。

    ### 参考文献

    岡崎直観ほか『機械学習帳』：
    [二値分類](https://chokkan.github.io/mlnote/classification/01binary.html)、
    [多クラス分類](https://chokkan.github.io/mlnote/classification/02multi.html)、
    [確率的勾配降下法](https://chokkan.github.io/mlnote/regression/04sgd.html)。
    """)
    return


if __name__ == "__main__":
    app.run()
