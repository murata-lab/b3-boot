import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="03 線形分類と勾配降下法", css_file="notebook.css")


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np

    return Path, json, mo, np


@app.cell(hide_code=True)
def _(Path, json, mo):
    _root = Path(__file__).resolve().parent
    _shell = (_root / "figure.html").read_text(encoding="utf-8")
    _script = (_root / "figures.js").read_text(encoding="utf-8")

    def figure(name, data, caption="", height=380):
        """figures.js の図を一つ埋め込む。高さは表示後に内容へ合わせて自動で変わる。"""
        config = json.dumps({"name": name, "data": data, "caption": caption}, ensure_ascii=False, allow_nan=False)
        html = _shell.replace("/* CONFIG */", "const CONFIG = " + config.replace("</", "<\\/") + ";")
        return mo.iframe(html.replace("/* FIGURES */", _script), height=f"{height}px")

    def check(choice, answer, explanation):
        """確認問題の答え合わせ。選ぶまでは何も表示しない。"""
        if choice.value is None:
            return mo.md("")
        correct = choice.value == answer
        head = "**正解です。**" if correct else f"**正解は「{answer}」です。**"
        return mo.callout(mo.md(head + " " + explanation), kind="success" if correct else "warn")

    return check, figure


@app.cell(hide_code=True)
def _(np):
    def sigmoid(s):
        return np.exp(-np.logaddexp(0.0, -np.asarray(s, dtype=float)))

    def cross_entropy(w, b, x, y):
        """平均の交差エントロピー。s = wx + b のとき −log q = log(1 + e^s) − ys なので、log(0) を避けてこう計算する。"""
        s = w * np.asarray(x) + b
        return float(np.mean(np.logaddexp(0.0, s) - y * s))

    def slope(w, x, y):
        """b = 0 のときの損失の傾き L'(w)。"""
        return float(np.mean((sigmoid(w * x) - y) * x))

    def descend(x, y, eta, steps, w=0.0):
        """b = 0 に固定し、w だけを勾配降下法で更新する。更新前と各更新後の w・損失・傾きを返す。"""
        states = []
        for _ in range(steps + 1):
            states.append({"w": w, "loss": cross_entropy(w, 0.0, x, y), "grad": slope(w, x, y)})
            w = w - eta * slope(w, x, y)
        return states

    def fit_logistic(features, y, bias=True):
        """交差エントロピーが最小になる重みを（図を作るために）ニュートン法で求める。戻り値は [w1, w2, ..., b]。"""
        design = np.column_stack([*features, np.ones(len(y))] if bias else features)
        theta = np.zeros(design.shape[1])
        for _ in range(50):
            p = sigmoid(design @ theta)
            theta -= np.linalg.solve((design * (p * (1 - p))[:, None]).T @ design, design.T @ (p - y))
        return theta

    return cross_entropy, descend, fit_logistic, sigmoid


@app.cell(hide_code=True)
def _(np, sigmoid):
    # 湿度と雨の例（説明用に作ったデータ）。ある町の6月の30日間の、朝の湿度（%）と、その日に雨が降ったか（1 = 降った）。
    # 本当の雨の確率は σ(0.12 × (湿度 − 70)) とした。1節で「まだ分からない」とする湿度78%の近くと、
    # 境目にあたる70%ちょうどの日は作らない。同じ行（降った・降らなかった）で2%未満の点は、右隣の空いている湿度へずらす。
    HUMID_CENTER, QUERY = 70, 78
    _allowed = [h for h in range(45, 96) if h != HUMID_CENTER and abs(h - QUERY) >= 3]
    _rng = np.random.default_rng(3094)
    HUMIDITY = np.sort(_rng.choice(_allowed, 30)).astype(float)
    RAIN = (_rng.random(30) < sigmoid(0.12 * (HUMIDITY - HUMID_CENTER))).astype(float)
    for _c in (0.0, 1.0):
        _idx = np.where(RAIN == _c)[0]
        _v = HUMIDITY[_idx]
        for _i in range(1, len(_v)):
            if _v[_i] < _v[_i - 1] + 2:
                _v[_i] = min(h for h in _allowed if h >= _v[_i - 1] + 2)
        HUMIDITY[_idx] = _v
    # 3節以降の入力 x は「湿度 − 70」
    X = HUMIDITY - HUMID_CENTER
    DAY_POINTS = [[int(h), int(r)] for h, r in zip(HUMIDITY, RAIN)]
    # 図2：湿度の区間（45〜54%、55〜64%、…、85〜95%）ごとの、雨が降った日の割合
    RAIN_BINS = []
    for _lo, _hi in [(45, 55), (55, 65), (65, 75), (75, 85), (85, 95)]:
        _in = (HUMIDITY >= _lo) & ((HUMIDITY < _hi) if _hi < 95 else (HUMIDITY <= _hi))
        RAIN_BINS.append({"lo": _lo, "hi": _hi, "days": int(_in.sum()), "rain": int(RAIN[_in].sum()),
                          "rate": float(RAIN[_in].mean())})
    return DAY_POINTS, HUMIDITY, HUMID_CENTER, QUERY, RAIN, RAIN_BINS, X


@app.cell(hide_code=True)
def _(HUMIDITY, RAIN, X, cross_entropy, fit_logistic):
    # 1節の曲線（湿度そのものを入力に、w と b を両方学習）と、3節以降で使う b = 0 のときの最適な w
    FIT_W, FIT_B = (float(v) for v in fit_logistic([HUMIDITY], RAIN))
    (BEST_W,) = (float(v) for v in fit_logistic([X], RAIN, bias=False))
    BEST_LOSS = cross_entropy(BEST_W, 0.0, X, RAIN)
    return BEST_LOSS, BEST_W, FIT_B, FIT_W


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <header class="hero">
      <p class="eyebrow">機械学習入門</p>
      <h1>線形分類と勾配降下法</h1>
      <p>雨が降るかどうかを確率で予測するモデルを作り、その重みを「坂を下る」ことで学習します。</p>
    </header>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    雨が降るかどうかのように、**どの種類（クラス）に当たるか**を予測する問題を**分類（classification）**と呼びます。
    朝の湿度から雨の確率を出し、予測のずれを小さくするように重みを学習します。

    **この章の目標：** 分類の損失と、勾配・学習率が重みの更新にどう関わるかを説明することです。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 雨が降るかを確率で予測する

    天気予報の「降水確率70%」は、雨が降るか降らないかを言い切らずに、**確率**で答えています。
    同じような空模様でも、降る日と降らない日があるからです。

    下の図は、ある町の6月の30日間について、朝の湿度と、その日に雨が降ったかを並べたものです（説明用に作ったデータです）。
    """)
    return


@app.cell(hide_code=True)
def _(DAY_POINTS, QUERY, figure):
    figure("rainDays", {"points": DAY_POINTS, "query": QUERY},
           f"図1　30日間の朝の湿度（横軸）と、その日に雨が降ったか（上の段が降った日、下の段が降らなかった日）。", height=310)
    return


@app.cell(hide_code=True)
def _(HUMIDITY, HUMID_CENTER, QUERY, RAIN, mo):
    _dry_rain = int(((HUMIDITY < HUMID_CENTER) & (RAIN == 1)).sum())
    _wet_dry = int(((HUMIDITY > HUMID_CENTER) & (RAIN == 0)).sum())
    mo.md(rf"""
    湿度が高い日ほど雨が多いのですが、境目ははっきりしません。
    湿度が{HUMID_CENTER}%に届かなかったのに雨が降った日が{_dry_rain}日、{HUMID_CENTER}%を超えていたのに降らなかった日が{_wet_dry}日あります。
    では、**湿度{QUERY}%の朝、雨が降る確率**はどれくらいでしょうか。

    手掛かりとして、湿度を10%ごとの区間に分け、区間ごとに、雨が降った日の割合を数えてみます。
    """)
    return


@app.cell(hide_code=True)
def _(RAIN_BINS, figure):
    figure("rainRates", {"bins": RAIN_BINS},
           "図2　湿度の区間ごとの、雨が降った日の割合。棒の上の数字は「雨が降った日数 / その区間の日数」。", height=340)
    return


@app.cell(hide_code=True)
def _(RAIN_BINS, mo):
    _few = min(b["days"] for b in RAIN_BINS)
    mo.md(rf"""
    湿度が高い区間ほど、雨が降った日の割合が高くなっています。この割合が、その湿度で雨が降る確率の目安になります。
    ただし、区間の区切り方を変えると値が変わりますし、{_few}日しかない区間の割合はあまり当てになりません。
    そこで、湿度から**滑らかに確率を出す式**を作ります。

    雨が降った・降らなかったのように、答えが二つのクラスのどちらかになる問題を**二値分類**と呼びます。
    雨が降った日を $y=1$、降らなかった日を $y=0$ と表し、モデルには $y$ そのものではなく、**雨が降る確率 $p$**（0から1の値）を出させます。

    ### スコアを確率に変える

    02と同じように、まず入力 $x$（湿度）から、直線の式で**スコア**を計算します。

    $$s = wx + b$$

    スコアは、雨が降りそうな日ほど大きくなる数です。ただし直線なので、負にも1より大きくもなり、そのままでは確率になりません。
    そこで、どんな数でも0から1の間に押し込む**sigmoid（シグモイド）関数** $\sigma$ に通します。

    $$p = \sigma(s) = \frac{{1}}{{1+e^{{-s}}}}$$
    """)
    return


@app.cell(hide_code=True)
def _(figure):
    figure("sigmoid", {"marks": [-2, 0, 2]},
           "図3　sigmoid関数。横軸がスコア、縦軸が確率。スコアがどんな値でも、確率は0と1の間に入る。", height=330)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    スコアが0なら確率はちょうど0.5、スコアが大きいほど1に、小さいほど0に近づきます。

    $w$ と $b$ をデータから学習すると（学習の方法は3〜4節で説明します）、次の曲線が得られます。
    縦軸を「雨の確率」と読めば、図1の点と同じ座標に曲線を描けます。図2の区間ごとの割合も、薄い棒で重ねました。
    """)
    return


@app.cell(hide_code=True)
def _(DAY_POINTS, FIT_B, FIT_W, QUERY, RAIN_BINS, figure):
    figure("rainModel", {"points": DAY_POINTS, "bins": RAIN_BINS, "w": FIT_W, "b": FIT_B, "query": QUERY},
           f"図4　学習したモデル $p$ = σ({FIT_W:.3f}$x$ − {-FIT_B:.2f}) が出す雨の確率（$x$ は湿度）。縦の点線が、予報が切り替わる境目。", height=400)
    return


@app.cell(hide_code=True)
def _(FIT_B, FIT_W, QUERY, mo, np):
    _p = 1 / (1 + np.exp(-(FIT_W * QUERY + FIT_B)))
    mo.md(rf"""
    湿度{QUERY}%の朝に雨が降る確率は、約{_p * 100:.0f}%と予測できました。
    曲線は、区間ごとの割合の段差をならすように通っています。区間に分けなくても、どの湿度にも確率を出せます。

    確率から「雨」「降らない」を一つに決めるときは、$p \geq 0.5$ なら「雨」と予報します。
    $p = 0.5$ になるのはスコアが0のとき、つまり $x = -b/w \approx {-FIT_B / FIT_W:.1f}$ % のところで、ここが予報の境目です。

    この例では $w$ は正で、絶対値が大きいほど曲線の変化が急になります。$w$ が負なら、湿度が高いほど確率が下がります。
    境目 $-b/w$ の位置は、$w$ と $b$ の比で決まります。

    このように、スコアをsigmoidで確率に変えるモデルを**ロジスティック回帰（logistic regression）**と呼びます。
    名前に「回帰」と付きますが、分類に使うモデルです。

    では、入力が二つになると、予報の境目はどうなるでしょうか。
    """)
    return


@app.cell(hide_code=True)
def _(fit_logistic, np, sigmoid):
    # 湿度と気圧の変化の例（説明用に作ったデータ、40日分）。本当の雨の確率は σ(0.12 × (湿度 − 70) − 0.3 × 気圧の変化)。
    # 近すぎる点は作り直す。
    _rng = np.random.default_rng(133)
    _pts = []
    while len(_pts) < 40:
        _p = np.array([float(_rng.integers(45, 96)), round(float(_rng.uniform(-8, 8)), 1)])
        if any(np.hypot((_p[0] - q[0]) / 50, (_p[1] - q[1]) / 16) < 0.06 for q in _pts):
            continue
        _pts.append(_p)
    _pts = np.array(_pts)
    _y = (_rng.random(40) < sigmoid(0.12 * (_pts[:, 0] - 70) - 0.3 * _pts[:, 1])).astype(float)
    TWO_THETA = [float(v) for v in fit_logistic([_pts[:, 0], _pts[:, 1]], _y)]
    TWO_POINTS = [[int(a), float(b), int(c)] for (a, b), c in zip(_pts, _y)]
    return TWO_POINTS, TWO_THETA


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 特徴量が二つのとき：決定境界

    雨の予報には、湿度のほかに**気圧の変化**も手掛かりになります。気圧が下がる（低気圧が近づく）と雨になりやすいことは、天気予報でもよく耳にします。
    朝の湿度を $x_1$、前日からの気圧の変化（hPa）を $x_2$ とし、スコアを次の式で計算します（別の40日のデータを用意しました）。

    $$s = w_1 x_1 + w_2 x_2 + b$$

    予測に使う入力の一つひとつを**特徴量**と呼ぶのは、02と同じです。
    学習したモデルで「雨」と予報される範囲に、色を付けたのが次の図です。
    """)
    return


@app.cell(hide_code=True)
def _(TWO_POINTS, TWO_THETA, figure):
    figure("boundary2d", {"points": TWO_POINTS, "theta": TWO_THETA},
           f"図5　40日間の朝の湿度と気圧の変化（前日比）、その日に雨が降ったか。黒い直線が、学習したモデル（$w$₁ = {TWO_THETA[0]:.2f}、$w$₂ = {TWO_THETA[1]:.2f}、$b$ = {TWO_THETA[2]:.2f}）の決定境界。".replace("-", "−"),
           height=560)
    return


@app.cell(hide_code=True)
def _(TWO_THETA, mo):
    _w1, _w2, _ = TWO_THETA
    mo.md(rf"""
    予報が切り替わるのは $p = 0.5$、つまりスコアが $w_1 x_1 + w_2 x_2 + b = 0$ となる場所で、これは**直線**になります。
    この線を**決定境界（decision boundary）**と呼びます。

    学習した重みは $w_1 = {_w1:.2f}$、$w_2 = {_w2:.2f}$ でした。湿度が1%上がるとスコアが {_w1:.2f} 増え、気圧が1hPa**下がる**とスコアが {-_w2:.2f} 増えます。
    そのため境界は右上がりになり、湿度が高い日でも気圧が上がっていれば「降らない」側に、湿度が低めでも気圧が大きく下がっていれば「雨」の側に入ります。

    確率の曲線（sigmoid）は曲がっていますが、予報の境界は直線（特徴量が三つ以上なら平面）です。
    このようなモデルを**線形分類器**と呼びます。直線では分けられないデータをどう扱うかは、04のニューラルネットワークで学びます。

    ここからは、湿度だけを使う例に戻ります。図4の $w$ と $b$ は、どうやって選んだのでしょうか。
    """)
    return


@app.cell(hide_code=True)
def _(FIT_B, FIT_W, HUMID_CENTER, mo):
    mo.md(rf"""
    ## 3. 予測のよさを測る：交差エントロピー

    02と同じく、予測のよさを一つの数（損失）で表し、それが最も小さいモデルを選びます。
    では、分類ではどんな数を使えばよいでしょうか。

    話を簡単にするため、ここからはパラメータを $w$ 一つにします。
    湿度そのものではなく、**湿度から{HUMID_CENTER}を引いた値**を入力 $x$ にします（湿度78%なら $x = 8$）。
    すると湿度{HUMID_CENTER}%の日のスコアはちょうど $b$ になります。図4で学習した境目もほぼ{HUMID_CENTER}%（{-FIT_B / FIT_W:.1f}%）だったので、
    湿度{HUMID_CENTER}%の日を五分五分とみなして $b = 0$ に固定します。

    次の三つのモデルは、$w$ だけが違います。
    """)
    return


@app.cell(hide_code=True)
def _(BEST_W, DAY_POINTS, HUMID_CENTER, RAIN, X, cross_entropy, figure, np):
    _acc = float(np.mean((X > 0) == (RAIN == 1)))
    THREE_W = [0.01, round(BEST_W, 2), 0.8]
    THREE_LOSS = [cross_entropy(w, 0.0, X, RAIN) for w in THREE_W]
    ACCURACY = _acc
    figure("threeModels", {"points": DAY_POINTS, "center": HUMID_CENTER, "models": [
        {"title": f"緩い（$w$ = {THREE_W[0]}）", "w": THREE_W[0], "stats": f"正解率 {_acc:.0%}"},
        {"title": f"中くらい（$w$ = {THREE_W[1]}）", "w": THREE_W[1], "stats": f"正解率 {_acc:.0%}"},
        {"title": f"急（$w$ = {THREE_W[2]:g}）", "w": THREE_W[2], "stats": f"正解率 {_acc:.0%}"},
    ]}, f"図6　$b$ = 0 のまま $w$ だけを変えた三つのモデル（$x$ = 湿度 − {HUMID_CENTER}）。縦の線は、正解（0か1）と予測した確率とのずれ。", height=380)
    return ACCURACY, THREE_LOSS, THREE_W


@app.cell(hide_code=True)
def _(ACCURACY, HUMIDITY, HUMID_CENTER, RAIN, THREE_W, mo):
    _worst = int(HUMIDITY[RAIN == 0].max())
    mo.md(rf"""
    三つとも、湿度が{HUMID_CENTER}%を超えた日を「雨」と予報するので、予報はまったく同じです。
    **正解率**（予報が当たった日の割合）はどれも{ACCURACY:.0%}ですが、よいモデルは真ん中に見えます。

    - **左**（$w={THREE_W[0]}$）は、雨の日にも降らなかった日にも、0.5に近い確率しか出していません。自信がなさすぎます。
    - **右**（$w={THREE_W[2]:g}$）は、湿度{_worst}%で降らなかった日にも「ほぼ確実に雨」と予測しています。自信を持って外しています。

    正解率ではこの違いを区別できません。そこで、**正解に与えた確率**を使います。
    $i$ 番目の日について、モデルが正解のクラスに与えた確率を $q_i$ とします。
    雨が降った日（$y_i=1$）なら $q_i = p_i$、降らなかった日（$y_i=0$）なら $q_i = 1-p_i$ です。
    図6の縦の線の長さは $1 - q_i$ にあたり、短いほど正解に高い確率を与えています。

    1日分の損失を $-\log q_i$ とします（$\log$ は自然対数）。グラフにすると次のようになります。
    """)
    return


@app.cell(hide_code=True)
def _(figure):
    figure("logLoss", {"marks": [0.1, 0.5, 0.9]},
           "図7　正解に与えた確率 $q$ と、1日分の損失 −log $q$。正解に確率1を与えれば損失は0。", height=330)
    return


@app.cell(hide_code=True)
def _(THREE_LOSS, mo):
    mo.md(rf"""
    正解に確率1を与えれば損失は0で、確率が0に近づくほど損失は急に大きくなります。
    「雨の確率90%」と予測して降らなかった日（$q = 0.1$）の損失は2.30で、「50%」と予測して降らなかった日（$q=0.5$）の0.69の3倍以上です。
    **自信を持って外すほど、大きな損失になります。**

    これを全日で平均したものが、分類で最もよく使われる損失、**交差エントロピー（cross-entropy）**です。$N$ は日数です。

    $$L = \frac{{1}}{{N}}\sum_{{i=1}}^{{N}} \left(-\log q_i\right)$$

    $q_i$ を $p_i$ と $y_i$ で書くと、1日分の損失は次のようにも書けます。$y_i$ が1なら前の項だけ、0なら後ろの項だけが残ります。

    $$-\log q_i = -y_i \log p_i - (1-y_i)\log(1-p_i)$$

    図6の三つのモデルの交差エントロピーは、左から {THREE_LOSS[0]:.3f}、{THREE_LOSS[1]:.3f}、{THREE_LOSS[2]:.3f} で、真ん中が最も小さくなります。
    正解率では区別できなかった違いを、交差エントロピーなら数値で比べられます。

    正解率には、もう一つ困ることがあります。$b=0$ のままなら、$w$ が正である限り、どう変えても予報は変わらず、正解率も変わりません。
    これでは、$w$ をどちらへ動かせばよいかの手掛かりになりません。
    交差エントロピーは $w$ を少し動かすと少し変わるので、次の節で使う「傾き」を計算できます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"なぜ −log を使うのか：尤度と最尤推定（任意）": mo.md(r"""
    モデルが正しいとすると、30日の天気がちょうど観測したとおりになる確率は、各日の $q_i$ の積 $q_1 q_2 \cdots q_N$ です
    （一日ごとの天気が独立に決まると考えます）。
    これをパラメータの関数として見たものを**尤度（ゆうど、likelihood）**と呼び、尤度が最も大きくなるパラメータを選ぶ方法を**最尤推定**と呼びます。
    「観測したデータを最もよく説明するパラメータを選ぶ」という考え方です。

    たくさんの確率の積はとても小さな数になって扱いにくいので、対数を取って和にします（対数は増加関数なので、最大になる場所は変わりません）。
    さらに符号を反転し、日数 $N$ で割ると、交差エントロピー $L$ になります。

    $$-\frac{1}{N}\log\left(q_1 q_2 \cdots q_N\right) = \frac{1}{N}\sum_{i=1}^{N}\left(-\log q_i\right) = L$$

    つまり、**交差エントロピーを最小にすることは、最尤推定と同じパラメータの選び方**です。
    """)})
    return


@app.cell(hide_code=True)
def _(DESCENT_ETA, mo):
    mo.md(rf"""
    ## 4. 勾配降下法：傾きを頼りに坂を下る

    交差エントロピーが最も小さくなる $w$ を探します。
    02の最小二乗法では、損失の谷底の位置を式で一度に求めました。交差エントロピーには、そのような式がありません。
    代わりに、**今いる場所の坂の傾きを調べ、下る向きに少しずつ動く**ことを繰り返します。

    $w$ を少しだけ増やしたとき、損失がどれだけの割合で変わるかが、損失の**傾き（微分）** $L'(w)$ です。

    - 傾きが**負**なら、$w$ を大きくすると損失が下がる。
    - 傾きが**正**なら、$w$ を小さくすると損失が下がる。

    どちらの場合も、**傾きと逆の向き**に動かせばよいことになります。これを式にしたのが**勾配降下法（gradient descent）**の更新です。

    $$w \leftarrow w - \eta\, L'(w)$$

    $\eta$（イータ）は1回に動かす量を決める正の数で、**学習率（learning rate）**と呼びます。
    傾きが急なほど大きく、緩やかなほど小さく動きます。
    この例では、傾きは次の式で計算できます（導き方は下の折りたたみにあります）。

    $$L'(w) = \frac{{1}}{{N}}\sum_{{i=1}}^{{N}} (p_i - y_i)\,x_i$$

    $p_i - y_i$ は、予測した確率と正解のずれです。
    たとえば、湿度85%（$x = 15$）で雨が降った日に低い確率しか出していないと、$(p_i - 1) \times 15$ は負になり、傾きを負の側へ引っ張ります。
    つまり「$w$ を大きくして、この日の雨の確率を上げよ」という向きです。全日分のこうした要求を平均したものが傾きです。

    1回の更新は、次の3段でできています。

    1. **予測して損失を計算**：今の $w$ で各日の雨の確率 $p$ を計算し、正解と比べて損失 $L$ を求める。
    2. **傾きを求める**：今の $w$ での損失の傾き $L'(w)$ を計算する。
    3. **$w$ を更新する**：$w \leftarrow w - \eta L'(w)$ で $w$ を動かす。

    そして、新しい $w$ でまた1.から繰り返します。下の図で、$w = 0$ から学習率 $\eta = {DESCENT_ETA}$ で、この3段を1段ずつ進めてみましょう。
    「重み w と損失 L」の図は、横軸が重み $w$、縦軸が損失です。
    """)
    return


@app.cell(hide_code=True)
def _(BEST_LOSS, BEST_W, DAY_POINTS, HUMID_CENTER, RAIN, X, cross_entropy, descend, figure, np):
    DESCENT_ETA = 0.01
    DESCENT = descend(X, RAIN, DESCENT_ETA, 12)
    _ws = np.linspace(-0.01, 0.16, 171)
    figure("descent", {
        "points": DAY_POINTS, "center": HUMID_CENTER, "states": DESCENT, "eta": DESCENT_ETA,
        "curve": [[float(w), cross_entropy(w, 0.0, X, RAIN)] for w in _ws],
        "best": [BEST_W, BEST_LOSS], "wlim": [-0.01, 0.16], "llim": [0.44, 0.74],
        "wticks": [0, 0.05, 0.1, 0.15], "wdigits": 2, "lticks": [0.5, 0.6, 0.7],
    }, f"図8　勾配降下法の1回の更新を3段に分けて進める（学習率 $η$ = {DESCENT_ETA}、$b$ = 0 に固定）。灰色の曲線は説明のために描いた損失の全体で、勾配降下法そのものは、今いる場所の損失と傾きしか使わない。", height=620)
    return DESCENT, DESCENT_ETA


@app.cell(hide_code=True)
def _(BEST_LOSS, DESCENT, DESCENT_ETA, mo):
    _s0, _s1 = DESCENT[0], DESCENT[1]
    _n = next(t for t, s in enumerate(DESCENT) if s["loss"] - BEST_LOSS < 0.005)
    mo.md(rf"""
    最初の1回では、$w=0$ での傾き $L'(0) \approx {_s0["grad"]:.2f}$ から、$w$ は $0 - {DESCENT_ETA} \times ({_s0["grad"]:.2f}) \approx {_s1["w"]:.3f}$ へ動き、
    損失は {_s0["loss"]:.3f} から {_s1["loss"]:.3f} に下がりました。
    次の①で予測し直すと、「データと予測」の図の横ばいだった確率の線が、右上がりの曲線に変わります。

    谷底に近づくほど傾きが緩やかになるので、同じ学習率でも1回に動く幅は小さくなります。
    {_n}回の更新で、損失は谷底の値 {BEST_LOSS:.3f} との差が0.005未満になりました。

    損失は0にはなりません。湿度が高くても降らなかった日、低くても降った日があるので、どんな $w$ でも、全日の正解に確率1を与えることはできないからです。
    図の灰色の曲線は、ふつうは見えません（パラメータが多いと描けません）。それでも、今いる場所の傾きさえ計算できれば谷底へ近づけるのが、勾配降下法のよいところです。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"傾きの式の導き方（任意）": mo.md(r"""
    1日分の損失 $\ell = -y\log p - (1-y)\log(1-p)$ は、$p = \sigma(s)$、$s = wx$ を通して $w$ につながっています。
    このように関数をつないだものの微分は、それぞれの微分を掛け合わせて求めます（連鎖律）。

    $$\frac{d\ell}{dw} = \frac{d\ell}{ds}\cdot\frac{ds}{dw}$$

    sigmoidの微分が $\sigma'(s) = \sigma(s)\bigl(1-\sigma(s)\bigr) = p(1-p)$ となることを使うと、

    $$\frac{d\ell}{ds} = \left(-\frac{y}{p} + \frac{1-y}{1-p}\right) p(1-p) = p - y, \qquad \frac{ds}{dw} = x$$

    となり、$\dfrac{d\ell}{dw} = (p - y)\,x$ です。平均の損失 $L$ の傾きは、これを全日で平均したものになります。
    04の逆伝播は、この連鎖律を何段にも重ねて使う方法です。
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 学習率の選び方

    学習率 $\eta$ は、学習の前に人が決める値（02で学んだハイパーパラメータ）です。
    大きいほど早く谷底に着きそうですが、本当にそうでしょうか。

    下の図は、同じ $w=0$ から12回更新したときの動きを、学習率ごとに比べたものです。
    「損失の曲線の上での w の動き」には更新ごとの $w$ の位置（数字は更新回数）を、「更新回数と損失」には損失の変化を描いています。学習率を切り替えて比べてください。
    """)
    return


@app.cell(hide_code=True)
def _(BEST_LOSS, BEST_W, RAIN, X, cross_entropy, descend, figure, np):
    RATES = [0.002, 0.01, 0.15]
    RATE_RUNS = [{"eta": eta, "states": descend(X, RAIN, eta, 12)} for eta in RATES]
    _ws = np.linspace(-0.03, 0.7, 147)
    figure("rates", {
        "runs": RATE_RUNS, "initial": 0,
        "curve": [[float(w), cross_entropy(w, 0.0, X, RAIN)] for w in _ws],
        "best": [BEST_W, BEST_LOSS], "wlim": [-0.03, 0.7], "llim": [0.4, 1.1],
        "wticks": [0, 0.2, 0.4, 0.6], "lticks": [0.4, 0.6, 0.8, 1.0],
    }, "図9　学習率だけを変えて、同じ $w$ = 0 から12回更新した。「更新回数と損失」の灰色の線は、選んでいないほかの学習率。", height=520)
    return (RATE_RUNS,)


@app.cell(hide_code=True)
def _(BEST_LOSS, DESCENT_ETA, RATE_RUNS, mo):
    _slow, _good, _fast = RATE_RUNS
    _fast_losses = [s["loss"] for s in _fast["states"]]
    mo.md(rf"""
    - **$\eta = {_slow["eta"]}$（小さすぎる）**：向きは正しいのですが、1回に動く幅が小さく、12回更新しても損失は {_slow["states"][-1]["loss"]:.3f} で、谷底（{BEST_LOSS:.3f}）に届きません。
    - **$\eta = {_good["eta"]}$**：数回で谷底の近くに着きます。
    - **$\eta = {_fast["eta"]}$（大きすぎる）**：最初の1回で谷を飛び越え、損失は {_fast_losses[0]:.3f} から {_fast_losses[1]:.3f} に**増えます**。
      その後も谷の両側を行ったり来たりして、損失は {min(_fast_losses[-4:]):.3f} と {max(_fast_losses[-4:]):.3f} の間を往復し、いつまでも落ち着きません。

    傾きから分かるのは、今いる場所のすぐ近くの坂の向きだけです。遠くまで同じ坂が続くとは限らないので、一度に大きく動きすぎると、かえって損失が増えることがあります。
    よい学習率はデータやモデルによって変わるので、損失の変化を見ながら選びます。

    ### コードで確かめる

    勾配降下法をNumPyで書くと、次のようになります。`humidity` と `rain` には、図の30日分のデータが入っています。
    ループの中の行が、確率 → 損失 → 傾き → 更新 に一つずつ対応しています。

    1. `learning_rate` を変えて、「更新5回」の行の損失を **{BEST_LOSS + 0.001:.3f}未満**（谷底の {BEST_LOSS:.3f} とほぼ同じ）にしてください。図8・図9の {DESCENT_ETA} では、5回では届きません。大きすぎても届きません。
    2. 最後の行の `-` を `+` に変えて（傾きと同じ向きに動かして）実行すると、損失はどう変わりますか。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    _code = "\n".join([
        "# humidity: 30日の朝の湿度（%）、rain: 雨が降ったか（1 = 降った、0 = 降らなかった）",
        "x = humidity - 70      # 湿度70%との差",
        "y = rain",
        "w = 0.0                # 重みの初期値",
        "learning_rate = 0.002  # 学習率",
        "",
        "for step in range(6):",
        "    p = 1 / (1 + np.exp(-w * x))                              # 雨の確率",
        "    loss = np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))  # 交差エントロピー",
        "    print(f'更新{step}回: w = {w:.3f}  損失 = {loss:.3f}')",
        "    grad = np.mean((p - y) * x)                               # 損失の傾き",
        "    w = w - learning_rate * grad                              # 勾配降下法の更新",
    ])
    ex_gd = mo.ui.code_editor(value=_code, language="python", min_height=250).form(
        submit_button_label="▶", submit_button_tooltip="実行", bordered=False)
    return (ex_gd,)


@app.cell(hide_code=True)
def _(HUMIDITY, RAIN, ex_gd, mo, np):
    def _run(code):
        import contextlib
        import io
        buffer = io.StringIO()
        scope = {"np": np, "humidity": HUMIDITY.copy(), "rain": RAIN.copy()}
        try:
            with contextlib.redirect_stdout(buffer), np.errstate(all="ignore"):
                exec(code, scope)
        except Exception as exc:
            return mo.callout(mo.md(f"**エラー：** `{type(exc).__name__}: {exc}`　直前に変えた行を見直してください。"), kind="danger")
        output = buffer.getvalue() or "（出力はありません）"
        if "loss" in scope and not np.all(np.isfinite(scope["loss"])):
            output += "\n損失が nan や inf になりました。更新幅が大きく、確率が計算上0や1になった可能性があります。学習率を小さくして比べてください。"
        return mo.plain_text(output)

    # 01のmarimoのセルに似せた欄。▶で実行し、実行するまで出力欄は出さない
    _output = "" if ex_gd.value is None else f'<div class="pycell-output">{_run(ex_gd.value).text}</div>'
    mo.Html(f'<div class="pycell">{ex_gd}{_output}</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. パラメータが多いとき：勾配とミニバッチ

    ここまでは $b=0$ に固定し、$w$ だけを学習しました。
    実際には $w$ と $b$ の両方（特徴量が多ければ $w_1, w_2, \dots$ のすべて）を同時に学習します。

    ほかのパラメータを固定し、一つのパラメータだけを動かしたときの傾きを**偏微分**と呼びます。
    すべてのパラメータの偏微分を並べたベクトルが**勾配（gradient）** $\nabla L$ です。ロジスティック回帰では、それぞれ次の式になります。

    $$\frac{\partial L}{\partial w_j} = \frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)\,x_{ij}$$

    $$\frac{\partial L}{\partial b} = \frac{1}{N}\sum_{i=1}^{N}(p_i-y_i)$$

    $x_{ij}$ は $i$ 番目の日の $j$ 番目の特徴量（湿度や気圧の変化）です。パラメータをまとめて $\boldsymbol\theta$ と書くと、更新は4節と同じ形になります。

    $$\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta\,\nabla L(\boldsymbol\theta)$$

    02の損失の地形で言えば、今いる場所で最も急に下る向きへ一歩進むことにあたります。
    04のニューラルネットワークではパラメータが何千個にもなりますが、勾配を求めて同じ式で更新します。

    ### 一部のデータで勾配を計算する：ミニバッチ

    ここまでは、1回の更新のたびに30日すべての損失から勾配を計算しました。これを**バッチ勾配降下法**と呼びます。
    データが何万件もあると、1回の更新のたびに全件を計算することになり、時間がかかります。
    そこで、データをランダムに並べ替えて少しずつ取り出し、**その一部だけで勾配を計算して更新する**方法がよく使われます。

    | 方法 | 1回の更新に使うデータ | 30日分を一巡する間の更新回数 |
    |---|---|---|
    | バッチ勾配降下法 | 全部（30日分） | 1回 |
    | **ミニバッチ勾配降下法** | 一部（例えば5日分） | 6回 |
    | **確率的勾配降下法（SGD）** | 1日分 | 30回 |

    データ全体を一巡することを**1エポック（epoch）**と呼びます。
    一部のデータで計算した勾配は、全データで計算した勾配と少しずれます。そのため、全データの損失は更新のたびに上下に揺れながら下がっていきます。
    その代わり、同じ計算量で何回も更新できます。

    05のPyTorchでは、128枚ずつのミニバッチで学習します。
    なお、実際のプログラムや文献では、ミニバッチを使う方法もまとめてSGDと呼ぶことがよくあります。
    """)
    return


@app.cell(hide_code=True)
def _(BEST_LOSS, RAIN, X, cross_entropy, figure, mo, np):
    _eta, _epochs = 0.005, 5
    _orders = [np.random.default_rng(3 + e).permutation(len(RAIN)) for e in range(_epochs)]

    def _history(size):
        """size 日分ずつ勾配を計算して更新し、更新のたびに全データの損失を測る。横軸はエポック。"""
        w, done, pts = 0.0, 0, [[0.0, cross_entropy(0.0, 0.0, X, RAIN)]]
        for order in _orders:
            for ids in np.array_split(order, len(RAIN) // size):
                p = 1 / (1 + np.exp(-w * X[ids]))
                w -= _eta * float(np.mean((p - RAIN[ids]) * X[ids]))
                done += len(ids)
                pts.append([done / len(RAIN), cross_entropy(w, 0.0, X, RAIN)])
        return pts

    _panels = []
    for _size, _title in [(30, "バッチ（30日分）"), (5, "ミニバッチ（5日分ずつ）"), (1, "SGD（1日分ずつ）")]:
        _pts = _history(_size)
        _panels.append({"title": _title, "stats": f"更新 {len(_pts) - 1}回", "pts": _pts})
    mo.accordion({"3つの方法で、損失の下がり方を比べる（任意）": mo.vstack([
        mo.md(rf"""
        同じ $w=0$、同じ学習率 $\eta = {_eta}$ で、5エポック学習しました。横軸はエポック、縦軸は**全データの**平均損失で、更新するたびに測っています。
        """),
        figure("batches", {"panels": _panels, "epochs": _epochs, "best": BEST_LOSS, "llim": [0.44, 0.74], "lticks": [0.5, 0.6, 0.7]},
               "図10　バッチ・ミニバッチ・SGDの損失の変化。同じデータ・初期値・学習率で、使うデータの数だけを変えた。", height=380),
        mo.md(rf"""
        バッチは1エポックに1回しか更新しないので、5エポック後も損失は {_panels[0]["pts"][-1][1]:.3f} で、まだ下りきっていません。
        ミニバッチとSGDは、同じ5エポックでも更新回数が多いので、早く谷底（{BEST_LOSS:.3f}）の近くに着きます。
        そのかわり、一部のデータだけで勾配を計算するので、損失は上下に揺れます。1日分ずつのSGDでは、選んだ1日に引っ張られて、損失が一時的に大きく上がることもあります。

        どの方法がよいかは、データの量、1回の計算の重さ、揺れの大きさの兼ね合いで決まります。
        実際には、数十〜数百件ずつのミニバッチがよく使われます。
        """),
    ])})
    return


@app.cell(hide_code=True)
def _(mo, np):
    SOFTMAX_CLASSES = ["晴れ", "くもり", "雨"]
    SOFTMAX_SCORES = [2.0, 1.0, -1.0]
    _exps = np.exp(SOFTMAX_SCORES)
    SOFTMAX_PROBS = (_exps / _exps.sum()).tolist()
    SOFTMAX_EXPS = _exps.tolist()
    mo.md(r"""
    ## 7. 3つ以上のクラス：softmax

    ここまでは「雨が降るか、降らないか」の二つでした。天気予報のように「晴れ・くもり・雨」の三つに分けるにはどうすればよいでしょうか。
    クラスが $K$ 個あるときは、クラスごとに重みとバイアスを用意して、$K$ 個のスコアを計算します。

    $$s_k = \boldsymbol w_k^\top \boldsymbol x + b_k \qquad (k = 1, \dots, K)$$

    各スコアをばらばらにsigmoidに通すと、三つの確率の合計が1になるとは限りません。
    そこで**softmax（ソフトマックス）関数**を使い、合計が1になる確率に変えます。

    $$p_k = \frac{e^{s_k}}{\sum_{j=1}^{K} e^{s_j}}$$

    次の図は、スコアが $[2, 1, -1]$ のときの計算です。
    """)
    return SOFTMAX_CLASSES, SOFTMAX_EXPS, SOFTMAX_PROBS, SOFTMAX_SCORES


@app.cell(hide_code=True)
def _(SOFTMAX_CLASSES, SOFTMAX_EXPS, SOFTMAX_PROBS, SOFTMAX_SCORES, figure):
    figure("softmax", {"classes": SOFTMAX_CLASSES, "scores": SOFTMAX_SCORES, "exps": SOFTMAX_EXPS, "probs": SOFTMAX_PROBS},
           "図11　softmaxの計算。① スコア $s$ を ② 指数関数で $eˢ$ に変えて正の数にし、③ 合計で割って確率にする。", height=260)
    return


@app.cell(hide_code=True)
def _(SOFTMAX_PROBS, mo, np):
    _p = SOFTMAX_PROBS
    mo.md(rf"""
    $e^{{s}}$ は、スコアが負でも正の値になり、スコアの大小の順番はそのまま保たれます。それを合計で割るので、確率はすべて正で、合計は1になります。
    スコアが最も大きい「晴れ」が、確率も最も大きいクラス（約{_p[0]:.0%}）で、予報は「晴れ」になります。

    損失の考え方は二値分類と同じで、**正解のクラスに与えた確率の $-\log$** です。$i$ 番目のデータの正解のクラスを $y_i$ とすると、

    $$\ell_i = -\log p_{{i,\,y_i}}$$

    実際の天気が「くもり」なら損失は $-\log {_p[1]:.3f} \approx {-np.log(_p[1]):.2f}$、「雨」なら $-\log {_p[2]:.3f} \approx {-np.log(_p[2]):.2f}$ です。
    予測した「晴れ」の確率ではなく、**正解のクラスの確率**を使うことに注意してください。
    この損失を全データで平均し、勾配降下法で重みを更新する流れは、二値分類とまったく同じです。

    05では、手書き数字の画像を0〜9の10クラスに分類します。そこでも、このsoftmaxと交差エントロピーを使います。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 確認問題

    選ぶと解説が出ます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_same = mo.ui.radio([
        "予報が同じなので、交差エントロピーも同じになる",
        "予報が同じでも、交差エントロピーは異なることがある",
        "予報が同じなら、自信の強い（確率が0や1に近い）モデルのほうが必ず交差エントロピーが小さい",
    ], label="**Q1.** 二つのモデルが、複数の日について、どの日も同じ予報（雨・降らない）をしました。二つのモデルの交差エントロピーについて、正しいのはどれですか？")
    quiz_same
    return (quiz_same,)


@app.cell(hide_code=True)
def _(check, quiz_same):
    check(quiz_same, "予報が同じでも、交差エントロピーは異なることがある",
          "交差エントロピーは予報したクラスだけでなく、正解に与えた確率から計算します。"
          "予報が同じでも確率が異なれば損失は異なることがあります。ただし、確率が違っても平均損失が同じになる場合はあります。"
          "自信が強いモデルは、外したときの損失も大きくなります。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_sign = mo.ui.radio(["0.2 増える", "0.2 減る", "0.4 減る"],
                            label="**Q2.** ある $w$ で、損失の傾きが $L'(w) = 0.4$ でした。学習率 $\\eta = 0.5$ で1回更新すると、$w$ はどうなりますか？")
    quiz_sign
    return (quiz_sign,)


@app.cell(hide_code=True)
def _(check, quiz_sign):
    check(quiz_sign, "0.2 減る",
          "更新は $w \\leftarrow w - \\eta L'(w) = w - 0.5 \\times 0.4 = w - 0.2$ です。"
          "傾きが正とは「$w$ を増やすと損失が増える」ことなので、逆向き（小さくする向き）に動きます。動く幅は傾きに学習率を掛けた分です。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_rate = mo.ui.radio([
        "学習率を小さくする",
        "学習率を大きくして、早く谷底に着くようにする",
        "更新回数を増やせば、そのうち落ち着くので待つ",
    ], label="**Q3.** 全データで勾配を計算して学習しているのに、損失が上がったり下がったりを繰り返し、いつまでも落ち着きません。まず試すべきことはどれですか？")
    quiz_rate
    return (quiz_rate,)


@app.cell(hide_code=True)
def _(check, quiz_rate):
    check(quiz_rate, "学習率を小さくする",
          "一歩が大きすぎて、谷を飛び越えては戻ることを繰り返している可能性があります。"
          "学習率を大きくすると、飛び越え方がもっとひどくなります。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_softmax = mo.ui.radio(["−log 0.2 ≈ 1.61", "−log 0.7 ≈ 0.36", "−log (1 − 0.2) ≈ 0.22"],
                               label="**Q4.** 写真を「犬・猫・鳥」に分類するモデルが、ある写真に犬・猫・鳥の順で確率 $[0.2, 0.7, 0.1]$ を出しました。正解は「犬」でした。この写真の損失はどれですか？")
    quiz_softmax
    return (quiz_softmax,)


@app.cell(hide_code=True)
def _(check, quiz_softmax):
    check(quiz_softmax, "−log 0.2 ≈ 1.61",
          "損失は**正解のクラス**（犬）に与えた確率の $-\\log$ です。モデルが最も高い確率を出した「猫」の0.7は使いません。"
          "正解に0.2しか与えていないので、損失は大きくなります。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_batch = mo.ui.radio([
        "一部のデータで勾配を計算しているので、揺れるのは自然。エポックを重ねて全体として下がっているかを見る",
        "勾配の計算が間違っているので、コードを見直す",
        "揺れているのは学習に失敗しているからなので、全データのバッチ勾配降下法に切り替える",
    ], label="**Q5.** 5日分ずつのミニバッチで学習していたら、全データの損失が、更新のたびに少し上がったり下がったりしながら、全体としては下がっていきました。どう考えますか？")
    quiz_batch
    return (quiz_batch,)


@app.cell(hide_code=True)
def _(check, quiz_batch):
    check(quiz_batch, "一部のデータで勾配を計算しているので、揺れるのは自然。エポックを重ねて全体として下がっているかを見る",
          "ミニバッチで計算した勾配は、全データで計算した勾配と少しずれるので、全データの損失が揺れることがあります。"
          "全体として下がっているかを確認します。いつまでも落ち着かない場合は学習率も疑います。")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    - **分類のモデルは確率を出す。** スコア $s = wx + b$ をsigmoid（3クラス以上ならsoftmax）で確率に変え、確率が最も大きいクラスを予測とする。特徴量が二つなら、予測の境目（決定境界）は直線になる。
    - **損失は交差エントロピー。** 正解に与えた確率 $q$ の $-\log$ を平均する。自信を持って外すほど大きい。
    - **勾配降下法で学習する。** 傾き（勾配）と逆の向きに、学習率 $\eta$ を掛けた分だけパラメータを動かす。$\eta$ が小さすぎると遅く、大きすぎると損失が増えたり落ち着かなかったりする。
    - **データが多いときはミニバッチで勾配を計算する。** 損失は揺れながら下がる。

    **この章の確認：** 勾配と逆向きに更新する理由と、学習率が大きすぎる場合の結果を説明できれば完了です。

    ここで小さくしたのは、学習に使ったデータ（訓練データ）の損失です。まだ見ていないデータでもよい予測ができるかは、02と同じく検証データで確かめます。

    次の04では、直線では分けられないデータをニューラルネットワークで分類し、逆伝播で勾配を求めます。

    ### もっと詳しく読むとき（任意）

    岡崎直観ほか『機械学習帳』の
    [二値分類](https://chokkan.github.io/mlnote/classification/01binary.html)、
    [多クラス分類](https://chokkan.github.io/mlnote/classification/02multi.html)、
    [確率的勾配降下法](https://chokkan.github.io/mlnote/regression/04sgd.html)
    が、この章に対応します。
    """)
    return


if __name__ == "__main__":
    app.run()
