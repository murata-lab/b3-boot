import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="04 ニューラルネットワークと逆伝播", css_file="notebook.css")


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np
    from contourpy import contour_generator

    return Path, contour_generator, json, mo, np


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

    def fmt(value, digits=2):
        """本文用の数値。−0 を出さず、マイナスは全角の − にする。"""
        value = float(value)
        if abs(value) < 0.5 * 10 ** -digits:
            value = 0.0
        return f"{value:.{digits}f}".replace("-", "−")

    def g(value):
        """重みなど、0.5刻みの値をそのまま書く（例：−0.5、1、2）。"""
        value = float(value)
        return f"{value + 0.0:g}".replace("-", "−")

    return check, figure, fmt, g


@app.cell(hide_code=True)
def _(np):
    # 部屋の例（説明用に作ったデータ）。気温20〜29℃かつ湿度35〜65%の部屋にいる人が「快適」と答えたことにする。
    # 快適な範囲の境界のすぐ近くの部屋と、図で近すぎる部屋は作り直す。4節から使う26℃・40%の部屋は必ず入れる。
    COMFORT = (20.0, 29.0, 35.0, 65.0)
    ROOM = (26.0, 40.0)

    def _gap(t, u):
        """快適な範囲の境界までの距離。気温1℃と湿度5%を同じ長さと数える。"""
        t0, t1, u0, u1 = COMFORT
        dt, du = max(t0 - t, 0.0, t - t1), max(u0 - u, 0.0, u - u1) / 5
        if dt == du == 0:
            return min(t - t0, t1 - t, (u - u0) / 5, (u1 - u) / 5)
        return float(np.hypot(dt, du))

    _rng = np.random.default_rng(3)
    _rooms = [(*ROOM, 1)]
    _need = {1: 26, 0: 30}
    while any(_need.values()):
        _t = round(float(_rng.uniform(14.5, 33.5)), 1)
        _u = float(round(_rng.uniform(22, 83)))
        _y = int(COMFORT[0] <= _t <= COMFORT[1] and COMFORT[2] <= _u <= COMFORT[3])
        if _need[_y] == 0 or _gap(_t, _u) < 0.9:
            continue
        if any(((_t - t) / 20) ** 2 + ((_u - u) / 65) ** 2 < 0.045 ** 2 for t, u, _ in _rooms):
            continue
        _rooms.append((_t, _u, _y))
        _need[_y] -= 1
    TEMP, HUMID, LABEL = (np.array(v, dtype=float) for v in zip(*_rooms))

    def to_input(t, u):
        """モデルへの入力：23℃・50%からのずれを、気温は3℃、湿度は10%を1として数える。"""
        return np.column_stack([(np.asarray(t, dtype=float) - 23) / 3, (np.asarray(u, dtype=float) - 50) / 10])

    X = to_input(TEMP, HUMID)
    ROOMS = np.column_stack([TEMP, HUMID, LABEL]).tolist()
    return LABEL, ROOM, ROOMS, X, to_input


@app.cell(hide_code=True)
def _(np):
    def sigmoid(s):
        return 1 / (1 + np.exp(-s))

    def predict(model, x):
        """快適な確率 p。model は線形モデル {w, b} か、隠れ層が1層のネットワーク {W, b, v, c}。"""
        x = np.atleast_2d(x)
        if "v" not in model:
            return sigmoid(x @ model["w"] + model["b"])
        return sigmoid(np.maximum(0, x @ model["W"].T + model["b"]) @ model["v"] + model["c"])

    def cross_entropy(p, y):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))

    def accuracy(p, y):
        return float(np.mean((p >= 0.5) == (y == 1)))

    def random_network(seed, n=4):
        rng = np.random.default_rng(seed)
        return {"W": rng.normal(0, 1, (n, 2)), "b": rng.normal(0, 0.5, n), "v": rng.normal(0, 1, n), "c": 0.0}

    # 3節のデモと4〜6節の手計算で使う、学習前のネットワーク。
    # ランダムな重みを手計算しやすいよう0.5刻みに丸めた。乱数の種は、4節の部屋で
    # 動いているニューロンと0を出すニューロンの両方があり、全部屋を分けられるまで学習が進むものを選んだ。
    INIT = {k: (np.round(v * 2) / 2 if k != "c" else 0.0) for k, v in random_network(88).items()}
    return INIT, accuracy, cross_entropy, predict, random_network, sigmoid


@app.cell(hide_code=True)
def _(contour_generator, np, predict, to_input):
    # 気温・湿度の平面を細かい格子に分け、確率やニューロンの値の等高線を求める。
    _t = np.linspace(14.0, 34.0, 121)
    _u = np.linspace(20.0, 85.0, 131)
    _grid = to_input(*(g.ravel() for g in np.meshgrid(_t, _u)))

    def _field(values):
        return values.reshape(len(_u), len(_t))

    def field_bands(values, levels):
        gen = contour_generator(_t, _u, _field(values), fill_type="OuterOffset")
        bands = []
        for lo, hi in zip(levels[:-1], levels[1:]):
            rings = []
            for pts, offsets in zip(*gen.filled(lo, hi)):
                for s, e in zip(offsets[:-1], offsets[1:]):
                    rings.append(np.round(pts[s:e], 2).ravel().tolist())
            bands.append(rings)
        return bands

    def field_lines(values, level):
        gen = contour_generator(_t, _u, _field(values))
        return [np.round(line, 2).ravel().tolist() for line in gen.lines(level)]

    def prob_map(model):
        """快適と予測する確率の色分け（0.2・0.5・0.8で区切る）と、決定境界 p = 0.5。"""
        p = predict(model, _grid)
        return {"bands": field_bands(p, [-0.01, 0.2, 0.5, 0.8, 1.01]), "boundary": field_lines(p, 0.5)}

    def neuron_map(w, b):
        """重み w・バイアス b のニューロンの出力 h の色分けと、ReLUが折れる直線 z = 0。"""
        z = _grid @ np.asarray(w) + b
        return {"bands": field_bands(np.maximum(0, z), [-1, 1e-9, 0.5, 1, 1.5, 2, 3, 1e9]), "fold": field_lines(z, 0.0)}

    return neuron_map, prob_map


@app.cell(hide_code=True)
def _(np, predict):
    def gradients(model, x, y):
        """全データの平均損失の勾配。"""
        n = len(y)
        ds = (predict(model, x) - y) / n
        if "v" not in model:
            return {"w": x.T @ ds, "b": ds.sum()}
        z = x @ model["W"].T + model["b"]
        h = np.maximum(0, z)
        dz = np.outer(ds, model["v"]) * (z > 0)
        return {"W": dz.T @ x, "b": dz.sum(0), "v": h.T @ ds, "c": ds.sum()}

    def descend(model, x, y, record, lr=0.5):
        """勾配降下法。record に含まれる更新回数のときの重みを返す。"""
        model = {k: np.array(v, dtype=float) for k, v in model.items()}
        snapshots = {}
        for step in range(max(record) + 1):
            if step in record:
                snapshots[step] = {k: v.copy() for k, v in model.items()}
            grad = gradients(model, x, y)
            model = {k: model[k] - lr * grad[k] for k in model}
        return snapshots

    return (descend,)


@app.cell(hide_code=True)
def _(INIT, LABEL, X, accuracy, cross_entropy, descend, predict, random_network):
    # 3節のデモ：学習前のネットワーク INIT から、全部屋の平均損失で学習する。比較のため線形モデルも同じ方法で学習する。
    STEPS = [0, 10, 20, 50, 100, 200, 500, 1000]
    _nn = descend(INIT, X, LABEL, STEPS)
    _linear = descend({"w": [0.0, 0.0], "b": 0.0}, X, LABEL, STEPS)

    def _stats(model):
        p = predict(model, X)
        return {"loss": cross_entropy(p, LABEL), "acc": accuracy(p, LABEL)}

    TRAINED = [_nn[s] for s in STEPS]
    NN_STATS = [_stats(m) for m in TRAINED]
    LINEAR = _linear[STEPS[-1]]
    LINEAR_STATS = [_stats(_linear[s]) for s in STEPS]

    # 3節の図：隠れ層のニューロンの数だけを変え、ランダムな重みから2000回学習する。
    COUNTS = [(1, 3), (2, 4), (4, 1), (8, 1)]  # （ニューロンの数, 乱数の種）
    COUNT_MODELS = [descend(random_network(seed, n), X, LABEL, [2000])[2000] for n, seed in COUNTS]
    return COUNTS, COUNT_MODELS, LINEAR, LINEAR_STATS, NN_STATS, STEPS, TRAINED


@app.cell(hide_code=True)
def _(INIT, LABEL, ROOM, X, accuracy, np, predict, to_input):
    # 4〜6節：26℃・40%の部屋（正解は快適）について、学習前のネットワークで順伝播・逆伝播・更新を一回ずつ行う。
    ETA = 0.5
    Y_ROOM = 1.0

    def trace(model, x, y):
        z = model["W"] @ x + model["b"]
        h = np.maximum(0, z)
        s = float(model["v"] @ h + model["c"])
        p = float(1 / (1 + np.exp(-s)))
        ds = p - y
        dh = ds * model["v"]
        dz = np.where(z > 0, dh, 0.0)
        grads = {"W": np.outer(dz, x), "b": dz, "v": ds * h, "c": ds}
        return {"x": x, "z": z, "h": h, "s": s, "p": p, "loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p))),
                "ds": ds, "dh": dh, "dz": dz, "grads": grads}

    X_ROOM = to_input(*ROOM)[0]
    BEFORE = trace(INIT, X_ROOM, Y_ROOM)
    UPDATED = {k: INIT[k] - ETA * BEFORE["grads"][k] for k in INIT}
    AFTER = trace(UPDATED, X_ROOM, Y_ROOM)
    INIT_ACC = accuracy(predict(INIT, X), LABEL)
    return AFTER, BEFORE, ETA, INIT_ACC, UPDATED, X_ROOM, Y_ROOM, trace


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <header class="hero">
      <p class="eyebrow">機械学習入門</p>
      <h1>ニューラルネットワークと逆伝播</h1>
      <p>直線では分けられないデータを分けるモデルを作り、その重みを学習する計算をたどります。</p>
    </header>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    03の線形モデルは、直線でクラスを分けました。点の並び方によっては、どこに直線を引いても分けられません。
    境界を曲げる**ニューラルネットワーク（neural network）**のしくみと、その重みを学習する計算を見ていきます。

    **この章の目標：** 隠れ層のReLUの役割と、順伝播・逆伝播・更新でそれぞれ何が変わるかを説明することです。
    """)
    return


@app.cell(hide_code=True)
def _(ROOMS, mo):
    mo.md(rf"""
    ## 1. 直線では分けられないデータ

    部屋の気温と湿度から、その部屋にいる人が「快適」と感じるかを予測したいとします。
    下の図は、{len(ROOMS)}部屋で気温と湿度を測り、快適かどうかを答えてもらった結果です（説明用に作ったデータです）。
    横軸が気温、縦軸が湿度で、点の一つが1部屋です。
    """)
    return


@app.cell(hide_code=True)
def _(ROOMS, figure):
    figure("rooms", {"rooms": ROOMS}, f"図1　{len(ROOMS)}部屋の気温・湿度と、快適と答えたかどうか。", height=420)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    快適な部屋（●）は真ん中に集まり、その周りを、暑すぎる・寒すぎる・湿りすぎる・乾きすぎる部屋（○）が囲んでいます。

    まず、03の線形モデル（ロジスティック回帰）で分けてみます。
    気温と湿度から作った入力 $x_1, x_2$ で、スコア $s = w_1x_1 + w_2x_2 + b$ を計算し、sigmoid で確率 $p$ に変えて、$p \ge 0.5$ なら快適と予測します。
    03で見たとおり、$p = 0.5$ となる**決定境界**は直線です。
    03と同じ勾配降下法で、損失がほとんど下がらなくなるまで学習した結果が次の図です。
    背景の色は、その場所の部屋を快適と予測する確率で、青いほど快適、橙ほど不快と予測しています。
    """)
    return


@app.cell(hide_code=True)
def _(LABEL, LINEAR, ROOMS, X, accuracy, figure, predict, prob_map):
    figure("probability", {"rooms": ROOMS, **prob_map(LINEAR), "acc": accuracy(predict(LINEAR, X), LABEL)},
           "図2　線形モデルで学習した結果。黒い線が決定境界（p = 0.5）。", height=440)
    return


@app.cell(hide_code=True)
def _(LABEL, LINEAR, X, accuracy, mo, predict):
    mo.md(rf"""
    正解率は{accuracy(predict(LINEAR, X), LABEL):.0%}で、半分ほどしか当たりません。背景の色もほとんど変わらず、どの部屋にも確率0.5前後を出しています。
    快適な部屋を不快な部屋が取り囲んでいるので、どこに直線を引いても、その両側に快適な部屋と不快な部屋が残ってしまうのです。

    快適な部屋を囲むには、境界を途中で折り曲げる必要があります。
    ニューラルネットワークは、そのような境界を作れるモデルです。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. ニューラルネットワークの計算

    ニューラルネットワークは、よく下の図の上半分のような形で描かれます。
    丸の一つひとつを**ニューロン（neuron）**、縦に並んだ丸のまとまりを**層（layer）**と呼びます。
    左端が入力を受け取る**入力層**、右端が予測を出す**出力層**です。その間の層は、外から値が直接見えないので**隠れ層（hidden layer）**と呼びます。
    線は、左の層の値が右の層のニューロンへ渡されることを表し、線1本ごとに重みが1つあります。
    """)
    return


@app.cell(hide_code=True)
def _(figure):
    figure("anatomy", {"layers": [2, 3, 3, 1], "focus": [1, 1]},
           "図3　上：入力層2個、隠れ層3個・3個、出力層1個のネットワーク。下：青いニューロン1つの中で行う計算。", height=470)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    どのニューロンも、図の下半分のように二段階の計算をします。

    $$z = w_1x_1 + w_2x_2 + b$$

    $$h = f(z)$$

    まず、前の層から受け取った値 $x_1, x_2$ の重み付き和 $z$ を計算します。ここは、03の線形モデルのスコアとまったく同じ計算です。
    次に、$z$ を**活性化関数（activation function）** $f$ に通し、その結果 $h$ を次の層へ渡します。
    03の線形分類器も、最後にsigmoidを使いました。このネットワークでは、**中間の重み付き和をReLUで変換し、その結果を次の層で組み合わせる**ことで、直線ではない境界を作れます。

    活性化関数にはいくつか種類があります。この章では、いま最もよく使われている **ReLU** を使います。

    $$\mathrm{ReLU}(z) = \max(0,\ z)$$

    下の図は、横軸が重み付き和 $z$、縦軸がReLUの出力 $h$ です。
    """)
    return


@app.cell(hide_code=True)
def _(figure):
    figure("relu", {}, "図4　ReLU。z が負なら0、正ならそのまま返す。z = 0 で折れ曲がる。", height=330)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ReLUは、$z$ が負なら0を返し、正ならそのまま返します。グラフは $z = 0$ で折れ曲がります。
    """)
    return


@app.cell(hide_code=True)
def _(figure, mo):
    mo.accordion({"ほかの活性化関数（任意）": mo.vstack([mo.md(r"""
    ReLUのほかに、次のような関数も活性化関数として使われます。

    | 名前 | 式 | 出力の範囲 |
    |---|---|---|
    | ReLU | $\max(0,\ z)$ | $0$ 以上 |
    | sigmoid | $\sigma(z) = \dfrac{1}{1 + e^{-z}}$ | $0$ 〜 $1$ |
    | tanh | $\tanh(z) = \dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$ | $-1$ 〜 $1$ |

    sigmoid は、03で確率を作るのに使った関数です。tanh は sigmoid に似た形で、0を中心に −1 から 1 の値をとります。
    どれも直線ではない（曲がっている）ことが大事で、その理由を2節の最後で説明します。
    sigmoid や tanh は $z$ の絶対値が大きいところでグラフが平らになり、傾きがほぼ0になります。
    ReLUは正の側で傾きが常に1なので、層が深くても学習が進みやすく、計算も簡単です。そのため現在よく使われ、05のPyTorchのモデルもReLUを使います。
    """), figure("activations", {}, "3つの活性化関数のグラフ。横軸が重み付き和 z、縦軸が出力。", height=300)])})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### この章のネットワーク

    この章では、入力層2個、隠れ層1層にニューロン4個、出力層1個の小さなネットワークを使います。
    入力は、気温と湿度を「23℃・50%からのずれ」に直した値です。

    $$x_1 = (\text{気温} - 23) \div 3$$

    $$x_2 = (\text{湿度} - 50) \div 10$$

    3や10で割るのは、二つの入力をどちらも0の前後の、同じくらいの大きさの数にそろえるためです。02の標準化とは違い、ここでの23・50と3・10はデータから求めた平均・標準偏差ではなく、説明しやすいように決めた基準と尺度です。

    下の図は、このネットワークの各ニューロンで行う計算を書き込んだものです。
    隠れ層の4つのニューロンは、それぞれ自分の重みとバイアスで重み付き和を計算し、ReLUに通します。
    出力層のニューロンは、4つの出力 $h_1, \dots, h_4$ の重み付き和 $s$ を計算し、活性化関数として03と同じ sigmoid を使って、快適な確率 $p$ にします。
    """)
    return


@app.cell(hide_code=True)
def _(figure, network_data):
    figure("network", network_data("structure"),
           "図5　この章のネットワーク。矢印は値が渡される向き。", height=420)
    return


@app.cell(hide_code=True)
def _(INIT, mo):
    _n = INIT["W"].size + INIT["b"].size + INIT["v"].size + 1
    mo.md(rf"""
    学習で決めるパラメータは、隠れ層の重み{INIT['W'].size}個とバイアス{INIT['b'].size}個、出力の重み{INIT['v'].size}個とバイアス $c$ の、合わせて{_n}個です。

    ### 1つのニューロンを平面で見る

    隠れ層のニューロン1つの出力 $h$ を、気温・湿度の平面に塗ってみます。
    $z = 0$ となる場所は平面上の直線で、その片側では $h = 0$、反対側では直線から離れるほど $h$ が大きくなります。
    """)
    return


@app.cell(hide_code=True)
def _(INIT, affine_text, figure, neuron_map):
    _examples = [INIT["W"][0], INIT["b"][0]], [INIT["W"][3], INIT["b"][3]]
    figure("neurons", {"panels": [{"title": f"例{i + 1}", "formula": "$z$ = " + affine_text(w, b), **neuron_map(w, b)}
                                  for i, (w, b) in enumerate(_examples)]},
           "図6　重みとバイアスが違う2つのニューロンの出力 h。白い場所では h = 0。破線は z = 0 の直線で、ここでReLUが折れ曲がる。",
           height=520)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    重みとバイアスを変えると、折れ曲がる直線の位置と向きが変わります。

    ### なぜ活性化関数が必要か

    活性化関数を外して $h_j = z_j$ とすると、出力層の重み付き和は

    $$s = \sum_{j} v_j\,(w_{j1}x_1 + w_{j2}x_2 + b_j) + c$$

    $$= \Big(\sum_j v_jw_{j1}\Big)x_1 + \Big(\sum_j v_jw_{j2}\Big)x_2$$

    $$+ \Big(\sum_j v_jb_j + c\Big)$$

    となり、**一つの重み付き和にまとまってしまいます。** これは1節の線形モデルと同じ形なので、境界は直線にしかなりません。
    ニューロンや層をいくら増やしても同じです。

    ReLUがあると、各ニューロンが図6のようにそれぞれの直線で折れ曲がるので、それらを足し合わせた $s$ も、ところどころで折れ曲がった形になります。
    では、このネットワークで、快適な部屋を囲めるのでしょうか。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 学習すると、境界が折れ曲がる

    ネットワークの17個のパラメータは、人が決めません。**ランダムな値**から始め、03と同じ勾配降下法で、全部屋の平均損失が下がるように少しずつ更新します。
    下の図は、図5のネットワークを更新していったときの変化です。右のグラフには、比較のため、1節の線形モデルを同じ方法で学習したときの損失も描いています。
    """)
    return


@app.cell(hide_code=True)
def _(LINEAR_STATS, NN_STATS, ROOM, ROOMS, STEPS, TRAINED, figure, prob_map):
    def _status(i):
        nn, lin = NN_STATS[i], LINEAR_STATS[i]
        if i == 0:
            return "ランダムな重みから始めた状態です。上の「更新回数」を右へ進めてください。"
        if nn["acc"] < 1:
            return f"{STEPS[i]}回更新しました。損失が下がるにつれて、快適と予測する範囲（青）が快適な部屋（●）に合っていきます。"
        return (f"{STEPS[i]}回更新しました。ネットワークは全部屋を正しく分けています（正解率100%）。"
                f"線形モデルの正解率は{lin['acc']:.0%}のままです。")

    figure("training", {"rooms": ROOMS, "room": list(ROOM), "steps": STEPS, "maps": [prob_map(m) for m in TRAINED],
                        "nn": NN_STATS, "linear": LINEAR_STATS, "status": [_status(i) for i in range(len(STEPS))]},
           "図7　左：ネットワークが快適と予測する範囲。右：訓練データの平均損失（交差エントロピー）。",
           height=640)
    return


@app.cell(hide_code=True)
def _(LINEAR_STATS, NN_STATS, STEPS, mo, np):
    _first = next(s for s, st in zip(STEPS, NN_STATS) if st["acc"] == 1)
    mo.md(rf"""
    ネットワークは{_first}回ほどの更新で全部屋を正しく分けられるようになり、その後も損失が下がり続けます（{STEPS[-1]}回で {NN_STATS[-1]['loss']:.3f}）。
    線形モデルの損失は、{STEPS[-1]}回更新しても {LINEAR_STATS[-1]['loss']:.2f} までしか下がりません。これは、どの部屋にも確率0.5と答えたときの損失 $-\log 0.5 \approx {-np.log(0.5):.2f}$ とほぼ同じです。

    学習後の境界は、直線をいくつかつないだ折れ線です。各ニューロンのReLUが折れる直線が組み合わさって、この形になっています。
    **各ニューロンにどんな役割を持たせるかは、人が決めていません。** どこで折れるかは、損失が下がるように重みを更新した結果として決まります。

    ### ニューロンの数を変えると

    隠れ層のニューロンが増えると、もっと複雑な境界を表せるようになります。実際に学習した境界はどうなるでしょうか。ニューロン数ごとに異なるランダムな初期値から学習した一例を見ます。
    """)
    return


@app.cell(hide_code=True)
def _(COUNTS, COUNT_MODELS, ROOMS, figure, prob_map):
    figure("counts", {"rooms": ROOMS, "panels": [{"title": f"ニューロン{n}個", **prob_map(m)} for (n, _), m in zip(COUNTS, COUNT_MODELS)]},
           "図8　ニューロン数ごとに異なるランダムな初期値から2000回学習した結果の一例。黒い線が決定境界。", height=620)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ニューロンが1個のときの境界は直線で、快適な部屋を囲めません。
    ニューロンを増やすと、境界に折れ目を作り、快適な部屋を囲む形も表せます。ただし、学習後にその力を使えるかは初期値や学習の進み方にもよります。図8の一回ずつの結果から、数を増やせば必ずよくなるとは言えません。

    ただし、これは学習に使った部屋での結果です。まだ見ていない部屋にも通用するかは、02と同じく検証データで確かめる必要があります。

    では、ランダムな重みから始めて、どうやって損失を下げる向きが分かったのでしょうか。
    ここからは、図7の0回目のネットワークを使い、1部屋について学習の1歩を追います。
    """)
    return


@app.cell(hide_code=True)
def _(ROOM, X_ROOM, g, mo):
    mo.md(rf"""
    ## 4. 入力から予測まで：順伝播

    図7の★の部屋（{ROOM[0]:g}℃・{ROOM[1]:g}%）を例にします。この部屋の人は「快適」と答えていますが、0回目のネットワークは不快と予測しています。
    重みは、ランダムな値を手計算しやすいように0.5刻みに丸めたものです。
    入力は $x_1 = ({ROOM[0]:g} - 23) \div 3 = {g(X_ROOM[0])}$、$x_2 = ({ROOM[1]:g} - 50) \div 10 = {g(X_ROOM[1])}$ です。
    この値をネットワークに入れて、中で数がどう計算されていくかを、一段ずつ追ってみます。図の青い動く点は、この部屋について計算した値が次の層へ渡る様子を表します。
    """)
    return


@app.cell(hide_code=True)
def _(ROOM, figure, forward_data):
    figure("forward", forward_data("steps"),
           f"図9　{ROOM[0]:g}℃・{ROOM[1]:g}%の部屋で、学習前のネットワークの値を入力側から順に計算する。"
           "小さなグラフは活性化関数で、点の横の位置が入力（$z$ や $s$）、縦の位置が出力（$h$ や $p$）。", height=620)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    同じ層のニューロンは、同じ入力を受け取り、互いに関係なく計算します。そのため、**層の中は同時に計算し、層から層へは順に進みます。**
    5節のコードでは、4つのニューロンの計算を行列の積 `W @ x` 1回でまとめて書いています。`W` の4行それぞれに2個の重みがあり、2個の入力 `x` と掛け合わせると、4個の重み付き和になります。

    このように、入力側から順に各層の値を計算して予測を出すことを**順伝播（forward propagation）**と呼びます。
    学習が終わったモデルで新しい部屋を予測する（**推論**する）ときは、順伝播だけを行います。正解は使いません。
    """)
    return


@app.cell(hide_code=True)
def _(BEFORE, INIT, fmt, g, mo):
    _n = INIT["W"].size + INIT["b"].size + INIT["v"].size + 1
    _on = [j for j in range(4) if BEFORE["h"][j] > 0]
    mo.md(rf"""
    ## 5. 重みの勾配を求める：連鎖律と逆伝播

    この部屋の正解は「快適」（$y = 1$）です。03と同じ交差エントロピーで損失を測ると、

    $$\ell = -\log p = -\log {BEFORE['p']:.2f} \approx {BEFORE['loss']:.2f}$$

    です。予測を正解に近づけるには、{_n}個のパラメータを、損失が下がる向きに少しずつ動かします。
    そのために、それぞれのパラメータ $\theta$ について、**勾配** $\partial\ell/\partial\theta$（$\theta$ を少し変えたとき、損失がどれだけの割合で変わるか）を求めます。

    ### 出力のパラメータ：03と同じ

    出力は、隠れ層の出力 $h_1,\dots,h_4$ を入力とする、03の線形モデルそのものです。したがって03と同じく、

    $$\frac{{\partial\ell}}{{\partial s}} = p - y = {fmt(BEFORE['ds'])}$$

    $$\frac{{\partial\ell}}{{\partial v_j}} = (p - y)\,h_j,\qquad \frac{{\partial\ell}}{{\partial c}} = p - y$$

    です。例えば $h_{_on[0] + 1} = {g(BEFORE['h'][_on[0]])}$ なので $\partial\ell/\partial v_{_on[0] + 1} = {fmt(BEFORE['grads']['v'][_on[0]])}$ です。$h_j = 0$ のニューロンの $v_j$ の勾配は0です。

    ### 隠れ層のパラメータ：変化をたどって掛ける

    隠れ層のパラメータは、出力までの間にいくつもの計算を通ります。
    例えばニューロン1のバイアス $b_1$ を0.01だけ増やすと何が起きるか、変化を一段ずつたどってみます。
    """)
    return


@app.cell(hide_code=True)
def _(BEFORE, INIT, fmt, figure, g):
    _factors = [1.0, 1.0, float(INIT["v"][0]), BEFORE["ds"]]
    _changes = [0.01]
    for _f in _factors:
        _changes.append(_changes[-1] * _f)
    figure("chain", {
        "nodes": ["$b$₁", "$z$₁", "$h$₁", "$s$", "損失 ℓ"],
        "changes": [("+" if v > 0 else "") + fmt(v, 4 if abs(v) < 0.01 else 2) for v in _changes],
        "factors": [g(f) if i < 3 else fmt(f) for i, f in enumerate(_factors)],
        "notes": ["$z$₁ = $w$₁₁$x$₁ + $w$₁₂$x$₂ + $b$₁", f"ReLUの傾き（$z$₁ = {g(BEFORE['z'][0])} > 0）", "$s$ の式の $v$₁", "$p$ − $y$"],
    }, "図10　b₁ の小さな変化が、損失まで伝わる様子。矢印の上の数は、その一段で変化が何倍になるか（その一段の微分）。", height=260)
    return


@app.cell(hide_code=True)
def _(BEFORE, INIT, X_ROOM, fmt, g, mo):
    _db1 = BEFORE["grads"]["b"][0]
    _v1 = INIT["v"][0]
    _off = [j for j in range(4) if BEFORE["z"][j] < 0]
    _big = max(_off, key=lambda j: abs(INIT["v"][j]))
    mo.md(rf"""
    $b_1$ が0.01増えると、$z_1$ も0.01増えます。$z_1$ は正なので、ReLUを通った $h_1$ も0.01増えます。
    $s$ はその $v_1 = {g(_v1)}$ 倍だけ変わり、損失はさらに $\partial\ell/\partial s = {fmt(BEFORE['ds'])}$ 倍だけ変わって、約{fmt(_db1 * 0.01, 4)}増えます。
    全体の倍率は、途中の倍率の積です。

    $$\frac{{\partial\ell}}{{\partial b_1}}
    = \underbrace{{\frac{{\partial\ell}}{{\partial s}}}}_{{{fmt(BEFORE['ds'])}}}
    \times\underbrace{{\frac{{\partial s}}{{\partial h_1}}}}_{{{g(_v1)}}}
    \times\underbrace{{\frac{{\partial h_1}}{{\partial z_1}}}}_{{1}}
    \times\underbrace{{\frac{{\partial z_1}}{{\partial b_1}}}}_{{1}}
    = {fmt(_db1)}$$

    このように、途中の微分を掛けて全体の微分を求める規則を**連鎖律（chain rule）**と呼びます。
    勾配が正なので、$b_1$ を減らすと損失が下がります。

    重み $w_{{11}}$ も同じ経路を通ります。違うのは最後の一段だけで、$\partial z_1/\partial w_{{11}} = x_1 = {g(X_ROOM[0])}$ です。
    同じように $\partial z_1/\partial w_{{12}} = x_2 = {g(X_ROOM[1])}$ です。

    $$\frac{{\partial\ell}}{{\partial w_{{11}}}} = {fmt(_db1)} \times {g(X_ROOM[0])} = {fmt(BEFORE['grads']['W'][0, 0])}$$

    $$\frac{{\partial\ell}}{{\partial w_{{12}}}} = {fmt(_db1)} \times ({g(X_ROOM[1])}) = {fmt(BEFORE['grads']['W'][0, 1])}$$

    ### ReLUが0を出しているニューロン

    ニューロン{_big + 1}は、出力の重みが $v_{_big + 1} = {g(INIT['v'][_big])}$ と大きいのに、$z_{_big + 1} = {g(BEFORE['z'][_big])}$ で、ReLUの出力は0でした。
    $b_{_big + 1}$ を少し変えても $z_{_big + 1}$ は負のままで、$h_{_big + 1}$ は0から変わりません。ReLUの傾きが0の場所にいるので、連鎖律の途中に「×0」が入ります。
    そのため、**この部屋については、$h = 0$ のニューロンの重みとバイアスの勾配はすべて0です。** この部屋の予測に関わっていないからです。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 逆伝播：出力側から順に、途中の結果を使い回す

    $\partial\ell/\partial b_1$ と $\partial\ell/\partial w_{11}$ は、途中までの積 $\partial\ell/\partial z_1$ が共通でした。
    そこで、出力側から順に「損失をその値で微分したもの」を一度ずつ計算し、一つ手前へ渡していきます。

    1. 出力：$\partial\ell/\partial s = p - y$
    2. 隠れ層の出力：$\partial\ell/\partial h_j = (\partial\ell/\partial s)\,v_j$
    3. 隠れ層の重み付き和：$\partial\ell/\partial z_j = \partial\ell/\partial h_j$（$z_j > 0$ のとき）、$0$（$z_j < 0$ のとき）
    4. 重みとバイアス：$\partial\ell/\partial w_{ji} = (\partial\ell/\partial z_j)\,x_i$、$\partial\ell/\partial b_j = \partial\ell/\partial z_j$

    この手続きを**逆伝播（backpropagation）**と呼びます。
    下の図で、図9の順伝播とは逆に、右の損失から左へ一段ずつ計算を進めてみます。橙の動く点は、損失を途中の値で微分して得た勾配を表します。
    """)
    return


@app.cell(hide_code=True)
def _(backward_data, figure):
    figure("backward", backward_data(),
           "図11　逆伝播。損失から入力側へ順に戻りながら、橙の値（損失をその値で微分したもの）を求める。"
           "小さなグラフはReLUで、点の位置の傾きを掛ける。", height=680)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    どの値も一度ずつ計算するだけなので、パラメータが何百万個あっても、順伝播と同じくらいの手間ですべての勾配が求まります。
    **逆伝播は勾配を計算するだけで、重みはまだ変えていません。** 重みを変えるのは、次の6節の更新です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"隠れ層が2層以上あるとき（任意）": mo.md(r"""
    隠れ層を2層にすると、1層目のニューロンの出力 $h^{(1)}_k$ は、2層目のすべてのニューロンの重み付き和に使われます。
    $h^{(1)}_k$ を少し変えると、その変化は2層目の各ニューロンを通る**複数の経路**で損失に届くので、各経路の寄与を**足し合わせます**。

    $$\frac{\partial\ell}{\partial h^{(1)}_k} = \sum_j \frac{\partial\ell}{\partial z^{(2)}_j}\,w^{(2)}_{jk}$$

    逆伝播では、2層目の $\partial\ell/\partial z^{(2)}_j$ を先に求めておき、この和で $\partial\ell/\partial h^{(1)}_k$ を計算して、さらに手前へ渡します。
    **一本の経路の中では微分を掛け、経路が分かれていたら足す。** 層がいくつあっても、連鎖律の使い方はこれだけです。
    """)})
    return


@app.cell(hide_code=True)
def _(BEFORE, INIT, ROOM, fmt, g, mo, to_input):
    # 課題2の部屋：暑すぎて不快な部屋。4節の部屋とは、ReLUが0を出すニューロンが変わる
    _other = (32, 50)
    _x = [float(v) + 0.0 for v in to_input(*_other)[0]]
    mo.md(rf"""
    ### コードで確かめる

    順伝播と逆伝播を、NumPyで書くと次のようになります。`@` は行列とベクトルの積、`np.where(条件, A, B)` は条件ごとにAかBを選ぶ計算、`np.outer(dz, x)` は各ニューロンの勾配と各入力を掛け合わせて表にする計算です。

    1. まずそのまま実行し、損失と $\partial\ell/\partial b$ が図11と同じになることを確かめてください。
       次に、`b` の最初の値 `{g(INIT['b'][0]).replace('−', '-')}` を `{g(INIT['b'][0] + 0.01).replace('−', '-')}` に変えて実行し、損失が約 {fmt(BEFORE['grads']['b'][0] * 0.01, 4)}（$\partial\ell/\partial b_1 \times 0.01$）増えることを確かめてください。
    2. `b` を元に戻し、部屋を{_other[0]:g}℃・{_other[1]:g}%（`x = np.array([{_x[0]}, {_x[1]}])`）、正解を不快（`y = 0.0`）に変えると、勾配が0でないのはどのニューロンですか。{ROOM[0]:g}℃・{ROOM[1]:g}%の部屋と比べてください。
    """)
    return


@app.cell(hide_code=True)
def _(INIT, X_ROOM, Y_ROOM, mo):
    def _arr(a):
        a = a.tolist()
        return repr(a).replace(" ", "").replace(",", ", ")

    _code = "\n".join([
        f"x = np.array({_arr(X_ROOM)})  # 入力：26℃・40%",
        f"y = {Y_ROOM}                   # 正解：快適",
        "",
        f"W = np.array({_arr(INIT['W'])})  # 隠れ層の重み",
        f"b = np.array({_arr(INIT['b'])})  # 隠れ層のバイアス",
        f"v = np.array({_arr(INIT['v'])})  # 出力の重み",
        f"c = {INIT['c']}                                  # 出力のバイアス",
        "",
        "# 順伝播",
        "z = W @ x + b        # 4つのニューロンの重み付き和",
        "h = np.maximum(0, z) # ReLU",
        "p = 1 / (1 + np.exp(-(v @ h + c)))",
        "loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))",
        "",
        "# 逆伝播",
        "ds = p - y                         # ∂ℓ/∂s",
        "dz = np.where(z > 0, ds * v, 0.0)  # ∂ℓ/∂z（ReLUが0を出すニューロンは0）",
        "db = dz                            # ∂ℓ/∂b",
        "dW = np.outer(dz, x)               # ∂ℓ/∂W（行がニューロン）",
        "",
        "print('h     =', h)",
        "print('p     =', round(p, 3), '  損失 =', round(loss, 4))",
        "print('∂ℓ/∂b =', db.round(3))",
    ])
    ex_backprop = mo.ui.code_editor(value=_code, language="python", min_height=450).form(
        submit_button_label="▶", submit_button_tooltip="実行", bordered=False)
    return (ex_backprop,)


@app.cell(hide_code=True)
def _(ex_backprop, mo, np):
    def _run(code):
        import contextlib
        import io
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                exec(code, {"np": np})
        except Exception as exc:
            return mo.callout(mo.md(f"**エラー：** `{type(exc).__name__}: {exc}`　直前に変えた行を見直してください。"), kind="danger")
        return mo.plain_text(buffer.getvalue() or "（出力はありません）")

    # 01のmarimoのセルに似せた欄。▶で実行し、実行するまで出力欄は出さない
    _output = "" if ex_backprop.value is None else f'<div class="pycell-output">{_run(ex_backprop.value).text}</div>'
    mo.Html(f'<div class="pycell">{ex_backprop}{_output}</div>')
    return


@app.cell(hide_code=True)
def _(BEFORE, ETA, INIT, UPDATED, fmt, g, mo, np):
    _rows = []
    for _key, _sym in [("W", "w"), ("b", "b"), ("v", "v"), ("c", "c")]:
        for _idx in np.ndindex(np.shape(INIT[_key])):
            _grad = np.asarray(BEFORE["grads"][_key])[_idx]
            if _grad == 0:
                continue
            _sub = "".join(str(i + 1) for i in _idx)
            _label = f"${_sym}_{{{_sub}}}$" if _sub else f"${_sym}$"
            _rows.append(f"| {_label} | {g(np.asarray(INIT[_key])[_idx])} | {fmt(_grad)} | {fmt(np.asarray(UPDATED[_key])[_idx], 3)} |")
    _n = INIT["W"].size + INIT["b"].size + INIT["v"].size + 1
    mo.md(rf"""
    ## 6. 重みを更新する

    勾配が求まったら、03と同じように、すべてのパラメータを勾配と逆の向きに動かします。

    $$\theta \leftarrow \theta - \eta\,\frac{{\partial\ell}}{{\partial\theta}}$$

    学習率は、3節の学習と同じ $\eta = {ETA:g}$ とします。勾配が0でないのは次の{len(_rows)}個なので、変わるのもこの{len(_rows)}個です。

    | パラメータ | 更新前 | 勾配 | 更新後 |
    |---|---|---|---|
    {chr(10).join(_rows)}

    ほかの{_n - len(_rows)}個は勾配が0なので変わりません。

    重みが変わると、予測はどう変わるのでしょうか。図12でパラメータを更新し、同じ部屋の出力をもう一度計算して確かめます。青い動く点は、更新した重みで再計算した値を表します。
    """)
    return


@app.cell(hide_code=True)
def _(ROOM, figure, forward_data):
    figure("forward", forward_data("update"),
           f"図12　{ROOM[0]:g}℃・{ROOM[1]:g}%の部屋で、重みの更新が出力に伝わる様子。"
           "灰色のニューロンは勾配が0で、重みも値も変わらない。", height=560)
    return


@app.cell(hide_code=True)
def _(AFTER, BEFORE, INIT, UPDATED, fmt, g, mo):
    _on = [j for j in range(4) if BEFORE["h"][j] > 0]
    _names = "と".join(str(j + 1) for j in _on)
    _hs = "、".join(f"$h_{j + 1}$" for j in _on)
    _vs = "、".join(f"$v_{j + 1}$" for j in _on)
    mo.md(rf"""
    ニューロン{_names}は、出力の重みが負（{_vs} $= {g(INIT['v'][_on[0]])}$）なので、$h$ が大きいほど $s$ を下げていました。
    更新で、{_hs} は {g(BEFORE['h'][_on[0]])} から {fmt(AFTER['h'][_on[0]])} に小さくなり、{_vs} は {fmt(UPDATED['v'][_on[0]])} と0に近づき、$c$ は {fmt(UPDATED['c'])} に増えました。
    どれも $s$ を上げる向きの変化で、$s$ は {g(BEFORE['s'])} から {fmt(AFTER['s'])} に上がりました。

    その結果、$p$ は {BEFORE['p']:.2f} から {AFTER['p']:.2f} に上がって0.5を超え、この部屋を正しく「快適」と予測するようになりました。
    損失も {BEFORE['loss']:.2f} から {AFTER['loss']:.2f} に下がっています。
    勾配は、その場所で損失が増える向きを表します。今回は逆向きに更新したことで、損失が下がり、予測が正解に近づきました。
    """)
    return


@app.cell(hide_code=True)
def _(INIT_ACC, STEPS, mo):
    mo.md(rf"""
    ただし、1部屋だけに合わせて重みを動かすと、ほかの部屋の予測が悪くなることもあります。
    実際の学習では、03と同じく、全部屋（またはミニバッチ）の損失の平均について勾配を求めます。
    そして、**順伝播 → 損失 → 逆伝播 → 更新**を何度も繰り返します。

    3節の図7は、0回目（正解率{INIT_ACC:.0%}）のネットワークから、この繰り返しを{STEPS[-1]}回まで行ったものでした。
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
    quiz_relu = mo.ui.radio(["層を重ねるほど、複雑に折れ曲がった境界を作れる", "一つの重み付き和にまとまり、境界は直線になる", "学習ができなくなり、境界が決まらない"],
                            label="**Q1.** 2クラスを最終スコアの符号で分けるネットワークがあります。隠れ層のReLUを外し（$h = z$ とし）、重み付き和だけの層を何層も重ねると、決定境界はどうなりますか？")
    quiz_relu
    return (quiz_relu,)


@app.cell(hide_code=True)
def _(check, quiz_relu):
    check(quiz_relu, "一つの重み付き和にまとまり、境界は直線になる",
          "重み付き和の重み付き和は、また一つの重み付き和です。層をいくつ重ねても、境界は直線です。"
          "ReLUのように折れ曲がる関数を挟むと、境界を折り曲げられます。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_chain = mo.ui.radio(["−0.8：途中の微分をすべて掛ける",
                              "0.4：途中の微分をすべて足す",
                              "0：ReLUを通ると勾配は消える"],
                             label="**Q2.** $h = \\operatorname{ReLU}(z)$ とします。損失 $\\ell$ からバイアス $b$ まで、$\\ell \\rightarrow s \\rightarrow h \\rightarrow z \\rightarrow b$ と一本の経路でつながっています。"
                                   "$\\partial\\ell/\\partial s = 0.4$、$\\partial s/\\partial h = -2$、$\\partial h/\\partial z = 1$、$\\partial z/\\partial b = 1$ のとき、$\\partial\\ell/\\partial b$ はいくつですか？")
    quiz_chain
    return (quiz_chain,)


@app.cell(hide_code=True)
def _(check, quiz_chain):
    check(quiz_chain, "−0.8：途中の微分をすべて掛ける",
          "連鎖律では、一本の経路の途中の微分を掛けます。$0.4 \\times (-2) \\times 1 \\times 1 = -0.8$ です。"
          "途中に微分が0の段があれば、全体の勾配も0になります。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_zero = mo.ui.radio(["0になる", "出力層の重みが大きければ、大きな値になる", "損失側から伝わる勾配と同じ値になる"],
                            label="**Q3.** あるニューロンは $z = wx + b$ をReLUに通して $h = \\max(0, z)$ を出し、損失には $h$ を通じてのみ影響します。"
                                  "$z = -2.5$ のとき、この損失の $b$ に関する勾配はどうなりますか？")
    quiz_zero
    return (quiz_zero,)


@app.cell(hide_code=True)
def _(check, quiz_zero):
    check(quiz_zero, "0になる",
          "バイアスを少し変えても $z$ は負のままで、ReLUの出力 $h$ は0から変わりません。予測も損失も変わらないので、勾配は0です。"
          "連鎖律で見ると、途中にReLUの傾き0が掛かるので、出力の重みがどれだけ大きくても0になります。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_backward = mo.ui.radio(["まだ変わっていない", "勾配の分だけ、すでに変わっている", "出力に近い層の重みだけ、すでに変わっている"],
                                label="**Q4.** 逆伝播ですべてのパラメータの勾配を求め終わった直後、ネットワークの重みはどうなっていますか？")
    quiz_backward
    return (quiz_backward,)


@app.cell(hide_code=True)
def _(check, quiz_backward):
    check(quiz_backward, "まだ変わっていない",
          "逆伝播は勾配を計算するだけで、そのあいだ重みは順伝播のときのまま使います。"
          "重みが変わるのは、6節の更新 $\\theta \\leftarrow \\theta - \\eta\\,\\partial\\ell/\\partial\\theta$ を行ったときです。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_infer = mo.ui.radio(["順伝播だけ", "順伝播と逆伝播", "順伝播・逆伝播・更新のすべて"],
                             label="**Q5.** 学習が終わったネットワークで、新しい部屋が快適かどうかを予測します。必要な計算はどれですか？")
    quiz_infer
    return (quiz_infer,)


@app.cell(hide_code=True)
def _(check, quiz_infer):
    check(quiz_infer, "順伝播だけ",
          "新しい部屋の正解はまだ分からないので、損失も勾配も計算できませんし、重みを変える必要もありません。"
          "入力から順に各層を計算して予測を出すだけです。損失・逆伝播・更新は、学習のときに使います。")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    | 工程 | 何を計算するか | 正解を使うか | 重みは変わるか |
    |---|---|---|---|
    | 順伝播 | 入力から各層の値と、予測 $p$ | 使わない | 変わらない |
    | 損失 | 予測と正解のずれ $\ell$ | 使う | 変わらない |
    | 逆伝播 | 各パラメータの勾配 $\partial\ell/\partial\theta$ | 損失を通して使う | 変わらない |
    | 更新 | $\theta \leftarrow \theta - \eta\,\partial\ell/\partial\theta$ | － | **ここで変わる** |

    - **この章の隠れ層は重み付き和をReLUに通し、出力層はsigmoidに通す。** 中間のReLUの折れ曲がりを次の層で組み合わせると、直線でない境界を作れる。
    - **重みは人が決めず、ランダムな値から学習する。** 各ニューロンがどこで折れるかも、学習の結果として決まる。
    - **勾配は連鎖律で求める。** 一本の経路では途中の微分を掛ける。逆伝播は出力側から順に、途中の結果を使い回す。
    - **学習は「順伝播 → 損失 → 逆伝播 → 更新」の繰り返し。** 推論は順伝播だけ。

    **この章の確認：** 逆伝播では勾配を求め、重みは更新の段階で初めて変わると説明できれば完了です。

    次の05では、PyTorchで手書き数字を分類するネットワークを学習します。逆伝播の計算はPyTorchに任せます。
    """)
    return


@app.cell(hide_code=True)
def _(g):
    def affine_text(w, b, names=("$x$₁", "$x$₂")):
        """重み付き和の式（例：−0.5$x$₁ − 0.5$x$₂ + 1）。0の重みは書かない。"""
        text = ""
        for wi, name in zip(w, names):
            if wi == 0:
                continue
            coef = "" if abs(wi) == 1 else g(abs(wi))
            sign = "−" if wi < 0 else "+"
            text += (f" {sign} " if text else ("−" if wi < 0 else "")) + coef + name
        return text + (f" {'−' if b < 0 else '+'} {g(abs(b))}" if b else "")

    return (affine_text,)


@app.cell(hide_code=True)
def _(INIT):
    _sub = "₁₂₃₄"

    def network_data(mode="structure"):
        """図5のネットワーク図に渡す値。"""
        inputs = [{"name": "$x$₁", "detail": "(気温 − 23) ÷ 3"}, {"name": "$x$₂", "detail": "(湿度 − 50) ÷ 10"}]
        hidden = [{"name": f"ニューロン{j + 1}", "weights": INIT["W"][j].tolist(),
                   "structure": [f"$z${k} = $w${k}₁$x$₁ + $w${k}₂$x$₂ + $b${k}", f"$h${k} = max(0, $z${k})"]}
                  for j, k in enumerate(_sub)]
        output = {"structure": ["$s$ = $v$₁$h$₁ + $v$₂$h$₂ + $v$₃$h$₃ + $v$₄$h$₄ + $c$", "$p$ = σ($s$)"]}
        return {"mode": mode, "inputs": inputs, "hidden": hidden, "output": output}

    return (network_data,)


@app.cell(hide_code=True)
def _(AFTER, BEFORE, INIT, ROOM, UPDATED, X_ROOM, Y_ROOM, fmt, g):
    _sub = "₁₂₃₄"

    def _num(value):
        """式に代入する数。負の数は括弧でくくる。"""
        return f"({g(value)})" if value < 0 else g(value)

    def _val(value):
        """0.5刻みならそのまま、そうでなければ小数2桁。"""
        return g(value) if float(value) * 2 == round(float(value) * 2) else fmt(value)

    def _change(name, old, new):
        return f"{name} {_val(old)} → {_val(new)}"

    def _verdict(p):
        return "快適と予測" if p >= 0.5 else "不快と予測"

    def forward_data(mode):
        """図9（順伝播を一段ずつ）と図12（1回の更新の前後）に渡す値。mode は steps / update。"""
        inputs = [{"name": f"$x${_sub[i]}", "value": g(X_ROOM[i]), "raw": f"{ROOM[i]:g}{'℃%'[i]}"} for i in range(2)]
        hidden = []
        for j in range(4):
            k = _sub[j]
            item = {"name": f"ニューロン{j + 1}", "zName": f"$z${k}", "hName": f"$h${k}",
                    "sym": f"$z${k} = $w${k}₁$x$₁ + $w${k}₂$x$₂ + $b${k}",
                    "terms": [f"{_num(w)}×{_num(x)}" for w, x in zip(INIT["W"][j], X_ROOM)] + [_num(INIT["b"][j])],
                    "z": float(BEFORE["z"][j]), "h": float(BEFORE["h"][j]),
                    "zText": g(BEFORE["z"][j]), "hText": g(BEFORE["h"][j])}
            if mode == "update":
                names = [f"$w${k}₁", f"$w${k}₂", f"$b${k}"]
                olds = [*INIT["W"][j], INIT["b"][j]]
                news = [*UPDATED["W"][j], UPDATED["b"][j]]
                item |= {"changes": [_change(n, a, b) for n, a, b in zip(names, olds, news) if a != b],
                         "beforeParams": [f"{n} = {_val(a)}" for n, a, b in zip(names, olds, news) if a != b],
                         "afterParams": [f"{n} = {_val(b)}" for n, a, b in zip(names, olds, news) if a != b],
                         "z2": float(AFTER["z"][j]), "h2": float(AFTER["h"][j]),
                         "zBefore": _val(BEFORE["z"][j]), "zAfter": _val(AFTER["z"][j]),
                         "hBefore": _val(BEFORE["h"][j]), "hAfter": _val(AFTER["h"][j]),
                         "zText": f"{_val(BEFORE['z'][j])} → {_val(AFTER['z'][j])}" if AFTER["z"][j] != BEFORE["z"][j] else _val(BEFORE["z"][j]),
                         "hText": f"{_val(BEFORE['h'][j])} → {_val(AFTER['h'][j])}" if AFTER["h"][j] != BEFORE["h"][j] else _val(BEFORE["h"][j])}
            hidden.append(item)
        output = {"sym": "$s$ = $v$₁$h$₁ + $v$₂$h$₂ + $v$₃$h$₃ + $v$₄$h$₄ + $c$",
                  "terms": [f"{_num(v)}×{g(h)}" for v, h in zip(INIT["v"], BEFORE["h"])] + [_num(INIT["c"])],
                  "s": BEFORE["s"], "sText": g(BEFORE["s"]),
                  "pText": f"$p$ = σ({g(BEFORE['s'])}) = {BEFORE['p']:.2f}",
                  "verdict": f"0.5{'以上' if BEFORE['p'] >= 0.5 else '未満'}なので{_verdict(BEFORE['p'])}"}
        if mode == "update":
            names = [f"$v${k}" for k in _sub] + ["$c$"]
            olds, news = [*INIT["v"], INIT["c"]], [*UPDATED["v"], UPDATED["c"]]
            output |= {"changes": [_change(n, a, b) for n, a, b in zip(names, olds, news) if a != b],
                       "beforeParams": [f"{n} = {_val(a)}" for n, a, b in zip(names, olds, news) if a != b],
                       "afterParams": [f"{n} = {_val(b)}" for n, a, b in zip(names, olds, news) if a != b],
                       "s2": AFTER["s"], "sText": f"{_val(BEFORE['s'])} → {_val(AFTER['s'])}",
                       "sBefore": _val(BEFORE["s"]), "sAfter": _val(AFTER["s"]),
                       "pBefore": f"$p$ = σ({_val(BEFORE['s'])}) = {BEFORE['p']:.2f}",
                       "pAfter": f"$p$ = σ({_val(AFTER['s'])}) = {AFTER['p']:.2f}",
                       "verdictBefore": _verdict(BEFORE["p"]), "verdictAfter": _verdict(AFTER["p"]),
                       "lossBefore": f"損失 ℓ = {BEFORE['loss']:.2f}", "lossAfter": f"損失 ℓ = {AFTER['loss']:.2f}",
                       "pText": f"$p$ = {BEFORE['p']:.2f} → {AFTER['p']:.2f}",
                       "verdict": f"{_verdict(BEFORE['p'])} → {_verdict(AFTER['p'])}",
                       "loss": f"損失 ℓ = {BEFORE['loss']:.2f} → {AFTER['loss']:.2f}"}
            return {"mode": mode, "inputs": inputs, "hidden": hidden, "output": output}

        _off = "と".join(str(j + 1) for j in range(4) if BEFORE["h"][j] == 0)
        # 説明は、PC幅で1行に収まる長さにする（長いと改行が入り、図の位置がずれる）
        status = [
            f"入力は $x$₁ = {g(X_ROOM[0])}、$x$₂ = {g(X_ROOM[1])} です。入力側の層から順に計算します。",
            "隠れ層の4つのニューロンが、それぞれ自分の重みとバイアスで $z$ を計算します。",
            f"4つの $z$ をReLUに通します。$z$ が負のニューロン{_off}は $h$ = 0 です。",
            f"出力のニューロンが $h$₁〜$h$₄ の重み付き和 $s$ を計算します。$h$ = 0 の項は0です。",
            f"$s$ を sigmoid に通すと $p$ = {BEFORE['p']:.2f}。{output['verdict']}しました（正解は快適）。",
        ]
        return {"mode": mode, "inputs": inputs, "hidden": hidden, "output": output,
                "stages": ["入力", "重み付き和", "ReLU", "出力の重み付き和", "sigmoid"], "status": status}

    def backward_data():
        """図11（逆伝播を一段ずつ）に渡す値。"""
        inputs = [{"name": f"$x${_sub[i]}", "value": g(X_ROOM[i]), "raw": f"{ROOM[i]:g}{'℃%'[i]}"} for i in range(2)]
        ds = fmt(BEFORE["ds"])
        hidden = []
        for j in range(4):
            k = _sub[j]
            on = BEFORE["z"][j] > 0
            dz = fmt(BEFORE["dz"][j]) if on else "0"
            grads = ([f"∂ℓ/∂$w${k}{_sub[i]} = {fmt(BEFORE['grads']['W'][j, i])}" for i in range(2)] + [f"∂ℓ/∂$b${k} = {dz}"]
                     if on else ["重みとバイアスの勾配はすべて0"])
            hidden.append({"name": f"ニューロン{j + 1}", "z": float(BEFORE["z"][j]), "active": bool(on),
                           "value": f"$z${k} = {g(BEFORE['z'][j])}、$h${k} = {g(BEFORE['h'][j])}",
                           "dh": f"∂ℓ/∂$h${k} = {ds} × {_num(INIT['v'][j])} = {fmt(BEFORE['dh'][j])}",
                           "slope": f"傾き {1 if on else 0}",
                           "dz": f"∂ℓ/∂$z${k} = {fmt(BEFORE['dh'][j])} × {1 if on else 0} = {dz}",
                           "grads": grads})
        _on = [j for j in range(4) if BEFORE["h"][j] > 0]
        output = {"values": [f"$p$ = {BEFORE['p']:.2f}、正解 $y$ = {Y_ROOM:g}", f"ℓ = −log $p$ = {BEFORE['loss']:.2f}"],
                  "ds": f"∂ℓ/∂$s$ = $p$ − $y$ = {ds}",
                  "grads": [f"∂ℓ/∂$v${_sub[j]} = {fmt(BEFORE['grads']['v'][j])}" for j in _on] + [f"∂ℓ/∂$c$ = {ds}"],
                  "rest": "（$h$ = 0 のニューロンでは ∂ℓ/∂$v$ = 0）"}
        _off = "と".join(str(j + 1) for j in range(4) if BEFORE["z"][j] < 0)
        status = [
            f"予測 $p$ = {BEFORE['p']:.2f} と正解 $y$ = {Y_ROOM:g} から、損失 ℓ = {BEFORE['loss']:.2f} を求めました。ここから逆向きに戻ります。",
            "損失を出力の重み付き和 $s$ で微分します。03と同じく ∂ℓ/∂$s$ = $p$ − $y$ です。",
            "∂ℓ/∂$s$ に、それぞれの出力の重み $v$ⱼ を掛けて、∂ℓ/∂$h$ⱼ を求めます。",
            f"ReLUの傾き（$z$ > 0 なら1、$z$ < 0 なら0）を掛けます。ニューロン{_off}は0になります。",
            "∂ℓ/∂$z$ⱼ に入力 $x$ᵢ を掛けると重みの勾配、そのままならバイアスの勾配です。",
        ]
        return {"inputs": inputs, "hidden": hidden, "output": output,
                "stages": ["損失", "出力", "隠れ層の出力", "ReLU", "重みとバイアス"], "status": status}

    return backward_data, forward_data


if __name__ == "__main__":
    app.run()
