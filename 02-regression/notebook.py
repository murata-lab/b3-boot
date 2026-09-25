import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="02 回帰とモデルの選び方", css_file="notebook.css")


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

    return check, figure


@app.cell(hide_code=True)
def _(np):
    def mse(y, predicted):
        return float(np.mean((np.asarray(predicted) - np.asarray(y)) ** 2))

    def least_squares(features, y):
        """切片と各特徴量の重みを、MSEが最小になるように求める。戻り値は [b, w1, w2, ...]。"""
        design = np.column_stack([np.ones(len(y)), *features])
        return np.linalg.lstsq(design, y, rcond=None)[0]

    def pairs(xs, ys, digits=3):
        return np.round(np.column_stack([xs, ys]), digits).tolist()

    return least_squares, mse, pairs


@app.cell(hide_code=True)
def _(np):
    # 家賃の例（説明用に作ったデータ）。家賃は広さで上がり、駅から遠いほど下がる。
    # 図で点が重ならないよう、近すぎる点と、1節で「家賃が分からない部屋」とする28㎡付近の点は作り直す。
    QUERY_AREA = 28
    _rng = np.random.default_rng(6)
    _rooms = []
    while len(_rooms) < 30:
        _a = round(float(_rng.uniform(15, 45)), 1)
        _m = float(_rng.integers(1, 21))
        _r = round(1.5 + 0.2 * _a - 0.12 * _m + float(_rng.normal(0, 0.35)), 1)
        if abs(_a - QUERY_AREA) < 1.5:
            continue
        if any(((_a - a) / 40) ** 2 + ((_r - r) / 12) ** 2 < 0.03 ** 2 for a, _, r in _rooms):
            continue
        _rooms.append((_a, _m, _r))
    AREA, WALK, RENT = (np.array(v) for v in zip(*_rooms))
    # 2節でMSEを手計算する5件（広さがばらけるように選ぶ）
    FIVE = np.argsort(AREA)[[2, 8, 15, 21, 27]]
    return AREA, FIVE, QUERY_AREA, RENT, WALK


@app.cell(hide_code=True)
def _(np, pairs):
    # 気温の例（説明用に作ったデータ）。本当の変化は14時に最高、2時に最低。
    NOISE = 1.3
    GRID_T = np.linspace(0, 24, 193)

    def true_temperature(t):
        return 16 + 6 * np.cos(2 * np.pi * (np.asarray(t) - 14) / 24)

    def measure(seed, n):
        """ほぼ等間隔の時刻で n 回測る（訓練データ）。"""
        rng = np.random.default_rng(seed)
        t = np.linspace(0.6, 23.4, n) + rng.uniform(-0.9, 0.9, n) * 10 / n
        return t, true_temperature(t) + rng.normal(0, NOISE, n)

    def measure_random(seed, n):
        """ランダムな時刻で n 回測る（検証・テストデータ）。"""
        rng = np.random.default_rng(seed)
        t = np.sort(rng.uniform(0, 24, n))
        return t, true_temperature(t) + rng.normal(0, NOISE, n)

    TRAIN = measure(13, 10)
    VALID = measure_random(99, 40)
    TEST = measure_random(2026, 100)
    # 「もし同じ回数を測り直したら」の訓練データ（時刻と測定誤差が変わる）
    OTHER_SEEDS = (21, 22, 23, 24, 25, 26)
    TEMPERATURE_BASE = {"grid": np.round(GRID_T, 3).tolist(), "truth": np.round(true_temperature(GRID_T), 3).tolist(),
                        "valid": pairs(*VALID)}
    return GRID_T, OTHER_SEEDS, TEMPERATURE_BASE, TEST, TRAIN, VALID, measure


@app.cell(hide_code=True)
def _(GRID_T, VALID, mse, np, pairs):
    def fit_polynomial(t, y, degree, strength=0.0):
        """d次の多項式を「訓練MSE + λΣw²」が最小になるように当てはめ、予測する関数と重みを返す。

        時刻は0〜1に直し、各特徴量 x, x², … を訓練データの平均と標準偏差で標準化してから解く。
        切片は罰則に含めない。
        """
        x = np.asarray(t) / 24
        powers = np.arange(1, degree + 1)
        features = x[:, None] ** powers
        mean, scale = features.mean(axis=0), features.std(axis=0)
        design = np.column_stack([np.ones(len(x)), (features - mean) / scale])
        penalty = np.diag([0.0] + [np.sqrt(len(x) * strength)] * degree)
        weights = np.linalg.lstsq(np.vstack([design, penalty]), np.concatenate([y, np.zeros(degree + 1)]), rcond=None)[0]

        def predict(new_t):
            z = ((np.asarray(new_t)[:, None] / 24) ** powers - mean) / scale
            return weights[0] + z @ weights[1:]

        return predict, weights

    def fit_panel(title, t, y, degree, show_valid):
        """図の1枚分：訓練データと、そこから学習した曲線。"""
        predict, _ = fit_polynomial(t, y, degree)
        stats = f"訓練MSE {mse(y, predict(t)):.2f}"
        if show_valid:
            stats += f"　検証MSE {mse(VALID[1], predict(VALID[0])):.2f}"
        return {"title": title, "stats": stats, "train": pairs(t, y), "curve": np.round(predict(GRID_T), 3).tolist()}

    return fit_panel, fit_polynomial


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <header class="hero">
      <p class="eyebrow">機械学習入門</p>
      <h1>回帰とモデルの選び方</h1>
      <p>家賃を予測する式を作り、未知のデータにも使えるか確かめます。</p>
    </header>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    部屋を探すとき、広さは分かっていても、その部屋の家賃はまだ分からないことがあります。近くの部屋の広さと家賃を手掛かりに、家賃の目安を出せるでしょうか。
    このように、分かっている情報から数値を予測する問題を**回帰（regression）**と呼びます。

    **この章の目標：** 訓練MSEだけでモデルを選べない理由を、検証データとの違いから説明することです。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 広さから家賃を予測する

    下の点は、駅の周辺にある30件の部屋の広さと家賃です（説明用のデータです）。
    """)
    return


@app.cell(hide_code=True)
def _(AREA, QUERY_AREA, RENT, figure, pairs):
    figure("rentScatter", {"points": pairs(AREA, RENT, 1), "query": QUERY_AREA},
           f"図1　30件の部屋の広さと家賃。{QUERY_AREA}㎡の部屋の家賃はまだ分からない。", height=400)
    return


@app.cell(hide_code=True)
def _(QUERY_AREA, mo):
    mo.md(rf"""
    広い部屋ほど家賃が高い傾向があります。では、**まだ家賃が分からない{QUERY_AREA}㎡の部屋**はいくらくらいでしょうか。
    点の集まりの真ん中を通る直線を引ければ、そこから読み取れそうです。直線は次の式で表せます。

    $$\hat y = wx + b$$

    | 記号 | 名前 | この例では |
    |---|---|---|
    | $x$ | **入力** | 部屋の広さ（㎡） |
    | $\hat y$（ワイハット） | **予測** | 式が出す家賃 |
    | $y$ | **正解** | 実際の家賃 |
    | $w,\ b$ | **パラメータ** | 直線の傾きと切片。**これをデータから決めることを学習と呼ぶ** |

    予測 $\hat y$ は式から計算できますが、正解 $y$ は実際に調べないと分かりません。二つを比べ、ずれが小さくなるようにパラメータを決めます。
    """)
    return


@app.cell(hide_code=True)
def _(figure):
    figure("pipeline", {}, "図2　学習の流れ。予測と正解のずれを損失で測り、損失が小さくなるようにパラメータを直す。", height=300)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 予測のずれを測る：平均二乗誤差（MSE）

    直線がデータにどれくらい合っているかを比べるには、ずれを一つの数にする必要があります。
    まず5件の部屋と、試しに引いた直線で考えます。
    """)
    return


@app.cell(hide_code=True)
def _(AREA, FIVE, RENT, figure, mse, pairs):
    _w, _b = 0.15, 2.5
    _pred = _w * AREA[FIVE] + _b
    _terms = " + ".join(f"{(p - y) ** 2:.2f}" for p, y in zip(_pred, RENT[FIVE]))
    figure("residualSquares", {"points": pairs(AREA[FIVE], RENT[FIVE], 1), "w": _w, "b": _b},
           f"図3　直線 $ŷ$ = {_w}$x$ + {_b} と5件の部屋。縦の線が残差、正方形の面積が二乗誤差。"
           f"MSE = ({_terms}) ÷ 5 = {mse(RENT[FIVE], _pred):.2f}", height=400)
    return


@app.cell(hide_code=True)
def _(FIVE, RENT, mo):
    _y = RENT[FIVE]
    mo.md(rf"""
    各点で、予測と正解の差 $\hat y_i - y_i$ を測ります。これを**残差**と呼びます。
    そのまま足すとプラスとマイナスが打ち消し合うので、二乗してから平均します。

    $$L = \frac{{1}}{{n}}\sum_{{i=1}}^{{n}}(\hat y_i - y_i)^2$$

    これが**平均二乗誤差（MSE: mean squared error）**です。図3の正方形の面積が各点の二乗誤差で、MSEはその平均です。
    予測のずれを表す数を一般に**損失（loss）**と呼び、小さいほどよい予測です。
    二乗しているので、大きく外れた点ほど強く効きます（ずれが2倍なら4倍）。

    ### コードで確かめる

    図3のMSEを、NumPyで計算するコードです。`squared` が各点の二乗誤差（図3の正方形の面積）です。

    1. そのまま実行し、図3と同じMSEになることを確かめてください。
    2. 最後の部屋の家賃 `{_y[-1]:.1f}`（万円）を `{_y[-1] + 6:.1f}` に変え、相場から大きく外れた部屋を1件だけ作ります。MSEは何倍になりましたか。どの点の二乗誤差が効いていますか。
    """)
    return


@app.cell(hide_code=True)
def _(AREA, FIVE, RENT, mo):
    _area = ", ".join(f"{v:.1f}" for v in AREA[FIVE])
    _rent = ", ".join(f"{v:.1f}" for v in RENT[FIVE])
    _code = "\n".join([
        f"x = np.array([{_area}])  # 広さ（入力）",
        f"y = np.array([{_rent}])  # 家賃（万円、正解）",
        "w = 0.15  # 傾き",
        "b = 2.5   # 切片",
        "",
        "y_hat = w * x + b          # 予測",
        "squared = (y_hat - y) ** 2  # 各点の二乗誤差",
        "print('二乗誤差:', squared.round(2))",
        "print('MSE:', round(np.mean(squared), 2))",
    ])
    ex_mse = mo.ui.code_editor(value=_code, language="python", min_height=210).form(
        submit_button_label="▶", submit_button_tooltip="実行", bordered=False)
    return (ex_mse,)


@app.cell(hide_code=True)
def _(ex_mse, mo, np):
    def _run(code):
        import contextlib
        import io
        buffer = io.StringIO()
        try:
            # 外れた点を作ると値の桁が大きく違い、指数表記（2.992e+01 など）になるのを防ぐ
            with contextlib.redirect_stdout(buffer), np.printoptions(suppress=True):
                exec(code, {"np": np})
        except Exception as exc:
            return mo.callout(mo.md(f"**エラー：** `{type(exc).__name__}: {exc}`　直前に変えた行を見直してください。"), kind="danger")
        return mo.plain_text(buffer.getvalue() or "（出力はありません）")

    # 01のmarimoのセルに似せた欄。▶で実行し、実行するまで出力欄は出さない
    _output = "" if ex_mse.value is None else f'<div class="pycell-output">{_run(ex_mse.value).text}</div>'
    mo.Html(f'<div class="pycell">{ex_mse}{_output}</div>')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 損失の地形：学習とは「谷底」を探すこと

    $w$ と $b$ を決めると直線が1本決まり、30件全体のMSEが一つ決まります。
    では、いろいろな $w, b$ を試すと、MSEはどう変わるでしょうか。

    下の図の右側は、横軸が傾き $w$、縦軸が切片 $b$ の平面です。**平面の1点が、1組の $w, b$、つまり1本の直線に対応します。**
    右側でいくつかの点を選び、左側の直線とMSEの変化を見てください。
    """)
    return


@app.cell(hide_code=True)
def _(AREA, RENT, contour_generator, figure, least_squares, np, pairs):
    _ws = np.linspace(0.05, 0.35, 161)
    _bs = np.linspace(-3.0, 5.0, 161)
    _W, _B = np.meshgrid(_ws, _bs)
    _Z = np.mean((_W[..., None] * AREA + _B[..., None] - RENT) ** 2, axis=-1)
    _levels = [0, 1, 1.3, 1.7, 2.3, 3.2, 4.5, 6.5, 9.5, 14, 20, 30, 1e9]
    _gen = contour_generator(_ws, _bs, _Z, fill_type="OuterOffset")
    _bands = []
    for _lo, _hi in zip(_levels[:-1], _levels[1:]):
        _rings = []
        for _pts, _offsets in zip(*_gen.filled(_lo, _hi)):
            for _s, _e in zip(_offsets[:-1], _offsets[1:]):
                _rings.append(np.round(_pts[_s:_e], 4).ravel().tolist())
        _bands.append(_rings)
    LINE_B, LINE_W = (float(v) for v in least_squares([AREA], RENT))
    figure("landscape", {
        "points": pairs(AREA, RENT, 1), "bands": _bands, "levels": _levels[1:-1],
        "wlim": [0.05, 0.35], "blim": [-3.0, 5.0], "best": [LINE_W, LINE_B],
    }, "図4　左：データと、選んだ w, b の直線（縦の線は残差）。右：傾き w と切片 b の平面。", height=600)
    return LINE_B, LINE_W


@app.cell(hide_code=True)
def _(LINE_B, LINE_W, QUERY_AREA, mo):
    mo.md(rf"""
    地形の一番低い場所（谷底）が、30件全体に最もよく合う直線です。
    MSEを最小にする $w, b$ を求める方法を**最小二乗法**と呼びます。
    手探りで谷底ぴったりに止まるのは大変ですが、この問題ではMSEが $w, b$ の2次式なので、谷底の位置を式で一度に計算できます。

    これで、図1の問いに答えられます。谷底の直線に $x = {QUERY_AREA}$ を代入すると、
    $\hat y = {LINE_W:.3f} \times {QUERY_AREA} {LINE_B:+.2f} \approx {LINE_W * QUERY_AREA + LINE_B:.1f}$ で、{QUERY_AREA}㎡の部屋の家賃の目安は約{LINE_W * QUERY_AREA + LINE_B:.1f}万円です。

    谷は細長く、斜めに伸びています。傾き $w$ を大きくしても、同時に切片 $b$ を小さくすれば、MSEはあまり増えません。
    左の図で見ると、直線が点の集まりの中心あたりを軸に回転している状態です。

    **学習とは、損失の低い場所を探すことです。** パラメータが多く地形全体を描けない場合は、坂の向きを手掛かりに少しずつ下ります。03で扱う勾配降下法です。
    """)
    return


@app.cell(hide_code=True)
def _(AREA, LINE_B, LINE_W, RENT, mo):
    _mx, _my = AREA.mean(), RENT.mean()
    mo.accordion({"最小二乗法の解（任意）": mo.md(rf"""
    MSEを $w$ と $b$ でそれぞれ微分して0とおくと、谷底は次の式で求まります（$\bar x, \bar y$ は平均）。

    $$w = \frac{{\sum_i (x_i-\bar x)(y_i-\bar y)}}{{\sum_i (x_i-\bar x)^2}},\qquad b = \bar y - w\bar x$$

    この例では $\bar x = {_mx:.2f}$、$\bar y = {_my:.2f}$ で、$w = {LINE_W:.3f}$、$b = {LINE_B:.2f}$ です。
    $b = \bar y - w\bar x$ は「直線は必ず点 $(\bar x, \bar y)$ を通る」という意味です。谷が斜めに伸びているのは、この点を軸に回転する直線のMSEがあまり変わらないからです。
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 手掛かりを増やす：重回帰

    最小二乗法で求めた直線でも、MSEは0になりません。図1をよく見ると、同じくらいの広さでも家賃がかなり違う部屋があります。
    何が違うのでしょうか。

    下の図では、部屋を**駅から10分以内**（●）と**10分より遠い**（○）に分けて描きました。
    「広さだけ」の直線に対して、●と○がどちら側にあるかを見てください。
    """)
    return


@app.cell(hide_code=True)
def _(AREA, LINE_B, LINE_W, RENT, WALK, figure, least_squares, mse, np):
    _b, _w1, _w2 = (float(v) for v in least_squares([AREA, WALK], RENT))
    MULTI = {"simple": {"w": LINE_W, "b": LINE_B, "mse": mse(RENT, LINE_W * AREA + LINE_B)},
             "multi": {"w1": _w1, "w2": _w2, "b": _b, "mse": mse(RENT, _w1 * AREA + _w2 * WALK + _b)}}
    figure("multiFeature", {"points": np.round(np.column_stack([AREA, RENT, WALK]), 1).tolist(), "split": 10,
                            "examples": [5, 15], **MULTI},
           "図5　駅からの徒歩時間で部屋を二つに分けた散布図。「使う特徴量」を切り替えると、予測の直線が変わる。", height=480)
    return (MULTI,)


@app.cell(hide_code=True)
def _(MULTI, mo):
    _s, _m = MULTI["simple"], MULTI["multi"]
    mo.md(rf"""
    「広さだけ」の直線に対して、駅に近い部屋（●）はほとんどが上に、遠い部屋（○）は下にあります。
    同じ広さでも駅に近いほど家賃が高いのに、広さだけの直線ではこの差を表せません。

    そこで、徒歩時間 $x_2$（分）も入力に加えます。

    $$\hat y = w_1 x_1 + w_2 x_2 + b$$

    予測に使う入力の一つひとつを**特徴量**と呼び、特徴量が複数ある線形回帰を**重回帰**と呼びます。
    パラメータが3つに増えても、MSEが最小になる値を最小二乗法で求めるのは同じです。この例では
    $\hat y = {_m['w1']:.3f}\,x_1 {_m['w2']:+.3f}\,x_2 {_m['b']:+.2f}$ になりました。

    徒歩時間を一つ決めると、$w_2 x_2 + b$ はただの数になります。つまり重回帰は、**徒歩時間ごとに高さがずれた直線**を使い分けていることになります。
    同じ広さなら、駅から1分遠くなるごとに予測は約{round(abs(_m['w2']) * 10000, -2):,.0f}円下がります。
    MSEは {_s['mse']:.2f} から {_m['mse']:.2f} に下がりました。

    特徴量を増やすと、学習に使ったデータのMSEは**上がりません**。役に立たない特徴量でも、たまたま合えば下がります。
    まだ見ていない部屋にも通用するかは、学習に使ったデータのMSEだけでは分かりません。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"行列で書く（任意）": mo.md(r"""
    $n$ 件のデータを1行1件の行列 $X$（先頭の列はすべて1）にまとめると、全件の予測は $\hat{\boldsymbol y} = X\boldsymbol w$ と1行で書けます。
    ここで $\boldsymbol w = [b, w_1, w_2]^\top$ です。MSEを最小にする $\boldsymbol w$ は $X^\top X \boldsymbol w = X^\top \boldsymbol y$ を解いて求まります（正規方程式）。
    05のPyTorchでも、入力を行列にまとめて一度に計算します。
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 曲線で表す：多項式回帰と過学習

    特徴量を増やしすぎると何が起きるかを、曲線の形で見やすい例で確かめます。
    今度は、1日に10回測った気温から、測っていない時刻の気温を推定します。
    """)
    return


@app.cell(hide_code=True)
def _(GRID_T, TRAIN, figure, fit_polynomial, pairs):
    _line, _ = fit_polynomial(*TRAIN, 1)
    figure("temperatureLine", {"train": pairs(*TRAIN), "line": pairs(GRID_T, _line(GRID_T))},
           "図6　10回測った気温と、最小二乗法で求めた直線。", height=380)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    気温は昼に上がって夜に下がるので、直線では表せません。
    そこで、時刻 $x$ から $x^2, x^3, \dots$ を作り、それらも特徴量にします。

    $$\hat y = b + w_1 x + w_2 x^2 + \dots + w_d x^d$$

    これを **$d$ 次の多項式回帰**と呼びます。特徴量を計算で作っただけなので、重回帰と同じく最小二乗法で重みを求められます。
    次数 $d$ を上げるほど重みが増え、曲線は複雑に曲がれるようになります。

    ### 訓練データのMSEだけで選ぶと

    学習に使う10回の測定を**訓練データ**と呼びます。3次と9次の曲線を比べましょう。
    """)
    return


@app.cell(hide_code=True)
def _(TEMPERATURE_BASE, TRAIN, figure, fit_panel):
    figure("compareFits", {**TEMPERATURE_BASE, "showValid": False, "panels": [
        fit_panel("3次", *TRAIN, 3, False), fit_panel("9次", *TRAIN, 9, False)]},
        "図7　同じ訓練データ10回から学習した3次と9次の曲線。", height=420)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    訓練データのMSEで比べると、9次のほうがずっと小さく、10点すべてをほぼ通っています。
    9次は重みが10個（切片を含む）あるので、10点ならどんな並びでも通れるのです。

    でも、9次の曲線の**点と点の間**を見てください。測った点の近くでは合っていても、その間で大きく上下しています。
    気温がこれほど急に上下したとは考えにくそうです。

    ### 学習に使わなかったデータで確かめる

    同じ日に別に測った40回を、図7の曲線に重ねます。このように、学習には使わず予測を確かめるデータを**検証データ**と呼びます。
    """)
    return


@app.cell(hide_code=True)
def _(TEMPERATURE_BASE, TRAIN, figure, fit_panel):
    figure("compareFits", {**TEMPERATURE_BASE, "showValid": True, "panels": [
        fit_panel("3次", *TRAIN, 3, True), fit_panel("9次", *TRAIN, 9, True)]},
        "図8　図7の曲線に、学習に使わなかった検証データ40回を重ねた。", height=420)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3次の曲線は、検証データにもおおむね合っています。9次の曲線は、訓練データにはぴったりなのに、検証データからは大きく外れます。
    訓練データの測定誤差（ノイズ）まで覚えてしまい、まだ見ていないデータに通用しなくなった状態を**過学習（overfitting）**と呼びます。

    ### 次数を変えて比べる

    次数1〜9でも比べてみます。左は選んだ次数の曲線、右は各次数の訓練MSE（青）と検証MSE（橙）です。

    左の灰色の曲線は、**同じ回数を測り直したときに**学習される曲線です。測り直すと、時刻と測定誤差が変わります。
    データが少し違うだけで、学習結果がどれくらい変わるかを表しています。
    """)
    return


@app.cell(hide_code=True)
def _(
    GRID_T,
    OTHER_SEEDS,
    TEMPERATURE_BASE,
    TRAIN,
    VALID,
    fit_polynomial,
    measure,
    mse,
    np,
    pairs,
):
    def explore(mode, labels, settings):
        """各設定（次数, λ）で訓練データから学習し、曲線とMSEをまとめる。別の測定から学習した曲線も添える。"""
        t, y = TRAIN
        others = [measure(s, len(t)) for s in OTHER_SEEDS]
        steps = []
        for degree, strength in settings:
            predict, weights = fit_polynomial(t, y, degree, strength)
            steps.append({
                "curve": np.round(predict(GRID_T), 3).tolist(),
                "ghosts": [np.round(fit_polynomial(ot, oy, degree, strength)[0](GRID_T), 2).tolist() for ot, oy in others],
                "train": mse(y, predict(t)), "valid": mse(VALID[1], predict(VALID[0])),
                "weights": np.abs(weights[1:]).tolist(),
            })
        return {**TEMPERATURE_BASE, "mode": mode, "labels": labels, "train": pairs(t, y), "steps": steps}

    DEGREE_DATA = explore("degree", [str(d) for d in range(1, 10)], [(d, 0.0) for d in range(1, 10)])
    LAMBDAS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    RIDGE_DATA = explore("ridge", ["0"] + [f"1e{int(np.log10(v))}" for v in LAMBDAS[1:]], [(9, v) for v in LAMBDAS])
    return DEGREE_DATA, LAMBDAS, RIDGE_DATA


@app.cell(hide_code=True)
def _(DEGREE_DATA, figure):
    figure("fitExplorer", DEGREE_DATA,
           "図9　左：訓練データ（●）から学習した曲線（黒）。灰色は同じ回数を測り直した場合の曲線（時刻と測定誤差が異なる）、点線は本当の気温の変化（実際には分からない）。右：次数ごとの訓練MSEと検証MSE。",
           height=620)
    return


@app.cell(hide_code=True)
def _(DEGREE_DATA, mo, np):
    _valid = [s["valid"] for s in DEGREE_DATA["steps"]]
    _best = int(np.argmin(_valid)) + 1
    mo.md(rf"""
    - **1〜2次**：曲がり方が足りず、訓練データにも検証データにも合いません。これを**過少適合（underfitting）**と呼びます。
    - **{_best}次**：検証MSEが最も小さく（{_valid[_best - 1]:.2f}）、本当の変化（点線）に近い曲線です。
    - **それより高い次数**：訓練MSEは下がり続けますが、検証MSEは上がっていき、9次で大きく悪化します。過学習です。

    灰色の曲線を見ると、次数が高いほど、**測り直して得たデータが少し違うだけで学習される曲線が大きく変わります。** 時刻だけでなく測定誤差も変わっています。
    訓練データのたまたまの揺れに、曲線が振り回されているのです。

    訓練MSEだけでは複雑なモデルを選びがちです。次数のように学習前に決める設定を**ハイパーパラメータ**と呼び、検証MSEで選びます。

    ### データを増やすと

    過学習は、モデルの複雑さに比べてデータが少ないときに起こりやすくなります。
    同じ9次で、訓練データを10回と40回に変えて比べます。
    """)
    return


@app.cell(hide_code=True)
def _(TEMPERATURE_BASE, TRAIN, figure, fit_panel, measure):
    figure("compareFits", {**TEMPERATURE_BASE, "showValid": True, "panels": [
        fit_panel("9次・訓練データ10回", *TRAIN, 9, True), fit_panel("9次・訓練データ40回", *measure(13, 40), 9, True)]},
        "図10　同じ9次の多項式を、訓練データの数だけ変えて学習した。", height=420)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    この例では40回測ると曲線の暴れが収まり、検証MSEも小さくなります。点が増え、測定誤差に合わせて曲がる余地が減ったためです。
    データを増やせないときは、曲がり方を抑える方法もあります。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 正則化：重みを大きくしすぎない

    9次の曲線が暴れているとき、重みはとても大きな値になっています。大きな重み同士が打ち消し合って、ようやく訓練点を通っている状態です。
    そこで、**重みが大きいこと自体に罰を与えます。**

    $$J = \underbrace{L_{\text{訓練}}}_{\text{訓練データへのずれ}} + \underbrace{\lambda \sum_{j=1}^{d} w_j^2}_{\text{重みの大きさへの罰}}$$

    この $J$ を最小にする重みを求める方法を**L2正則化**と呼びます。
    $\lambda$（ラムダ）は罰の強さで、これもハイパーパラメータです。$\lambda=0$ なら普通の最小二乗法です。切片 $b$ は罰の対象にしません。

    9次のまま $\lambda$ を変え、曲線・MSE・重みの大きさを比べます。
    """)
    return


@app.cell(hide_code=True)
def _(RIDGE_DATA, figure):
    figure("fitExplorer", RIDGE_DATA,
           "図11　9次の多項式で、罰の強さ λ だけを変える。下の棒は、標準化した特徴量に掛かる重みの大きさ。",
           height=760)
    return


@app.cell(hide_code=True)
def _(RIDGE_DATA, mo, np):
    _steps = RIDGE_DATA["steps"]
    _best = int(np.argmin([s["valid"] for s in _steps]))
    mo.md(rf"""
    - $\lambda = 0$ では、重みの最大値が約 {max(_steps[0]["weights"]) / 1e4:,.0f} 万になっています。少し罰を加えるだけで重みが小さくなり、曲線が落ち着きます。灰色の曲線のばらつきも小さくなります。
    - 訓練MSEは $\lambda = 0$ で最小です。この例では罰を加えると訓練MSEは上がりますが、検証MSEは下がります（$\lambda = {RIDGE_DATA["labels"][_best].replace("1e", "10^{") + "}" if _best else "0"}$ で最小の {_steps[_best]["valid"]:.2f}）。
    - $\lambda$ を大きくしすぎると、曲がる力まで失い、訓練MSEも検証MSEも悪化します（過少適合）。

    正則化は、訓練データへの当てはまりを少し抑えて、未知のデータへの予測を改善する工夫です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion({"特徴量の尺度を揃える：標準化（任意）": mo.md(r"""
    $x, x^2, \dots, x^9$ は値の大きさがまったく違うので、そのままでは罰が特徴量ごとに不公平に効きます。
    そこで各特徴量から訓練データでの平均を引き、標準偏差で割って尺度を揃えてから学習しています。これを**標準化**と呼びます。
    検証・テストデータにも、訓練データで求めた平均と標準偏差をそのまま使います。図11の重みは、標準化した特徴量に掛かる値です。
    """)})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 最後に、テストデータで一度だけ評価する

    検証データで次数や $\lambda$ を選びました。では、そのモデルが**まだ見ていない測定にも合うか**をどう確かめればよいでしょうか。

    データの一部を最初から別に残し、モデルを選び終わってから初めて使います。これが**テストデータ（test data）**です。データが十分あるときは、全体の1〜2割ほどを残すことがあります。
    """)
    return


@app.cell(hide_code=True)
def _(figure):
    figure("dataRoles", {},
           "図12　訓練で重みを求め、検証でモデルを選び、残しておいたテストデータで最後に確かめる。箱の幅はデータの割合を表さない。",
           height=250)
    return


@app.cell(hide_code=True)
def _(DEGREE_DATA, LAMBDAS, RIDGE_DATA, TEST, TRAIN, VALID, fit_polynomial, mo, mse, np):
    _degree_i = int(np.argmin([s["valid"] for s in DEGREE_DATA["steps"]]))
    _ridge_i = int(np.argmin([s["valid"] for s in RIDGE_DATA["steps"]]))
    _ridge_lambda = LAMBDAS[_ridge_i]
    _ridge_text = "0" if _ridge_lambda == 0 else f"10^{{{int(round(np.log10(_ridge_lambda)))}}}"
    _rows = []
    for _d in range(1, 10):
        for _lam in LAMBDAS:
            _f, _ = fit_polynomial(*TRAIN, _d, _lam)
            _rows.append((mse(VALID[1], _f(VALID[0])), _d, _lam, _f))
    _valid, _d, _lam, _f = min(_rows, key=lambda r: r[0])
    _test = mse(TEST[1], _f(TEST[0]))
    _lam_text = "0" if _lam == 0 else f"10^{{{int(round(np.log10(_lam)))}}}"
    _verdict = (f"この例では、正則化しない{_d}次が最良でした。正則化は候補を改善する方法の一つで、必ず使うわけではありません。" if _lam == 0
                else f"この例では、正則化を加えた{_d}次が最良でした。")
    _gap = ("テストMSEが検証MSEより大きいのは、よくあることです。81通りの中から検証データに一番合うものを選んだので、検証MSEは実力より少しよく見えやすいのです。"
            if _test > _valid else "")
    mo.md(rf"""
    図9で次数1〜9を、図11で9次の $\lambda$ を比べました。最後は、**次数1〜9と $\lambda$ の9通りを組み合わせた81通り**を、同じ検証データで比べます。

    | 候補 | 検証MSE |
    |---|---:|
    | 図9で最良の {_degree_i + 1}次・$\lambda=0$ | {DEGREE_DATA["steps"][_degree_i]["valid"]:.2f} |
    | 図11で最良の9次・$\lambda={_ridge_text}$ | {RIDGE_DATA["steps"][_ridge_i]["valid"]:.2f} |
    | **81通りで最良の {_d}次・$\lambda={_lam_text}$** | **{_valid:.2f}** |

    {_verdict}

    選んだ {_d}次・$\lambda = {_lam_text}$ を、残しておいたテストデータで確かめると、テストMSEは **{_test:.2f}** でした。この値を、今後の測定にどれくらい合いそうかの目安にします。
    {_gap}

    検証データはモデルを**選ぶため**、テストデータは選んだモデルを**最後に確かめるため**に使います。テストの結果を見て選び直したら、そのデータはもう最後の公平な確認には使えません。
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
    quiz_param = mo.ui.radio(["部屋の広さ x", "実際の家賃 y", "傾き w と切片 b"],
                             label="**Q1.** 家賃を $\\hat y = wx + b$ で予測するとき、データから決める（学習する）ものはどれですか？")
    quiz_param
    return (quiz_param,)


@app.cell(hide_code=True)
def _(check, quiz_param):
    check(quiz_param, "傾き w と切片 b",
          "$w$ と $b$ がパラメータです。広さ $x$ は入力、家賃 $y$ は正解で、どちらもデータとして与えられます。"
          "学習は、予測 $\\hat y$ と正解 $y$ のずれ（損失）が小さくなるように、パラメータを決めることです。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_valley = mo.ui.radio(["データ全体のMSEが最も小さくなる w, b の組", "家賃の平均", "MSEがちょうど0になる w, b の組"],
                              label="**Q2.** 横軸を傾き $w$、縦軸を切片 $b$ とし、各点の色で予測のMSEを表す地形を考えます。谷底の点は何を表しますか？")
    quiz_valley
    return (quiz_valley,)


@app.cell(hide_code=True)
def _(check, quiz_valley):
    check(quiz_valley, "データ全体のMSEが最も小さくなる w, b の組",
          "地形の1点が1組の $w, b$（1本の直線）で、色がその直線のMSEです。谷底はMSEが最小の直線で、最小二乗法の解です。"
          "データが一直線上に並んでいなければ、谷底でもMSEは0になりません。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_feature = mo.ui.radio(["言える。MSEが下がったから", "言えない。学習に使っていないデータで確かめる必要がある", "言えない。特徴量を増やすと必ず過学習するから"],
                               label="**Q3.** 特徴量を一つ増やしたら、訓練データのMSEが下がりました。これで「予測がよくなった」と言えますか？")
    quiz_feature
    return (quiz_feature,)


@app.cell(hide_code=True)
def _(check, quiz_feature):
    check(quiz_feature, "言えない。学習に使っていないデータで確かめる必要がある",
          "特徴量を増やすと、役に立たない特徴量でも訓練MSEは下がります（少なくとも上がりません）。"
          "予測がよくなったかは、検証データのMSEで確かめます。"
          "ただし、増やせば必ず過学習するわけでもありません。駅徒歩のように家賃に本当に関係する特徴量なら、まだ見ていない部屋への予測もよくなることが多いです。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_overfit = mo.ui.radio(["訓練MSEが最も小さいので、このモデルを選ぶ", "過学習している可能性が高いので、次数を下げるか正則化を検討する", "テストMSEを見て、よければこのモデルに決める"],
                               label="**Q4.** あるモデルは訓練MSEがほぼ0ですが、検証MSEは他の候補よりずっと大きくなりました。どう判断しますか？")
    quiz_overfit
    return (quiz_overfit,)


@app.cell(hide_code=True)
def _(check, quiz_overfit):
    check(quiz_overfit, "過学習している可能性が高いので、次数を下げるか正則化を検討する",
          "訓練データにだけ合わせすぎ、まだ見ていないデータに通用していない可能性があります。"
          "テストデータは選び終わった後に一度だけ使うもので、候補を比べるためには使いません。")
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_split = mo.ui.radio(["訓練データ", "検証データ", "テストデータ"],
                             label="**Q5.** 多項式の次数や正則化の強さ $\\lambda$ を選ぶときに、MSEを比べるデータはどれですか？")
    quiz_split
    return (quiz_split,)


@app.cell(hide_code=True)
def _(check, quiz_split):
    check(quiz_split, "検証データ",
          "訓練データは重みを求めるため、検証データは次数や $\\lambda$ を選ぶため、テストデータは選び終わった後の最終評価のために使います。"
          "訓練データのMSEで選ぶと、いつも一番複雑なモデルが選ばれてしまいます。")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    - **回帰は数値を予測する。** 予測 $\hat y$ はパラメータで決まり、正解 $y$ とのずれを損失（MSE）で測る。
    - **学習は、損失が最も小さくなるパラメータを探すこと。** 損失の地形の谷底を探す。
    - **訓練データの損失が小さいだけでは不十分。** 検証データでハイパーパラメータを選び、最後にテストデータで一度だけ評価する。

    **この章の確認：** 訓練・検証・テストの役割を区別し、図9で過学習の兆しを指せれば完了です。

    次の03では、坂の向き（勾配）を頼りに損失を下げる方法を学びます。

    ### もっと詳しく読むとき（任意）

    [機械学習帳](https://chokkan.github.io/mlnote/)で、式の導出をじっくり追えます。
    [単回帰](https://chokkan.github.io/mlnote/regression/01sra.html)、[重回帰](https://chokkan.github.io/mlnote/regression/02mra.html)、[モデル選択と正則化](https://chokkan.github.io/mlnote/regression/03regularization.html)が、この章に対応します。
    """)
    return


if __name__ == "__main__":
    app.run()
