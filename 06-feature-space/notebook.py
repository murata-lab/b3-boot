import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="06 特徴空間と次元削減", css_file="notebook.css")


@app.cell(hide_code=True)
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import numpy as np

    return Path, json, mo, np


@app.cell(hide_code=True)
def _(Path, json, mo):
    root = Path(__file__).resolve().parent
    shell = (root / "figure.html").read_text(encoding="utf-8")
    script = (root / "figures.js").read_text(encoding="utf-8")
    sample = json.loads((root / "assets/mnist_maps.json").read_text(encoding="utf-8"))

    def figure(name, data, caption="", height=380):
        config = json.dumps({"name": name, "data": data, "caption": caption}, ensure_ascii=False, allow_nan=False)
        html = shell.replace("/* CONFIG */", "const CONFIG = " + config.replace("</", "<\\/") + ";")
        return mo.iframe(html.replace("/* FIGURES */", script), height=f"{height}px")

    def check(choice, answer, explanation):
        if choice.value is None:
            return mo.md("")
        correct = choice.value == answer
        head = "**正解です。**" if correct else f"**正解は「{answer}」です。**"
        return mo.callout(mo.md(head + " " + explanation), kind="success" if correct else "warn")

    return check, figure, sample


@app.cell(hide_code=True)
def _(mo):
    mo.Html('''<div class="hero"><p class="eyebrow">OPTIONAL · 06</p><h1>特徴空間と次元削減</h1>
    <p>ニューラルネットの中で、手書き数字はどう表されている？</p></div>''')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    05では手書き数字を分類するモデルを学習しました。正解率だけでは、モデルが画像をどんな形で扱うようになったかは分かりません。
    この章ではモデルの途中の値を取り出し、画像どうしの位置関係を見ます。使うのは05と同じ `784 → 128 → 10` のモデルです。

    ## 1. モデルの途中で、画像は何になっている？

    1枚の画像は、最初は28×28個の画素値です。隠れ層で128個の重み付き和を計算し、ReLUを通します。最後の10個の値は、各数字の**スコア（score）**です。
    途中の128個の値を、その画像についてモデルが作った**特徴量（features）**、または**表現（representation）**と呼びます。
    """)
    return


@app.cell(hide_code=True)
def _(figure, sample):
    idx = sample["neighbor_example"]["anchor"]
    figure("pipeline", {"image": sample["images"][idx], "label": sample["labels"][idx],
                        "hidden": sample["example_hidden"], "scores": sample["example_scores"]},
           "図1　1枚の画像が、隠れ層で128個の特徴量に、出力層で10個のスコアに変わる。特徴量は先頭の5個だけ表示。0.0はReLUで負の値が0になったもの。", 200)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    02の「広さ・駅徒歩」は人が選んだ特徴量でした。このモデルは、隠れ層の重みも分類の誤りが小さくなるよう学習するため、特徴量の作り方も変わります。
    ただし、128個の各値に「丸み」「線の傾き」のような名前が付くとは限りません。まずは値の組全体を、一つの位置として考えましょう。

    ## 2. 画像1枚を、空間の1点として見る

    特徴量が2個なら平面の $(x_1,x_2)$、3個なら立体の $(x_1,x_2,x_3)$ に1点を置けます。**次元（dimension）**は、点の位置を表す座標の数です。128個なら128次元で、紙にそのまま描けません。
    モデル内の特徴量が作る空間を**特徴空間（feature space）**、文脈によっては**潜在空間（latent space）**と呼びます。潜在空間は2次元の図だけの名前ではありません。

    2点の近さは、座標ごとの差を2乗して足し、平方根をとった**距離（distance）**で測ります。平面での距離と同じ計算を、784個や128個の座標で行います。
    表し方が違えば「近い画像」も変わるでしょうか。同じ1枚を基準に、画素値で最も近い画像と、学習後の特徴量で最も近い画像を比べます。
    """)
    return


@app.cell(hide_code=True)
def _(figure, sample):
    _e = sample["neighbor_example"]
    figure("neighbors", {"labels": [sample["labels"][_e[k]] for k in ("anchor", "pixel", "feature")],
                         "images": [sample["images"][_e[k]] for k in ("anchor", "pixel", "feature")]},
           "図2　「画素値で近い」は784個の画素値、「特徴量で近い」は学習後の128個の特徴量で、基準の画像との距離が最も小さい画像。この章で使う600枚の中から選んだ。", 230)
    return


@app.cell(hide_code=True)
def _(mo, sample):
    _e = sample["neighbor_example"]
    mo.md(rf"""
    二つの近さで選ぶ画像が食い違う例を選びました。基準の数字は **{sample['labels'][_e['anchor']]}** です。画素値で近い画像の正解は **{sample['labels'][_e['pixel']]}**、学習後の特徴量で近い画像の正解は **{sample['labels'][_e['feature']]}** でした。
    画素値の距離は、同じ位置に白い線がどれだけ重なるかで決まります。そのため、数字が違っても線の多くが重なる画像が近くなることがあります。600枚全体での傾向は6節で確かめます。

    では、128次元にある多くの点をどうすれば目で見られるでしょうか。

    ## 3. 128次元を紙の上へ — PCA

    **次元削減（dimensionality reduction）**は、多数の座標を少数の座標へ置き換えることです。まず、2次元の点を1本の軸に写して1次元にする場面を見ます（説明用に作った点群です）。

    下の図では、横軸と縦軸が元の2個の特徴量で、青い点が1件のデータです。橙の直線が写す軸で、各点から軸へ垂直に下ろした位置（○）が写した後の位置です。その位置だけを1本の目盛りに並べ直したものが「軸に写した位置」です。
    写した位置が広く散らばっていれば、元の点どうしの違いがよく残っています。逆に、写した位置が詰まると、違う点どうしが見分けにくくなります。
    """)
    return


@app.cell(hide_code=True)
def _(np):
    rng = np.random.default_rng(11)
    t = np.linspace(-2.5, 2.5, 25)
    points = np.column_stack([t + rng.normal(0, .12, len(t)), .55 * t + rng.normal(0, .27, len(t))])
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    angle = float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180)
    directions = [0., 90., angle]
    views = []
    total_var = float(np.sum(np.var(centered, axis=0)))
    for a in directions:
        vector = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
        projected = centered @ vector
        views.append({"angle": round(a, 1), "values": np.round(projected, 3).tolist(),
                      "share": round(float(np.var(projected) / total_var), 3)})
    pca_demo = {"points": np.round(centered, 3).tolist(), "views": views}
    return (pca_demo,)


@app.cell(hide_code=True)
def _(figure, pca_demo):
    figure("pcaDemo", pca_demo,
           "図3　写す軸の向きを切り替えると、写した位置の散らばり方が変わる。「軸に写した位置」の○は、重ならないよう上に積んでいる。「残るばらつき」は、元の2次元のばらつきのうち、この軸に写した後に残る割合。", 420)
    return


@app.cell(hide_code=True)
def _(mo, pca_demo, sample):
    mo.md(rf"""
    この点群では、残るばらつきは横方向で **{pca_demo['views'][0]['share']:.0%}**、縦方向で **{pca_demo['views'][1]['share']:.0%}**、点群が伸びる斜めの方向で **{pca_demo['views'][2]['share']:.0%}** です。
    **主成分分析（PCA: Principal Component Analysis）**は、このように写した点のばらつきが最も大きい方向を選びます。2次元へ写すときは、1本目と直角な方向のうち、次にばらつきが大きい方向を2本目にします。選んだ方向を順に**第1主成分**、**第2主成分**と呼びます。それ以外の方向の情報は失われます。

    05のモデルが作った128個の特徴量もPCAで2次元にしてみましょう。横軸・縦軸は第1・第2主成分です。各点は1枚の画像で、色は正解の数字を示します。
    """)
    return


@app.cell(hide_code=True)
def _(figure, sample):
    figure("scatter", {"points": sample["pca"], "labels": sample["labels"], "images": sample["images"], "title": "学習後の特徴量をPCAで2次元へ",
                       "xname": "第1主成分", "yname": "第2主成分"},
           "図4　学習後の128個の特徴量をPCAで2次元にした図。点に触れると元画像と正解が出る。点を選ぶと、その画像が図の下に残る。", 500)
    return


@app.cell(hide_code=True)
def _(mo, sample):
    mo.md(rf"""
    0や1はある程度まとまっていますが、4・7・9や、3・5・8は大きく重なっています。
    第1・第2主成分に残るばらつきは、合計 **{sum(sample['pca_variance']):.1%}** しかありません。図で重なっても、残り126方向では離れているかもしれません。
    PCAはまっすぐな軸へ写す方法です。点の並びが曲がっていたら、曲がりに沿った位置関係をどう扱えばよいでしょうか。

    ## 4. 曲がったデータを広げる — Isomap

    細長い紙を折り返した形を、横から見た模式図を使います。両端は図の上では近くても、紙に沿って端から端へ進む距離は長くなります。
    下図は紙の中心線だけを点で描いています。「近傍 2点」は、各点を最も近い2個の点と結ぶという意味です。黒い線は、その接続をたどって始点から終点へ進む最短経路です。
    図の下段は、この経路の距離をもとに、点を1本の軸へ並べ直した結果です。
    """)
    return


@app.cell(hide_code=True)
def _(np):
    # 折り返した細い紙の中心線。隣接点の間隔より上下の隙間を広くしておく。
    top = np.column_stack([np.linspace(-4, 4, 19), np.full(19, .5)])
    theta = np.linspace(0, np.pi, 7)[1:-1]
    turn = np.column_stack([4 + .5 * np.sin(theta), .5 * np.cos(theta)])
    bottom = np.column_stack([np.linspace(4, -4, 19), np.full(19, -.5)])
    line = np.concatenate([top, turn, bottom])
    dist = np.linalg.norm(line[:, None, :] - line[None, :, :], axis=2)
    np.fill_diagonal(dist, np.inf)

    def graph_and_layout(k):
        neighbors = np.argsort(dist, axis=1)[:, :k]
        edges = sorted({tuple(sorted((i, int(j)))) for i, row in enumerate(neighbors) for j in row})
        linked = {i: set() for i in range(len(line))}
        geo = np.full(dist.shape, np.inf)
        np.fill_diagonal(geo, 0.)
        for i, j in edges:
            linked[i].add(j)
            linked[j].add(i)
            geo[i, j] = geo[j, i] = dist[i, j]
        # ② すべての点の組について、接続をたどる最短経路の長さを求める。
        for m in range(len(line)):
            geo = np.minimum(geo, geo[:, [m]] + geo[[m], :])
        path = [len(line) - 1]
        while path[-1] != 0:
            here = path[-1]
            path.append(min(linked[here], key=lambda j: abs(geo[0, j] + dist[j, here] - geo[0, here])))
        # ③ 経路の距離をなるべく保つように、点を1本の軸へ並べる（Isomapの最後の段階）。
        center = np.eye(len(line)) - 1 / len(line)
        values, vectors = np.linalg.eigh(-.5 * center @ geo ** 2 @ center)
        layout = vectors[:, -1] * np.sqrt(values[-1])
        if layout[0] > layout[-1]:
            layout = -layout
        return {"k": k, "edges": edges, "path": path[::-1], "distance": round(float(geo[0, -1]), 2),
                "layout": np.round(layout, 3).tolist()}

    isomap_demo = {"points": np.round(line, 3).tolist(), "views": [graph_and_layout(2), graph_and_layout(5)],
                   "direct": round(float(dist[0, -1]), 2)}
    return (isomap_demo,)


@app.cell(hide_code=True)
def _(figure, isomap_demo):
    figure("isomap", isomap_demo,
           "図5　折り返した紙の中心線（説明用の模式図）。上段は近傍の接続と、始点から終点への最短経路。下段は、経路の距離をもとに点を1本の軸へ並べた結果。距離はこの図の座標の単位。", 470)
    return


@app.cell(hide_code=True)
def _(isomap_demo, mo):
    mo.md(rf"""
    **Isomap**は、①各点を近くの点と結ぶ、②結んだ線をたどる最短経路の長さで、紙に沿った距離を近似する、③その距離をなるべく保つように低次元へ点を置く、という方法です。
    図の点は折れ曲がった線の上にありますが、線上の位置は「始点から線に沿ってどれだけ進んだか」という1個の座標で表せます。紙全体なら、縦・横の2個の座標で足ります。このように、高次元の空間の中にあっても、各点の周りでは少数の座標で位置を表せる形を**多様体（manifold）**と呼びます。

    近傍2点では、下段で紙が端から端までまっすぐ広がります。近傍5点にすると、折り返しの反対側の点も近傍に入り、始点から終点への経路は **{isomap_demo['views'][0]['distance']:.2f} → {isomap_demo['views'][1]['distance']:.2f}** に縮みます。その結果、下段では紙の上側と下側が同じ場所に重なります。近傍の選び方を誤ると、曲がった形を正しく広げられません。

    近所とのつながりを使いながら、別の方法で配置を作るのがUMAPです。

    ## 5. 近所のつながりを写す — UMAP

    **UMAP（Uniform Manifold Approximation and Projection）**も元の空間で近い点を探します。ただし、接続するかどうかの二択ではなく、周囲の密集具合を考慮してつながりに強さを付けます。
    下図では線が太いほどつながりが強いことを表します。2次元に仮置きした点を動かし、強いつながりを近くに残しながら、点が一か所に潰れないようにも調整します。
    """)
    return


@app.cell(hide_code=True)
def _(figure):
    figure("umapSteps", {}, "図6　UMAPの模式図。近傍のつながりを強さ付きで作り、低次元側の配置を更新する。", 360)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    UMAPは、元のつながりと2次元のつながりの食い違いを小さくするよう、点の位置を少しずつ更新します。03・04で見た「損失を小さくする更新」と同じ考え方です。
    Isomapは経路から得た点同士の距離を手掛かりにし、UMAPは強さ付きの近傍のつながりを手掛かりにします。どちらも近所の選び方で結果が変わります。

    ## 6. 学習すると、特徴量はどう変わる？

    同じ600枚の画像を、画素値、初期重み（学習前）のモデルが作る128個の特徴量、学習後の128個の特徴量で表し、それぞれUMAPで2次元にしました。3枚の図は別々に計算したので、回転や反転、島の並び順が変わります。図を切り替えたときの座標そのものは比べられません。
    正解ラベルはモデルの学習には使いましたが、今回のUMAPには渡していません。図の色付けと画像の確認にだけ使っています。

    2次元の図だけに頼らないよう、図の上に元の空間で測った値も示します。各画像について、距離が最も小さい10枚のうち、同じ数字の画像が占める割合の平均です。
    """)
    return


@app.cell(hide_code=True)
def _(figure, sample):
    figure("compare", {"maps": {k: sample["maps"][k] for k in ("pixels", "before", "after")},
                       "labels": sample["labels"], "images": sample["images"],
                       "agreement": sample["neighbor_agreement"]},
           "図7　同じ600枚を、三つの表現から別々にUMAPで2次元にした図。点を選ぶと、その手書き画像を図の下に表示する。", 560)
    return


@app.cell(hide_code=True)
def _(mo, sample):
    _a = sample["neighbor_agreement"]
    mo.md(rf"""
    近い10枚のうち同じ数字の割合は、画素値で **{_a['pixels']:.0%}**、学習前の特徴量で **{_a['before']:.0%}**、学習後の特徴量で **{_a['after']:.0%}** でした。乱数で決めた初期重みの特徴量は、画素値より同じ数字どうしを近づけてはいません。学習後は同じ数字の画像が互いに近くなり、2次元の図でも数字ごとの島がはっきりします。
    ただし、混ざりや離れた点も残ります。学習で直接「同じ数字を一つの島へ集める」よう指定したわけではなく、最後の層が数字を分けやすい表現を作るよう重みを更新した結果です。

    この600枚に対するモデルの正解率は **{sample['sample_accuracy']:.1%}** です。この数値は元の128次元の特徴量から出した予測についてのものです。2次元の図の見た目だけで分類性能は判断できません。

    同じ特徴量でも、UMAPの設定を変えると図はどうなるでしょうか。**近傍数（n_neighbors）**は、つながりを作るときに見る近所の点の数です。**min_dist**は、2次元で点どうしをどこまで詰めてよいかの下限で、大きいほど点がゆったり広がります。次の二つは、入力の特徴量が同じで、この二つの設定だけが違います。
    """)
    return


@app.cell(hide_code=True)
def _(figure, sample):
    figure("settings", {"left": sample["maps"]["after"], "right": sample["maps"]["after_wide"],
                        "settings": [sample["umap_settings"]["default"], sample["umap_settings"]["wide"]],
                        "labels": sample["labels"], "images": sample["images"]},
           "図8　学習後の同じ特徴量を、見出しの二つの設定でUMAPにかけた図。点を選ぶと、同じ画像が両方の図で大きく表示される。ラベルは配置に使っていない。", 650)
    return


@app.cell(hide_code=True)
def _(mo, sample):
    _d, _w = sample["umap_settings"]["default"], sample["umap_settings"]["wide"]
    mo.md(rf"""
    近傍数{_d['n_neighbors']}・min_dist {_d['min_dist']}では島がはっきり分かれ、近傍数{_w['n_neighbors']}・min_dist {_w['min_dist']}では島がふくらんで互いに接しています。点を一つ選ぶと、同じ画像が二つの図で別の場所にあることも分かります。
    このように、UMAPの島どうしの距離や島の大きさは設定で変わります。そこから数字の「本当の類似度」や128次元でのばらつきをそのまま読み取ることはできません。確かめたいことは、元画像と元の特徴量へ戻って調べます。

    ## 確認問題

    選ぶとすぐに解説が出ます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    q1 = mo.ui.radio(["学習によって、この画像の特徴量が左下から右上へ移動した", "別々に計算した配置なので、座標の違いは学習による移動を表さない", "学習前は、この画像が誤って分類されていた"],
                     label="**Q1.** 同じ600枚について、学習前と学習後の特徴量を別々にUMAPで2次元にしました。ある画像の点が、学習前の図では左下、学習後の図では右上にありました。どう解釈しますか？")
    q1
    return (q1,)


@app.cell(hide_code=True)
def _(check, q1):
    check(q1, "別々に計算した配置なので、座標の違いは学習による移動を表さない", "UMAPは表現ごとに独立に配置を作るので、回転・反転や島の並び順が変わります。学習による変化は、同じ数字どうしの近さのように、元の空間で測れる量で比べます。")
    return


@app.cell(hide_code=True)
def _(mo):
    q2 = mo.ui.radio(["128次元でも必ず近い", "第1・第2主成分は分類に最も役立つ方向なので、モデルもこの2枚を区別できない", "残りの126方向で離れている可能性がある"],
                     label="**Q2.** 128次元の特徴量をPCAで2次元に写すと、異なる数字の2点が重なりました。元の128次元について、どの解釈が妥当ですか？")
    q2
    return (q2,)


@app.cell(hide_code=True)
def _(check, q2):
    check(q2, "残りの126方向で離れている可能性がある", "2次元へ写すと、残りの方向の情報は失われます。また、PCAはラベルを使わずにばらつきが大きい方向を選ぶので、分類に役立つ方向とは限りません。")
    return


@app.cell(hide_code=True)
def _(mo):
    q3 = mo.ui.radio(["折り返しの反対側へ近道ができ、紙に沿った距離より短く見積もっている", "近傍が増えたので、紙に沿った距離がより正確に測れるようになった", "③の配置の計算が崩れたので、経路距離も短くなった"],
                     label="**Q3.** 折り返した紙の上に並んだ点にIsomapを使います。近傍数を増やしたら、始点から終点までの経路距離が急に短くなりました。どう解釈しますか？")
    q3
    return (q3,)


@app.cell(hide_code=True)
def _(check, q3):
    check(q3, "折り返しの反対側へ近道ができ、紙に沿った距離より短く見積もっている", "紙に沿った距離は変わっていません。反対側の点と結ばれ、最短経路が紙を横切る近道を通るようになりました。経路距離は②で決まり、③の配置はその距離から後で作ります。")
    return


@app.cell(hide_code=True)
def _(mo):
    q4 = mo.ui.radio(["すべての点の組の経路距離を正確に保つ", "近傍のつながりの強さを2次元でも再現する", "正解ラベルごとに一つの島を指定する"],
                     label="**Q4.** 高次元の点をUMAPで2次元に配置するとき、何を手掛かりに点を動かしますか？")
    q4
    return (q4,)


@app.cell(hide_code=True)
def _(check, q4):
    check(q4, "近傍のつながりの強さを2次元でも再現する", "元の空間の近傍関係に強さを付け、低次元でも近い点のつながりを再現するよう調整します。正解ラベルは渡していません。")
    return


@app.cell(hide_code=True)
def _(mo):
    q5 = mo.ui.radio(["島の距離をそのまま数字の類似度と読む", "島の分かれ具合だけから分類性能を見積もる", "元画像や128次元の特徴量、予測結果に戻って確かめる"],
                     label="**Q5.** 手書き数字の128次元の特徴量をUMAPで2次元に配置すると、数字ごとの島が見えました。ある2点の似方や元のモデルの分類性能を確かめるには、何を調べますか？")
    q5
    return (q5,)


@app.cell(hide_code=True)
def _(check, q5):
    check(q5, "元画像や128次元の特徴量、予測結果に戻って確かめる", "2次元配置は情報を失い、設定でも変わります。分類性能は元のモデルの予測と正解を比べて測ります。")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    - 隠れ層の128個の値は、分類に使うためモデルが作った表現です。
    - PCAはばらつきの大きい直線方向、Isomapは近傍をたどる距離、UMAPは近傍のつながりを手掛かりに次元を減らします。
    - 学習後の特徴量では、同じ数字の画像が元の空間でも近くなりました。ただし、別々に作った2次元の図の座標どうしは比べられません。
    - 2次元の図は高次元の一面です。重なりや島の距離は、元画像・元の特徴量・予測と一緒に解釈します。

    参考：[scikit-learnのIsomap](https://scikit-learn.org/stable/modules/manifold.html#isomap) · [UMAPの仕組み](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) · [UMAPの設定](https://umap-learn.readthedocs.io/en/latest/parameters.html)
    """)
    return


if __name__ == "__main__":
    app.run()
