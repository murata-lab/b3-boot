# /// script
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# auto_reload = "lazy"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="05 PyTorchによる学習と評価",
    css_file="notebook.css",
)


@app.cell(hide_code=True)
def _():
    import json as _json
    import math as _math
    import sys as _sys
    from pathlib import Path as _Path

    import marimo as mo

    _chapter = _Path(__file__).resolve().parent
    if str(_chapter) not in _sys.path:
        _sys.path.insert(0, str(_chapter))
    from lesson_ui import file_preview as _file_preview

    _shell = (_chapter / "figure.html").read_text(encoding="utf-8")
    _script = (_chapter / "figures.js").read_text(encoding="utf-8")

    def figure(name, data, caption="", height=380):
        """figures.js の図を一つ埋め込む。高さは表示後に内容へ合わせて自動で変わる。"""
        config = _json.dumps({"name": name, "data": data, "caption": caption}, ensure_ascii=False, allow_nan=False)
        html = _shell.replace("/* CONFIG */", "const CONFIG = " + config.replace("</", "<\\/") + ";")
        return mo.iframe(html.replace("/* FIGURES */", _script), height=f"{height}px")

    def curves(history, caption):
        """学習曲線の図。損失が nan や inf になったとき（学習率が大きすぎるときなど）は、図の代わりに理由を示す。"""
        values = [r[split][key] for r in history for split in ("train", "validation") for key in ("loss", "accuracy")]
        if not all(_math.isfinite(v) for v in values):
            return mo.callout(mo.md("損失が大きくなりすぎて数値でなくなった（`nan` や `inf`）ため、グラフを描けません。"
                                    "学習率が大きすぎると起こります。"), kind="warn")
        return figure("curves", {"history": history}, caption, height=330)

    def pixels_hex(image):
        """28×28の画像（0〜255の整数、または0〜1の小数）を、図に渡す16進数の文字列にする。"""
        if image.is_floating_point():
            image = (image * 255).round()
        return bytes(image.reshape(-1).byte().tolist()).hex()

    # model.py の本文と、図で指す行番号（「self.hidden = 」などが書かれた行）
    _model_path = _chapter / "model.py"
    _model_lines = _model_path.read_text(encoding="utf-8").splitlines()
    MODEL_LINE = {name: next(i for i, line in enumerate(_model_lines, 1) if f"self.{name} = " in line)
                  for name in ["flatten", "hidden", "relu", "output"]}
    model_source = mo.Html(_file_preview(_model_path))
    return MODEL_LINE, curves, figure, mo, model_source, pixels_hex


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <header class="hero">
      <p class="eyebrow">機械学習入門</p>
      <h1>PyTorchによる学習と評価</h1>
      <p>学習と検証の結果で設定を比べ、選んだモデルを保存して最終評価します。</p>
    </header>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    04では、小さなネットワークの予測と学習を手で計算しました。手書き数字の画像を扱うには、もっと多くの重みが必要です。
    そこで**PyTorch**を使い、画像の形が各層でどう変わるか、どの行で重みを更新するかをコードで追います。6節まで読んだら、5節の設定を一つ変えて比較し、7・8節へ進みます。

    ### この画面について

    セルは最初に一度実行されています。書き換えて実行すると、下のセルは**実行待ち**になります。色が薄くなり、右上の ▶ が黄色になったセルを上から順に実行してください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 手書き数字のデータ

    今回は **MNIST** という、手書き数字の画像のデータを使います。
    1件のデータは、白黒の画像と、「この画像は7」のような**正解ラベル**の組です。
    画像を入力して、0〜9のどの数字かを当てるモデルを作ります。

    下のセルでデータを読み込みます。`train=True` は訓練用の画像を選び、`download=True` は手元になければダウンロードします。
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Subset, TensorDataset
    from torchvision.datasets import MNIST

    torch.set_num_threads(2)  # 計算に使うCPUの数。学習の設定ではない。

    data_dir = Path("data")
    # 公式の訓練用画像を取得する。取得済みなら、保存してあるものを使う。
    mnist = MNIST(root=data_dir, train=True, download=True)
    print("画像の枚数:", len(mnist))
    print("保存先:", data_dir.resolve())
    return (
        DataLoader,
        MNIST,
        Path,
        Subset,
        TensorDataset,
        data_dir,
        mnist,
        nn,
        torch,
    )


@app.cell(hide_code=True)
def _(mnist, mo):
    mo.md(f"""
    {len(mnist):,}枚の画像が用意できました。どんな画像なのか、先頭の16枚を見てみます。
    """)
    return


@app.cell(hide_code=True)
def _(figure, mnist, pixels_hex):
    figure("digits", {"images": [{"pixels": pixels_hex(mnist.data[i]), "label": int(mnist.targets[i])} for i in range(16)]},
           "図1　MNISTの訓練用画像の先頭16枚と、それぞれに付いている正解ラベル。", height=260)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    同じ数字でも、太さや傾き、形がさまざまです。このばらつきがあっても当てられるモデルを作ります。

    ### 画像は、数の表として渡す

    モデルに渡せるのは数だけです。1枚の画像は、縦28×横28のマス（**画素**）に分かれていて、各画素は明るさを0（黒）〜255（白）の整数で持っています。
    先頭の画像の一部を拡大して、値を読んでみます。
    """)
    return


@app.cell(hide_code=True)
def _(figure, mnist, pixels_hex):
    # 値が0でも255でもない画素（字の縁）を最も多く含む7×7の範囲を選んで、値を見せる。
    _img = mnist.data[0]
    _mid = ((_img > 0) & (_img < 255)).int()
    _best = max(((int(_mid[r:r + 7, c:c + 7].sum()), r, c) for r in range(22) for c in range(22)))
    figure("pixels", {"pixels": pixels_hex(_img), "box": {"row": _best[1], "col": _best[2], "size": 7}},
           "図2　左は先頭の画像。右は、左の橙の枠の中の49画素の値。字の部分は255に近く、背景は0。")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    PyTorchでは、このような数の並びを **Tensor（テンソル、tensor）** という配列で扱います。01で学んだNumPyの配列と同じように、`shape`（形）を持っています。

    下のセルでは、全部の画像を一つのTensorにまとめます。

    - 画素の値を255で割り、0〜1の小数に揃えます。画素値の既知の範囲を使う変換で、02のように訓練データの平均から求めるものではありません。学習時も、新しい画像を使うときも同じ変換をします。
    - `unsqueeze(1)` で1画素あたりの値の個数を表す**チャンネル**の軸を加え、形を `[枚数, 1, 28, 28]` にします。白黒画像は明るさを1個の数で表すので、チャンネル数は1です。
    - 正解ラベルは、0〜9の整数のままにします（`long()` は整数型にする指定です）。
    """)
    return


@app.cell
def _(mnist):
    images = mnist.data.unsqueeze(1).float() / 255.0  # [枚数, 1, 28, 28]、0〜1の小数
    labels = mnist.targets.long()  # [枚数]、0〜9の整数
    print("画像:", tuple(images.shape))
    print("正解:", tuple(labels.shape))
    print("画素の値の範囲:", images.min().item(), "〜", images.max().item())
    return images, labels


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. モデルを読み込む

    モデルが受け取るのは、1枚あたり 28×28 = 784個の数です。返してほしいのは、0〜9のどれか、つまり10通りの答えです。
    そこで、03の多クラス分類と同じく、**10個の数字それぞれのスコア**を出させ、スコアが最も大きい数字を予測とします。

    04と同じく、隠れ層の重み付き和をReLUに通し、次の層で組み合わせます。入力は784個です。04の出力はsigmoidに通した二値の確率でしたが、ここでは10クラスのスコアを出し、softmaxと損失の計算は損失関数側で行います。
    データの形は、ネットワークの中で次のように変わります。
    """)
    return


@app.cell(hide_code=True)
def _(MODEL_LINE, figure, model):
    _hidden = model.hidden.out_features
    figure("shapes", {
        "nodes": [{"name": "画像", "shape": "[B, 1, 28, 28]"}, {"name": "784個の画素値", "shape": "[B, 784]"},
                  {"name": f"隠れ層の{_hidden}個の値", "shape": f"[B, {_hidden}]"}, {"name": "10個のスコア", "shape": "[B, 10]"}],
        "ops": [{"code": "self.flatten", "note": "一列に並べる", "line": MODEL_LINE["flatten"]},
                {"code": "self.hidden → self.relu", "note": "重み付き和 → ReLU", "line": f'{MODEL_LINE["hidden"]}・{MODEL_LINE["relu"]}'},
                {"code": "self.output", "note": "重み付き和", "line": MODEL_LINE["output"]}],
    }, "図3　model.py のネットワークの中で、データの形（shape）が変わる様子。$B$ は一度に渡す画像の枚数。行番号は下の model.py の行。", height=200)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    このネットワークは、教材フォルダの `05-pytorch/model.py` というファイルに書いてあります。
    モデルの設計を別のファイルに分けておくと、学習の手順（このページ）とは独立に読めて、ほかの実験でも同じモデルを使えます。
    まず、ファイルの全体を眺めてください。
    """)
    return


@app.cell(hide_code=True)
def _(model_source):
    model_source
    return


@app.cell(hide_code=True)
def _(MODEL_LINE, mo, model):
    _hidden = model.hidden.out_features
    mo.md(rf"""
    `class MLP(...)` は、モデルの作り方をまとめた**クラス（class）**の定義です。中に二つの関数があります。

    - `__init__`：モデルを作るときに一度だけ呼ばれ、使う層を用意します。{MODEL_LINE["hidden"]}行目の `nn.Linear(784, {_hidden})` は、784個の入力から{_hidden}個の重み付き和を計算する層で、重みとバイアスの初期値もここで作られます。
    - `forward`：入力 `x` を、どの順に層へ通すかを書きます。図3の矢印の順です。

    `self` は「作られたモデル自身」を指し、`self.hidden` はそのモデルが持つ層です。
    `class MLP(nn.Module)` と書くと、PyTorchの `nn.Module` の機能を受け継ぎます。
    そのおかげで、すべての層の重みをまとめて取り出したり、ファイルに保存したりできます。

    出力にsoftmaxを付けていないのは、後で使う損失関数が内部でsoftmaxを計算するからです。

    別のファイルのクラスは、`import` で読み込みます。`from model import MLP` は「`model.py` から `MLP` を読み込む」という意味で、`.py` は付けません。
    `MLP()` と括弧を付けて呼ぶと、モデルが一つ作られます。この作られたものを**インスタンス（instance）**と呼びます。
    """)
    return


@app.cell
def _(torch):
    from model import MLP

    torch.manual_seed(42)  # 重みの初期値は乱数で決まる。乱数の出発点を固定して、毎回同じ初期値にする。
    model = MLP()  # 層と重みの初期値を用意する。まだ何も学習していない。
    print(model)
    return MLP, model


@app.cell(hide_code=True)
def _(mo, model):
    _count = sum(p.numel() for p in model.parameters())
    _hidden = model.hidden.out_features
    mo.md(rf"""
    `print(model)` で、`__init__` で用意した4つの層が表示されました。
    `Linear(in_features=784, out_features={_hidden}, bias=True)` は、784個の入力から{_hidden}個の出力を作る、バイアス付きの層です。
    重みとバイアスは全部で **{_count:,}個** あり、今はでたらめな初期値です。これを学習で決めていきます。

    ## 3. データを、訓練・検証・テストに分ける

    02で見たように、学習に使ったデータでよく当たっても、初めて見るデータで当たるとは限りません。
    そこで、学習に使わないデータを分けて成績を測ります。
    """)
    return


@app.cell(hide_code=True)
def _(figure):
    figure("split", {"bars": [
        {"title": "公式の訓練用データ 60,000枚", "parts": [{"kind": "valid", "name": "検証", "count": 5000},
                                                 {"kind": "train", "name": "訓練", "count": 40000},
                                                 {"kind": "unused", "name": "使わない", "count": 15000}]},
        {"title": "公式のテスト用データ 10,000枚", "parts": [{"kind": "test", "name": "テスト", "count": 10000}]},
    ]}, "図4　初期設定でのデータの分け方。訓練枚数を変えると、訓練と使わない部分の枚数も変わる。テスト用の10,000枚は最後の評価まで使わない。", height=200)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | 名前 | 使いみち | 重み |
    |---|---|---|
    | **訓練データ（training data）** | 予測と正解を比べて、重みを更新する | 変える |
    | **検証データ（validation data）** | 学習の途中で成績を測り、学習の回数や学習率などの設定を比べる | 変えない |
    | **テストデータ（test data）** | 設定をすべて決めた後に一度だけ、最終的な成績を測る | 変えない |

    検証データで設定を選ぶと、そこにたまたま合う設定を選ぶ可能性があります。最終的な成績は、設定選びにも使わないテストデータで測ります。

    MNISTは初めから、訓練用60,000枚とテスト用10,000枚に分けて配られています。
    初期設定では、訓練用のうち15,000枚は学習の時間を短くするために使いません。訓練枚数を変える場合、この枚数も変わります。

    下のセルでは、0〜59,999の番号を `randperm` でランダムに並べ替え、先頭の5,000個を検証、次の40,000個を訓練に使います。これは初期設定での分割を試すセルです。5節の `train_mnist` を変えた場合は、その関数の中で改めて分けます。`[5_000:45_000]` のような**スライス**は、開始位置を含み、終了位置を含みません。
    重ならない範囲を取るので、同じ画像が両方に入ることはありません。
    乱数の出発点（seed）を42に固定しているので、何度実行しても同じ分け方になります。
    """)
    return


@app.cell
def _(Subset, TensorDataset, images, labels, torch):
    dataset = TensorDataset(images, labels)  # 画像と正解を、同じ番号で取り出せるようにまとめる。

    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))
    validation_data = Subset(dataset, order[:5_000].tolist())  # 選んだ番号のデータだけを取り出す。
    train_data = Subset(dataset, order[5_000:45_000].tolist())

    print("訓練:", len(train_data), "枚")
    print("検証:", len(validation_data), "枚")
    print("両方に入った画像:", len(set(order[:5_000].tolist()) & set(order[5_000:45_000].tolist())), "枚")
    return train_data, validation_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. 学習の部品を、一つずつ確かめる

    学習では、03・04と同じく「予測 → 損失 → 勾配 → 更新」を繰り返します。
    この節では、その部品をPyTorchで一つずつ動かします。5節で、これらをつなげて学習します。

    ### 128枚ずつ渡す：DataLoader

    03で学んだミニバッチのように、訓練データを少しずつ（今回は128枚ずつ）モデルに渡して、そのたびに重みを更新します。
    データをこの大きさに分けて順に取り出す道具が `DataLoader` です。
    訓練では `shuffle=True` にして、エポックごとに並び順を変えます。

    下のセルでは、最初の128枚を取り出して、学習前のモデルに渡してみます。`iter(train_loader)` で順に取り出す道具を作り、`next(...)` で最初の1組を受け取ります。
    """)
    return


@app.cell
def _(DataLoader, model, torch, train_data, validation_data):
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=512)  # 検証は並び順を変えない。

    batch_images, batch_labels = next(iter(train_loader))  # 最初の128枚を取り出す。
    with torch.no_grad():  # 予測を見るだけなので、勾配の計算に使う記録をしない。
        batch_scores = model(batch_images)
    print("入力の画像:", tuple(batch_images.shape))
    print("正解:", tuple(batch_labels.shape))
    print("モデルの出力:", tuple(batch_scores.shape))
    return train_loader, validation_loader


@app.cell(hide_code=True)
def _(mo, train_data, train_loader):
    _last = len(train_data) - 128 * (len(train_loader) - 1)
    mo.md(rf"""
    出力の形 `(128, 10)` は、「128枚それぞれについて、10個のスコアが出た」という意味です。図3の形のとおりです。
    訓練データ{len(train_data):,}枚を128枚ずつに分けると{len(train_loader)}組になります（最後の組だけ{_last}枚）。
    つまり、1エポックで{len(train_loader)}回、重みを更新します。

    `torch.no_grad()` は、この中の計算では勾配を求めない、という指定です。予測を見るだけのときに使います。

    ### 予測を採点する：損失関数

    **損失関数（loss function）**には、03で学んだ、softmaxと交差エントロピーを組み合わせた `nn.CrossEntropyLoss()` を使います。
    モデルのスコアと、正解の整数を渡すと、各画像の損失 $-\log p_{{\text{{正解}}}}$ の平均を返します。
    まず4枚で試します。
    """)
    return


@app.cell
def _(images, labels, model, nn, torch):
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        scores = model(images[:4])  # 4枚の画像のスコア。形は [4, 10]
        first_loss = loss_fn(scores, labels[:4])
    print("予測:", scores.argmax(dim=1).tolist())  # 各画像で、スコアが最大の数字
    print("正解:", labels[:4].tolist())
    print("損失:", round(first_loss.item(), 3))
    return first_loss, loss_fn


@app.cell(hide_code=True)
def _(first_loss, mo):
    mo.md(rf"""
    `argmax(dim=1)` は、各画像の10個のスコアから、最も大きいものの位置（＝予測した数字）を取り出します。
    学習前なので、予測はほとんど当たりません。

    損失は {first_loss.item():.2f} でした。学習前のモデルは、どの数字にもほぼ同じ確率 1/10 を付けるので、
    損失は $-\log(1/10) \approx 2.30$ に近くなります。学習が進むと、正解の数字の確率が上がり、損失は0に近づきます。

    ### 重みを1回更新する：optimizer

    重みを更新する道具を **optimizer** と呼びます。
    今回は、03で学んだ勾配降下法そのものである `torch.optim.SGD` を使います。
    作るときに、更新する重み（`model.parameters()`）と学習率 `lr` を渡します。

    1回の更新は、次の4行で書きます。

    1. `optimizer.zero_grad()`：前回の勾配を消す。PyTorchは勾配を足し合わせて記録するので、毎回消す必要がある。
    2. `loss = loss_fn(model(x), y)`：予測して、損失を計算する。
    3. `loss.backward()`：逆伝播で、すべての重みの勾配を計算する。**まだ重みは変わらない。**
    4. `optimizer.step()`：勾配を使って重みを更新する。

    下のセルでは、新しいモデルを作って4枚で1回だけ更新し、03の更新式 $\theta \leftarrow \theta - \eta\,\nabla L$ のとおりに重みが変わったかを確かめます。
    `images[:4]` は先頭から4枚を取り出すスライスです。開始位置を省くと0から始まります。
    """)
    return


@app.cell
def _(MLP, images, labels, loss_fn, torch):
    _model = MLP()  # 確認用のモデル
    _optimizer = torch.optim.SGD(_model.parameters(), lr=0.3)
    _x, _y = images[:4], labels[:4]
    _w = _model.hidden.weight  # 隠れ層の重みを例に見る
    _before = _w.detach().clone()  # 更新前の値をコピーしておく

    _optimizer.zero_grad()
    _loss = loss_fn(_model(_x), _y)
    _loss.backward()
    print("backward() の後、重みは変わっていない:", torch.equal(_w, _before))
    _optimizer.step()
    print("step() の後の重み ＝ 更新前 − 0.3 × 勾配:", torch.allclose(_w, _before - 0.3 * _w.grad))

    with torch.no_grad():
        _after = loss_fn(_model(_x), _y)
    print(f"この4枚の損失: {_loss.item():.3f} → {_after.item():.3f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `backward()` では重みは変わらず、`step()` で「更新前 − 学習率 × 勾配」に変わりました。03・04で手で計算した更新と同じです。
    同じ4枚の損失は、1回の更新で下がりました。

    ### 成績を測る関数：evaluate

    学習の途中や最後に、損失と**正解率**（予測が当たった割合）を何度も測るので、関数にしておきます。
    測るだけなので、`torch.no_grad()` の中で予測し、optimizerは使いません。重みは変わりません。

    `model.eval()` は、モデルを評価用のモードにする指定です。
    今回のモデルでは結果は変わりませんが、学習中と評価中で動きが変わる層もあるので、評価の前に書くのが決まりです（学習の前には `model.train()` で戻します）。
    """)
    return


@app.cell
def _(loss_fn, model, torch, validation_loader):
    def evaluate(model, loader, loss_fn):
        model.eval()  # 評価用のモードにする
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():  # 重みを更新しないので、勾配は求めない
            for x, y in loader:
                scores = model(x)
                loss_sum += loss_fn(scores, y).item() * len(y)  # 平均に枚数を掛けて、合計に直す
                correct += (scores.argmax(dim=1) == y).sum().item()  # 当たった枚数
                total += len(y)
        return {"loss": loss_sum / total, "accuracy": correct / total}

    before_training = evaluate(model, validation_loader, loss_fn)
    print(f"学習前のモデル（検証データ）: 損失 {before_training['loss']:.3f}、正解率 {before_training['accuracy']:.1%}")
    return before_training, evaluate


@app.cell(hide_code=True)
def _(before_training, mo):
    mo.md(rf"""
    学習前のモデルの正解率は {before_training['accuracy']:.1%} で、10個の数字からでたらめに選んだとき（10%）と同じくらいです。
    ここから、学習でどこまで上がるかを5節で確かめます。

    ### ここまでの確認問題

    1〜4節から2問です。
    """)
    return


@app.cell(hide_code=True)
def _(figure, model):
    _hidden = model.hidden.out_features
    figure("quiz", {"questions": [
        {"question": f"<b>Q1.</b> 28×28画素の画像を1枚ずつ、{_hidden}個のニューロンを持つ隠れ層を経て10クラスのスコアに変えるモデルがあります。64枚の画像をまとめて渡すと、出力の形はどれですか？",
         "options": ["<code>[64, 10]</code>", "<code>[64, 784]</code>", "<code>[10]</code>", f"<code>[64, {_hidden}]</code>"], "answer": 0,
         "explanation": "先頭の軸は画像の枚数で、1枚ごとに10個のスコアが出ます。まとめて渡しても、枚数の軸はなくなりません。"},
        {"question": "<b>Q2.</b> 学習のループから <code>optimizer.zero_grad()</code> の行を消すと、どうなりますか？",
         "options": ["前回までの勾配が足し合わされ、その値で重みを更新する", "重みが更新されなくなる",
                     "何も変わらない（勾配は毎回計算し直される）"], "answer": 0,
         "explanation": "PyTorchの <code>backward()</code> は、計算した勾配を前の値に<b>足して</b>記録します。消さないと、今のミニバッチの勾配に前回までの勾配が混ざります。"
                        "<code>step()</code> は呼ばれるので重みは変わりますが、今のミニバッチだけの勾配による更新にはなりません。"},
    ]}, height=620)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. 部品をつなげて、学習する

    4節の部品をつなげて、実際に学習します。下のセルの関数 `train_mnist` には、データの準備から学習・検証までがすべて書いてあります。
    前のセルの変数を使っていません。通常のPythonファイルで試すなら、このセルを `05-pytorch/experiment.py` に置き、教材ルートから `uv run python 05-pytorch/experiment.py` と実行します。`model.py` と同じフォルダに置くので、`from model import MLP` が見つかります。

    03・04で学んだことは、コードの次の行に当たります。

    | 学んだこと | 式・意味 | `train_mnist` の中のコード |
    |---|---|---|
    | 予測（順伝播） | 入力から各層を順に計算する | `scores = model(x)` |
    | 損失 | $\ell_i = -\log p_{i,\,y_i}$ の平均 | `loss = loss_fn(scores, y)` |
    | 勾配（逆伝播） | $\nabla L$ | `loss.backward()` |
    | 更新 | $\theta \leftarrow \theta - \eta\,\nabla L$ | `optimizer.step()` |
    | 学習率 | $\eta$ | `lr=learning_rate` |
    | ミニバッチ | 1回の更新に使う一部のデータ | `DataLoader(..., batch_size=128, shuffle=True)` |
    | エポック | 訓練データ全体を一巡する | `for epoch in range(1, epochs + 1):` |

    各エポックの終わりに、訓練データと検証データの両方で `evaluate` を呼び、成績を記録します。成績は `{'loss': 値, 'accuracy': 値}` という**辞書**に入れ、`result['loss']` のように名前で取り出します。`f"...{値:.1%}..."` は値を文字列に埋め込む書き方です。`with` など、今は設定を変えない補助行は暗記しなくてかまいません。
    最後の行の `train_mnist(...)` が、関数を実行する部分です。後の設定比較では、この行の数値を変えます。
    """)
    return


@app.cell
def _():
    def train_mnist(epochs=5, learning_rate=0.3, train_size=40_000, seed=42):
        import time
        from pathlib import Path

        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset, TensorDataset
        from torchvision.datasets import MNIST

        from model import MLP

        # 1. 乱数の出発点を揃える。重みの初期値と、訓練データの並び順が毎回同じになる。
        torch.set_num_threads(2)
        torch.manual_seed(seed)

        # 2. 画像を用意して、Tensorにする（1節）。
        mnist = MNIST(root=Path("data"), train=True, download=True)
        images = mnist.data.unsqueeze(1).float() / 255.0
        labels = mnist.targets.long()
        dataset = TensorDataset(images, labels)

        # 3. 検証5,000枚と訓練 train_size 枚に、重ならないように分ける（3節）。テスト用の画像は使わない。
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))
        validation_data = Subset(dataset, order[:5_000].tolist())
        train_data = Subset(dataset, order[5_000:5_000 + train_size].tolist())
        train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
        train_check_loader = DataLoader(train_data, batch_size=512)  # 訓練データの成績を測る用
        validation_loader = DataLoader(validation_data, batch_size=512)

        # 4. モデル、損失関数、optimizerを用意する（2・4節）。
        model = MLP()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # 5. 成績を測る関数（4節と同じ）。
        def evaluate(model, loader, loss_fn):
            model.eval()
            loss_sum, correct, total = 0.0, 0, 0
            with torch.no_grad():
                for x, y in loader:
                    scores = model(x)
                    loss_sum += loss_fn(scores, y).item() * len(y)
                    correct += (scores.argmax(dim=1) == y).sum().item()
                    total += len(y)
            return {"loss": loss_sum / total, "accuracy": correct / total}

        # 6. 学習する。1エポックごとに、訓練データと検証データで成績を測って記録する。
        history = []
        started = time.perf_counter()
        for epoch in range(1, epochs + 1):
            model.train()  # 学習用のモードに戻す
            for x, y in train_loader:  # x は128枚の画像、y はその正解
                optimizer.zero_grad()  # 前回の勾配を消す
                scores = model(x)  # 予測
                loss = loss_fn(scores, y)  # 損失
                loss.backward()  # 勾配（まだ重みは変わらない）
                optimizer.step()  # 更新（ここで重みが変わる）

            train = evaluate(model, train_check_loader, loss_fn)
            validation = evaluate(model, validation_loader, loss_fn)
            history.append({"epoch": epoch, "train": train, "validation": validation,
                            "train_size": len(train_data), "hidden_size": model.hidden.out_features,
                            "learning_rate": learning_rate})
            print(f"エポック {epoch}/{epochs}  訓練: 損失 {train['loss']:.3f} 正解率 {train['accuracy']:.1%}"
                  f"  検証: 損失 {validation['loss']:.3f} 正解率 {validation['accuracy']:.1%}")

        print(f"学習にかかった時間: {time.perf_counter() - started:.1f}秒")
        return model, history


    # ここで学習を実行する。設定を変えて試すときは、この行の数値を変える。
    trained_model, history = train_mnist(epochs=5, learning_rate=0.3)
    return history, train_mnist, trained_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    出力の数字だけでは変化をつかみにくいので、エポックごとの成績をグラフにします。
    横軸がエポック、左が損失、右が正解率です。青い実線が訓練データ、橙の破線が検証データでの成績です。
    """)
    return


@app.cell(hide_code=True)
def _(curves, history):
    _setting = history[-1]
    curves(history, f"図5　エポックごとの損失と正解率（訓練{_setting['train_size']:,}枚・隠れ層{_setting['hidden_size']}個・学習率{_setting['learning_rate']:g}）。点にカーソルを合わせると値が出る。")
    return


@app.cell(hide_code=True)
def _(history, mo):
    import math as _math
    _first, _last = history[0], history[-1]
    _gap = (_last["train"]["accuracy"] - _last["validation"]["accuracy"]) * 100
    _values = [r[split][key] for r in history for split in ("train", "validation") for key in ("loss", "accuracy")]
    _result = (f"最初のエポックの検証正解率は {_first['validation']['accuracy']:.1%}、最後は {_last['validation']['accuracy']:.1%} でした。最後の訓練損失は {_last['train']['loss']:.3f}、検証損失は {_last['validation']['loss']:.3f} です。\n\n"
               f"最後の訓練正解率は {_last['train']['accuracy']:.1%}、検証正解率は {_last['validation']['accuracy']:.1%} で、訓練から検証を引いた差は {_gap:.1f} ポイントです。曲線を見て、設定を変えるとこの差や損失がどう変わるか比べましょう。") if all(_math.isfinite(v) for v in _values) else (
               "この設定では損失または正解率が `nan` / `inf` になり、曲線を比較できません。更新幅が大きすぎる可能性があるので、学習率を小さくして訓練・検証の結果を比べてください。")
    mo.md(rf"""
    {_result}

    ## 6. 訓練データでの成績だけでは、分からないこと

    訓練データでの成績と検証データでの成績は、いつも近いのでしょうか。
    02で見たように、訓練データが少ないと、モデルが訓練データに合わせすぎることがあります。
    同じ `train_mnist` で、訓練データ **1,000枚** を30エポック学習した例を見ます。
    """)
    return


@app.cell
def _(train_mnist):
    _small_model, small_history = train_mnist(epochs=30, learning_rate=0.3, train_size=1_000)
    return (small_history,)


@app.cell(hide_code=True)
def _(curves, small_history):
    curves(small_history, "図6　訓練データを1,000枚にしたときの損失と正解率。検証データ（5,000枚）は図5と同じ。")
    return


@app.cell(hide_code=True)
def _(history, mo, small_history):
    _small, _large = small_history[-1], history[-1]
    _late = sum(r["validation"]["loss"] for r in small_history[-10:]) / 10  # 最後の10エポックの平均
    mo.md(rf"""
    訓練データでの正解率は {_small['train']['accuracy']:.1%} まで上がりました。訓練に使った1,000枚は、ほぼすべて当てられます。
    ところが、検証データでの正解率は {_small['validation']['accuracy']:.1%} にとどまりました。
    損失でも、訓練データでは {_small['train']['loss']:.2f} まで下がり続けた一方で、検証データでは途中から {_late:.1f} 前後で下がらなくなりました。

    モデルは、見た1,000枚の細かな特徴まで覚えてしまい、初めて見る画像には通用しない部分が増えました。
    02で学んだ**過学習（overfitting）**です。
    **訓練データでの成績だけを見ていたら、この違いに気付けません。** だから、学習に使わない検証データで成績を測ります。

    図5では、訓練{history[-1]['train_size']:,}枚で{len(history)}エポック学習し、最後の検証正解率は {_large['validation']['accuracy']:.1%} でした。図6は訓練1,000枚で30エポック学習した結果です。設定条件が異なり得るため、ここでは図6の中で訓練と検証の曲線が離れることに注目します。

    ### 設定を変える前の確認

    学習で重みを変える行を指し、図5・図6の訓練と検証の違いを説明できるか確かめましょう。次に5節の設定を一つ変え、**図5の訓練・検証の曲線で比較してから**、7・8節へ進みます。

    ## 7. 学習した重みを保存し、読み込む

    学習した重みは、今はPythonのメモリの中にしかありません。この画面を閉じると消えてしまいます。
    設定を検証結果で一つ選んだら、その設定で学習した `trained_model` を保存します。ボタンを押す前に、学習結果と図5が選んだ設定のものか確かめます。

    モデルの `state_dict()` は、層の名前と、その重み・バイアスの値の組です。これを `torch.save` でファイルに書き出します。
    保存するのは**数値だけ**で、ネットワークの形（どの層をどの順に通すか）は `model.py` に書いてあります。
    保存先は `outputs/05-pytorch/notebook/mnist_mlp.pt` です。同じ場所に保存し直すと、前のファイルは上書きされます。
    """)
    return


@app.cell(hide_code=True)
def _(history, mo):
    _epochs = len(history)
    save_model = mo.ui.run_button(label=f"検証で選んだ{_epochs}エポックのモデルを保存")
    save_model
    return (save_model,)


@app.cell
def _(Path, mo, save_model, torch, trained_model):
    mo.stop(not save_model.value)
    weights = trained_model.state_dict()
    for name, value in weights.items():
        print(name, tuple(value.shape))  # 層の名前と、値の形

    weights_path = Path("outputs/05-pytorch/notebook/mnist_mlp.pt")
    weights_path.parent.mkdir(parents=True, exist_ok=True)  # 保存先のフォルダがなければ作る
    torch.save(weights, weights_path)
    print("保存先:", weights_path.resolve())
    return (weights_path,)


@app.cell(hide_code=True)
def _(mo, trained_model):
    _hidden = trained_model.hidden.out_features
    mo.md(rf"""
    `hidden.weight` の形 `({_hidden}, 784)` は、隠れ層の{_hidden}個のニューロンそれぞれが、784個の入力に対する重みを持つ、という意味です。

    保存した重みを使うときは、まず `MLP()` で同じ形の新しいモデルを作り、`load_state_dict` でファイルの数値を入れます。
    **モデルの形の読み込み（`import`）と、重みの読み込み（`load`）は別の操作**です。
    下のセルでは、読み込んだモデルと学習直後のモデルの成績が一致するかを、検証データで確かめます。
    """)
    return


@app.cell
def _(
    MLP,
    evaluate,
    loss_fn,
    torch,
    trained_model,
    validation_loader,
    weights_path,
):
    restored_model = MLP()  # 学習前の、新しいモデル
    restored_model.load_state_dict(torch.load(weights_path, weights_only=True))  # 保存した重みを入れる

    for _name, _model in [("学習直後のモデル", trained_model), ("読み込んだモデル", restored_model)]:
        _result = evaluate(_model, validation_loader, loss_fn)
        print(f"{_name}（検証データ）: 損失 {_result['loss']:.4f}、正解率 {_result['accuracy']:.2%}")
    return (restored_model,)


@app.cell(hide_code=True)
def _(mo, restored_model):
    _ = restored_model
    mo.md(r"""
    二つの結果が同じなので、読み込み後も同じ検証成績が得られました。
    別の日にPythonを起動し直しても、`MLP` の `import` とこの読み込みをすれば、学習をやり直さずに使えます。

    ## 8. テスト画像で、最終的な成績を測る

    検証データで設定を決め、モデルを保存・読み込みした後で、テスト用の10,000枚を使います。**テスト結果を見て設定選びへ戻さない**ことが「最後に一度だけ」の意味です。再読み込みを禁止する意味ではありません。
    `train=False` を指定すると、テスト用の画像を取得します。画素の値は、訓練のときと同じく0〜1に揃えます。
    評価には、ファイルから読み込んだ `restored_model` を使います。
    """)
    return


@app.cell(hide_code=True)
def _(history, mo, restored_model):
    _epochs = len(history)
    _ = restored_model
    final_test = mo.ui.run_button(label=f"設定を決めたので最終テストを実行（{_epochs}エポックのモデル）")
    final_test
    return (final_test,)


@app.cell
def _(
    DataLoader,
    MNIST,
    TensorDataset,
    data_dir,
    evaluate,
    final_test,
    loss_fn,
    mo,
    restored_model,
):
    mo.stop(not final_test.value)
    mnist_test = MNIST(root=data_dir, train=False, download=True)  # テスト用の画像
    test_images = mnist_test.data.unsqueeze(1).float() / 255.0  # 訓練のときと同じ前処理
    test_labels = mnist_test.targets.long()
    test_loader = DataLoader(TensorDataset(test_images, test_labels), batch_size=512)

    test_result = evaluate(restored_model, test_loader, loss_fn)
    print(f"テスト正解率: {test_result['accuracy']:.2%}（{len(test_labels):,}枚）")
    return test_images, test_labels, test_result


@app.cell(hide_code=True)
def _(mo, test_result):
    mo.md(rf"""
    テストデータでの正解率は **{test_result['accuracy']:.2%}** でした。学習にも設定選びにも使っていない画像での成績なので、これをこのモデルの最終的な成績とします。

    ### どんな画像を、どう間違えたか

    正解率という一つの数だけでなく、実際の画像と予測も見てみます。
    03で学んだsoftmaxでスコアを確率に直すと、モデルがどのくらい迷ったかも分かります。
    下のセルでは、テスト画像すべての確率と予測を求め、予測が正解と違う画像の番号を探します。
    """)
    return


@app.cell
def _(restored_model, test_images, test_labels, torch):
    with torch.no_grad():
        test_scores = restored_model(test_images)  # [10000, 10]
    test_probabilities = torch.softmax(test_scores, dim=1)  # 各画像の10個の確率（合計1）
    test_predictions = test_probabilities.argmax(dim=1)

    wrong_ids = (test_predictions != test_labels).nonzero().flatten()  # 間違えた画像の番号
    print("間違えた枚数:", len(wrong_ids))
    print("間違えた画像の番号（先頭8枚）:", wrong_ids[:8].tolist())
    return test_predictions, test_probabilities, wrong_ids


@app.cell(hide_code=True)
def _(
    figure,
    pixels_hex,
    test_images,
    test_labels,
    test_predictions,
    test_probabilities,
    wrong_ids,
):
    def _item(i):
        return {"index": i, "pixels": pixels_hex(test_images[i, 0]), "label": int(test_labels[i]),
                "pred": int(test_predictions[i]), "probs": [round(float(p), 4) for p in test_probabilities[i].nan_to_num(0.0)]}

    figure("predictions", {"rows": [
        {"title": "テストの先頭8枚", "items": [_item(i) for i in range(8)]},
        {"title": "間違えた画像", "note": "番号の小さい順に8枚", "items": [_item(int(i)) for i in wrong_ids[:8]]},
    ]}, "図7　テスト画像の正解と予測（予測が違うものは橙）。下のグラフは、選んだ画像について、モデルが10個の数字に付けた確率。", height=700)
    return


@app.cell(hide_code=True)
def _(mo, test_labels, test_probabilities, wrong_ids):
    _wrong = test_probabilities[wrong_ids]
    _second = (_wrong.argsort(dim=1, descending=True)[:, 1] == test_labels[wrong_ids]).float().mean().item()
    mo.md(rf"""
    間違えた画像には、崩れた字や、別の数字に似た字が目立ちます。
    間違えた{len(wrong_ids)}枚のうち {_second:.0%} では、正解の数字がモデルの2番目に大きい確率でした。
    画像を押して、確率の分かれ方を比べてみてください。

    なお、この正解率はMNISTの手書き数字に対するものです。自分で撮った写真のように、背景や文字の大きさが違う画像では、同じようには当たりません。

    ## 確認問題

    5〜8節から3問です。
    """)
    return


@app.cell(hide_code=True)
def _(figure):
    figure("quiz", {"questions": [
        {"question": "<b>Q3.</b> 画像分類モデルの訓練データでの正解率は98%、学習に使わなかった検証データでは75%でした。検証データと同じ集め方をした新しい画像での成績をどう見積もりますか？",
         "options": ["訓練データに合わせすぎている。初めて見る画像では、検証データの成績くらいと考える",
                     "訓練データで98%なので、新しい画像でもほぼ98%当たる",
                     "検証データの成績は使わず、訓練データの成績で判断する"], "answer": 0,
         "explanation": "訓練データは、モデルが重みを合わせたデータなので、成績がよく出ます。初めて見る画像での成績は、学習に使わなかった検証データで見積もります。"
                        "大きな差は過学習の兆しです。ただし、新しい画像が検証データと同じような分布から来ることが前提です。"},
        {"question": "<b>Q4.</b> 学習率を 0.03・0.3・3 の3通りで学習し、一番よいものを選びたいとします。何を比べて選びますか？",
         "options": ["テストデータでの正解率", "検証データでの損失・正解率", "訓練データでの損失"], "answer": 1,
         "explanation": "設定を選ぶには検証データを使います。テストデータで選ぶと、最後の評価が公平でなくなります。訓練データの損失だけでは、過学習を見逃します。"},
        {"question": "<b>Q5.</b> 隠れ層が128個のニューロンを持つモデルの重みを保存しました。その後、隠れ層を256個に変更したモデルへ、保存した重みを <code>load_state_dict</code> で読み込むとどうなりますか？",
         "options": ["読み込めるが、正解率が下がる", "足りない重みは0で補われて、読み込める", "重みの形が合わないので、エラーになる"], "answer": 2,
         "explanation": "保存したのは128個のニューロンに対応する重みです。256個に変えたモデルでは重みの形が合わず、読み込み時にエラーになります。"
                        "重みファイルは、保存したときと同じ構造のモデルに読み込みます。"},
    ]}, height=900)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## まとめ

    | 工程（使うデータ） | コード | 重み |
    |---|---|---|
    | 予測・損失（訓練） | `loss = loss_fn(model(x), y)` | 変わらない |
    | 勾配（訓練） | `loss.backward()` | 変わらない |
    | 更新（訓練） | `optimizer.step()` | **ここで変わる** |
    | 途中の確認・設定選び（検証） | `evaluate(…, validation_loader, …)` | 変わらない |
    | 最終的な評価（テスト） | `evaluate(…, test_loader, …)` | 変わらない |

    - **PyTorchのモデルは、`__init__` で層を用意し、`forward` で通す順を書くクラス。** データの形は `[B, 1, 28, 28]` → `[B, 10]` と変わる。
    - **1回の更新は、`zero_grad` → 予測・損失 → `backward` → `step`。** `step()` は、03の $\theta \leftarrow \theta - \eta\,\nabla L$ そのもの。
    - **訓練データでの成績だけでは、初めて見るデータで当たるか分からない。** 設定は検証データで選び、最後にテストデータで一度だけ測る。
    - **重みファイルに入っているのは数値だけ。** 使うときは、同じ形の `model.py` のモデルに読み込む。

    **この章の確認：** 訓練・検証の結果で設定を選び、保存したモデルの最終テスト結果を設定選びに戻さずに説明できれば完了です。

    設定比較では、5節の最後の行 `train_mnist(epochs=5, learning_rate=0.3)` の数値を一つだけ変えて学習し直し、図5の訓練・検証の曲線を比べます。設定を選んでから7節で保存し、8節で最終テストを実行します。進め方は教材全体の `README.md` の「05の設定を一つ変えて比べる」にあります。

    もっと詳しく知りたいときは、PyTorchの公式チュートリアル
    [Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) と
    [重みの保存と読み込み](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) が参考になります。
    """)
    return


if __name__ == "__main__":
    app.run()
