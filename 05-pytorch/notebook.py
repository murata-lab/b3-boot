# /// script
# [tool.marimo.runtime]
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="05 PyTorchによる学習と評価")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from pathlib import Path as _Path
    import sys as _sys
    _chapter = _Path(__file__).resolve().parent
    if str(_chapter) not in _sys.path:
        _sys.path.insert(0, str(_chapter))
    from lesson_ui import STYLE as _STYLE, file_preview as _file_preview
    model_source = mo.Html(_STYLE + _file_preview(_chapter / "model.py"))
    return mo, model_source


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 05 PyTorchによる学習と評価

    **別のファイルに書かれたモデルを読み込み、手書き数字を学習させ、保存した重みで初めて見る画像を分類します。**
    この章では、実際にモデルを作るときと同じ順序で、Pythonのコードを実行していきます。
    使う道具がPyTorchです。重みや勾配を管理する仕組みがあり、自分で全て計算し直す必要はありません。

    ### この画面の進め方

    説明の下にあるPythonの枠が、編集・実行できる**コードセル**です。
    アプリから開くと、初回は全セルを実行して出力を表示します（データ取得・学習・保存・テストを含みます）。
    上から順に読み、コードセルを選んで **Shift+Enter** で実行し直せます。
    `print(...)` の結果やグラフが、そのセルの出力に現れます。
    実行前に「何が入力で、何が出てくるか」を考え、出力と照らし合わせてみましょう。

    この章は、コードを変更しても後続の学習が勝手に再実行されない設定です。
    変更に影響されるセルは実行待ちになります。上から順に実行し直してください。
    同じ名前の変数を別セルに追加するのではなく、もとのセルを書き換えます。
    画面上部の **Run all（すべて実行）** を使うと、データ取得から学習・保存・テストまでまとめて進みます。
    初回のデータ取得にはネット接続が必要です。

    読みながらの実習は約60分。学習は訓練40,000枚・5周に抑え、CPUで2〜3分以内を目指す設定にしています。
    初回の環境準備とデータ取得の時間は別です。実際の学習時間は後のセルで測ります。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 別ファイルのモデルを読み込む

    長いプログラムは、役割ごとにファイルを分けると読みやすくなります。
    今回はモデルの設計を `model.py` に置き、学習の手順をこの `notebook.py` に書きます。
    モデルの定義を別の実験でも使いたくなったら、同じファイルを読み込めます。

    この教材は、ターミナルの現在地を **b3-ml-bootcamp** にして開きます。
    `pwd` で現在地、`ls` でその中のファイルを確認できます。

    ```text
    b3-ml-bootcamp/               ← 起動時の現在地
    ├── app.py                    ← 教材一覧
    ├── 05-pytorch/
    │   ├── notebook.py           ← 説明と、これから実行するコード
    │   └── model.py              ← MLPクラスの定義
    ├── data/MNIST/               ← 後で取得する画像
    └── outputs/05-pytorch/notebook/
        └── mnist_mlp.pt          ← 後で保存する重み
    ```

    ここでは本編に必要なファイルだけを示しています。
    `data` と `outputs` は実行時に作られます。

    ### フォルダの場所の書き方

    **相対パス**は、現在地を基準にした場所の指定です。
    現在地が教材全体のフォルダなら `05-pytorch/model.py`、章のフォルダへ移動した後なら `model.py` が同じファイルを指します。
    `./` は現在地、`../` は一つ上です。**絶対パス**は `/Users/...` や `C:/Users/...` のように、場所を最初から指定します。

    次が実際の **model.py** の内容です。ファイル名と行番号を見ながら、まず全体を眺めてください。
    この表示はファイルから直接読み取っています。ここはモデルの定義を読む場所で、実行するセルはその下にあります。
    """)
    return


@app.cell(hide_code=True)
def _(model_source):
    model_source
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### importして、モデルの構造を表示する

    `from model import MLP` は「`model.py` にある `MLP` という名前を読み込む」という意味です。
    `import` には `.py` や `/` を書きません。この教材は、隣にあるファイルを検索できるようにしてあります。
    通常のPythonでも、同じフォルダのスクリプトからこの形で読み込めます。

    `MLP` はモデルの作り方をまとめた**クラス**です。`MLP()` と括弧を付けて呼ぶと、一つのモデルが作られます。
    この実物を**インスタンス**といいます。`model` という変数に入れて、まず `print(model)` で層の構造を表示しましょう。
    """)
    return


@app.cell
def _():
    # 同じフォルダのmodel.pyから、モデルの作り方（クラス）を読み込む。
    from model import MLP

    model = MLP()  # 層と重みの初期値を用意する。この時点では未学習。
    print(model)
    return MLP, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    出力の `MLP(...)` の中に、`flatten`、`hidden`、`relu`、`output` が並びます。
    `Linear(in_features=784, out_features=128, bias=True)` は、784個の入力を128個の出力に変える全結合層です。
    `bias=True` はバイアスも使うという意味です。**importやインスタンスの作成だけでは、まだ数字を学習していません。**

    ### クラスの中では、何をしている？

    モデルには「持っている層・重み」と「入力から出力を計算する処理」があります。クラスはこの二つをまとめます。

    - `__init__`：`MLP()` で作られるときに呼ばれ、使う層を用意します。重みの初期値もここで作られます。
    - `self`：今作ったモデル自身を指します。`self.hidden` と書くと、そのモデルが持つ層になります。
    - `forward`：どの順序で層を通すかを決めます。`model(画像)` と呼ぶと、この計算が動きます。

    画像は28×28画素なので、`Flatten` で一列に並べると784個の数になります。
    それを128個の値へ変換し、`ReLU` で負の値を0にし、最後に0〜9それぞれに対応する10個のスコアを出します。
    高いスコアの数字を予測として選びます。
    全結合層を重ねるだけでは全体も線形の変換になるため、間にReLUを入れて、より複雑な特徴を学べるようにしています。
    ファイルの冒頭の `from torch import nn` は、これらの層やモデルの基本機能をまとめた `nn` をPyTorchから読み込む行です。

    ### なぜ `nn.Module` を継承するの？

    `class MLP(nn.Module)` は、PyTorchの `nn.Module` を親として、その仕組みを受け継ぐ書き方です。
    自分で書くのは、このモデルに必要な層と `forward` の計算です。
    親が持つ仕組みによって、層の重みをまとめて取得したり、保存・読み込みをしたりできます。

    例えば後で使う `model.parameters()` は全ての層の重みとバイアスを集めます。
    普通のPythonクラスに `forward` だけを書いても、この機能は付きません。
    `super().__init__()` は親の初期化を行い、その後で `self.hidden` などに代入する層を管理できるようにします。

    `model(...)` はPyTorch共通の処理を経て `forward` を呼びます。普段は `model.forward(...)` を直接呼びません。
    単純な直列のモデルは `nn.Sequential` でも書けます。今回はファイルのクラスを読み込んで使う練習として、この形を使います。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 画像データをダウンロードする

    今回は **MNIST** という手書き数字のデータを使います。
    一つのデータは、白黒の画像と「この画像は7」のような**正解ラベル**の組です。
    画像だけでは、モデルが出した答えが正しいか分かりません。学習ではこの正解と比べて重みを直します。

    ### コードでダウンロードする

    `torchvision` は、画像のデータセットなどを使うためのライブラリです。
    `MNIST(root=..., train=True, download=True)` で公式の訓練用データを取得します。
    `train=True` は「訓練用のデータを選ぶ」指定で、これだけで学習は始まりません。
    `download=True` は、手元になければダウンロードする指定です。取得済みならそのデータを使います。

    `root` はデータを置くフォルダです。ここでは `Path("data")` とその場で指定します。
    `Path.cwd()` は現在地、`resolve()` はその保存先を絶対パスで表示するために使います。
    学習の設定とは別の話なので、保存先のために設定ファイルを編集する必要はありません。
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import torch
    from torch import nn
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader, TensorDataset, Subset
    import matplotlib.pyplot as plt

    # 小さなモデルをCPUで動かす。この値は学習率ではなくCPUの並列数。
    torch.set_num_threads(2)
    data_dir = Path("data")
    print("現在地:", Path.cwd())
    print("データの保存先:", data_dir.resolve())
    # 公式の訓練用画像を取得する。取得済みなら再利用し、ここでは学習しない。
    mnist = MNIST(root=data_dir, train=True, download=True)
    print("画像の枚数:", len(mnist))
    return (
        DataLoader,
        MNIST,
        Path,
        Subset,
        TensorDataset,
        data_dir,
        mnist,
        nn,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### まず1枚見てみる

    `mnist[0]` は最初の「画像とラベル」の組です。添字は0から始まります。
    下のセルでは画像を描き、その正解を見出しに表示します。
    ここで正解を表示できるのは、データに答えが添えられているからです。モデルの予測はまだしていません。
    """)
    return


@app.cell
def _(mnist, plt):
    # 画像と、その画像に付いている正解ラベルを一緒に取り出す。
    image, label = mnist[0]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 数値の配列（Tensor）にする

    コンピュータには画像を数値の集まりとして渡します。PyTorchで使う配列が **Tensor（テンソル）** です。
    MNISTの画素値は0〜255で、大きいほど白くなります。今回は255で割り、0〜1の小数に揃えます。
    値の尺度を揃えることで、学習の計算を扱いやすくします。テスト画像にも同じ処理が必要です。

    `shape` は各方向の大きさです。元は `[枚数, 縦, 横]` ですが、白黒の1チャンネルを表す軸を
    `unsqueeze(1)` で加え、`[枚数, 1, 28, 28]` にします。
    `float()` は画素値を小数、`long()` はラベルを整数として扱う指定です。
    `TensorDataset` は、この画像と正解の対応を保って取り出せるようにまとめます。
    """)
    return


@app.cell
def _(TensorDataset, mnist):
    # 白黒のチャンネル軸を追加し、画素値を0〜255の整数から0〜1の小数へ揃える。
    images = mnist.data.unsqueeze(1).float() / 255.0
    labels = mnist.targets.long()  # 正解は確率ではなく、0〜9の整数。
    # 画像と正解の対応を保ったまま、同じ番号で取り出せるようにする。
    dataset = TensorDataset(images, labels)
    print("画像:", images.shape, images.dtype)
    print("正解:", labels.shape, labels.dtype)
    print("画素の範囲:", images.min().item(), images.max().item())
    return (dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. なぜ、全ての画像を学習に使わないのか

    練習問題の答えを覚えても、初めて見る問題を解けるとは限りません。
    モデルも同じです。訓練に使った画像ではよく当たるのに、別の画像では間違えることがあります。
    訓練データの細部に合わせすぎて過学習を起こし、新しいデータに対応できなくなります。

    全てを学習に使い、その同じデータで正解率を調べるだけでは、この問題に気付きにくくなります。
    そこで、**学習に使わない画像を先に取り分けておき、初めて見る画像でも当たるか確認します。**
    この最後の確認に使うものが**テストデータ**です。
    例えば全体の1〜2割をテストに残す方法があります。割合はデータの量や性質で変わり、必ず2割という決まりではありません。
    分けること自体が過学習を防ぐのではなく、未知のデータへの性能を確かめる手段になります。

    MNISTは最初から、訓練用60,000枚とテスト用10,000枚に分かれています。
    テストは全70,000枚の約14%です。この公式の分け方をそのまま使い、テスト用画像は最後まで学習に渡しません。

    ### 学習途中の確認には「検証データ」を使う

    学習を何周続けるか、学習率をいくつにするかは、人が選ぶ設定です。
    その度にテストの点数を見て選び直すと、テストの問題にも間接的に合わせてしまいます。
    すると「初めて見る問題での点数」として扱いにくくなります。

    そこで訓練用60,000枚から、さらに5,000枚を**検証データ**として取り分けます。
    重みを更新するための訓練データとは分け、学習途中の成績を見るために使います。
    今回は、各周の終わりに検証し、検証で最も損失が小さかった時点の重みを残します。

    | 名前 | 使う時期と目的 | 今回の枚数 |
    |---|---|---:|
    | 訓練（train） | 画像と正解を使って重みを更新する | 40,000 |
    | 検証（validation） | 学習途中に確認し、残す重みや設定を選ぶ | 5,000 |
    | テスト（test） | 設定を決めた後、最終的な正解率を測る | 10,000 |

    残りの訓練用15,000枚は、実習時間を短くするため今回は使いません。
    これをテストに足すこともしません。

    下のコードでは、0〜59,999の番号を `randperm` でランダムに並べ替え、最初の5,000個を検証、その次の40,000個を訓練に使います。
    重ならない範囲から取るので、同じ画像が両方に入りません。`Subset` は選んだ番号のデータだけを取り出す道具です。
    `seed` は乱数の出発点です。同じ番号に固定すると、実行のたびに分け方が変わるのを避けられます。
    """)
    return


@app.cell
def _(Subset, dataset, torch):
    # 同じseedなら同じ分け方になるように、画像の番号をシャッフルする。
    split_generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(dataset), generator=split_generator)
    # 重ならない範囲を使う。検証用の画像では重みを更新しない。
    validation_ids = indices[:5_000]
    train_ids = indices[5_000:45_000]
    # 残り15,000枚は実習時間を短くするため使わず、テストにも加えない。
    train_data = Subset(dataset, train_ids.tolist())
    validation_data = Subset(dataset, validation_ids.tolist())
    print("訓練:", len(train_data), "検証:", len(validation_data))
    print("両方に入った画像:", len(set(train_ids.tolist()) & set(validation_ids.tolist())))
    return train_data, validation_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 一度に128枚ずつ渡す：batchとDataLoader

    全40,000枚を一度に計算するとメモリが多く必要です。そこで小分けにして処理します。
    この一まとまりが**バッチ（batch）**、その枚数が `batch_size` です。
    `DataLoader` はデータをこのまとまりに分け、`for` で順に取り出せるようにします。

    訓練では `shuffle=True` にして、毎周同じ順番ばかりで学習しないようにします。
    検証では重みを更新しないので、順番を変える必要はありません。
    下では1バッチを取り出し、読み込んだモデルに渡して出力の形も確認します。
    これはまだ学習前なので、数字が正しく当たることは期待しません。
    """)
    return


@app.cell
def _(DataLoader, model, torch, train_data, validation_data):
    batch_size = 128
    # 訓練では毎epochの順番を変え、128枚ずつ取り出す。
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(42))
    # 検証では更新をしないため順番を固定する。512枚ずつ計算しても全画像を評価する。
    validation_loader = DataLoader(validation_data, batch_size=512, shuffle=False)
    batch_images, batch_labels = next(iter(train_loader))
    # 学習前の出力の形だけを確認するので、勾配は記録しない。
    with torch.no_grad():
        batch_scores = model(batch_images)
    print("入力の画像:", batch_images.shape)
    print("正解:", batch_labels.shape)
    print("モデルの出力:", batch_scores.shape)
    return batch_size, train_loader, validation_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    出力の `[128, 10]` は「128枚それぞれに、0〜9の10個のスコアが出た」という意味です。
    入力の画像がモデルの `forward` を通り、この形に変わりました。
    `torch.no_grad()` は、ここでは出力を見るだけなので、重みを直すための勾配を記録しない指定です。

    ## 4. 学習に必要な部品を、一つずつ確かめる

    ここまではモデル・画像・データを渡す仕組みを用意しました。
    次は、予測を採点する部品と重みを直す部品を、小さなコードで確かめます。
    最後に、ここまでの部品を全て含む一つの学習プログラムにまとめます。

    ### 予測を採点する：損失関数

    学習では、予測が正解に近付くように重みを少しずつ変えます。そのために二つの道具を用意します。

    **損失関数**は、予測と正解のずれを一つの数にするものです。小さいほどよい、と考えます。
    今回は10種類から一つを選ぶ分類なので `nn.CrossEntropyLoss()` を使います。
    モデルのスコアと正解の整数を渡します。必要な処理を内部で行うので、渡す前に `softmax` で確率へ変換しません。

    また、全ての訓練データを一巡することを **1エポック（epoch）** と呼びます。
    今回は5周します。40,000枚を128枚ずつに分けるため、1周に313回重みを更新し、最後だけ64枚です。
    """)
    return


@app.cell
def _(batch_size, nn, train_data):
    epochs = 5  # 選んだ訓練データを何周するか。
    learning_rate = 0.001  # 重みの更新の大きさに関わる設定。
    seed = 42  # 初期値と訓練の並び順を再現するための乱数の出発点。
    # softmax前の10個のスコアと、正解の整数を比較する分類用の損失。
    loss_fn = nn.CrossEntropyLoss()
    print(f"訓練 {len(train_data):,}枚 / {epochs} epoch / batch {batch_size}")
    return epochs, learning_rate, loss_fn, seed


@app.cell
def _(images, labels, loss_fn, model, torch):
    # まず4枚だけ採点する。予測を見るだけなので、重みは更新しない。
    with torch.no_grad():
        _scores = model(images[:4])
        _loss = loss_fn(_scores, labels[:4])
    print("予測した数字:", _scores.argmax(dim=1).tolist())
    print("正解の数字:", labels[:4].tolist())
    print("損失:", _loss.item())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    予測と正解を比較して、損失という一つの数を計算できました。
    正解率と違い、損失はスコアの変化に応じて変わるので、重みをどちらへ動かすかを計算するために使えます。

    ### 重みを直す：optimizerと逆伝播

    **optimizer** は、勾配を使って重みを更新する道具です。今回はAdamという方法を使います。
    **勾配**は、重みを少し変えたときに損失がどちらへ、どれくらい変わるかを表す量です。
    その情報をもとに、損失が小さくなる方向へ重みを調整します。
    `model.parameters()` で更新する重み・バイアスを渡し、`lr` で**学習率**を指定します。
    学習率は更新の大きさに関わる設定です。大きすぎると学習が不安定になり、小さすぎると進みが遅くなることがあります。
    次のセルでは、仕組みの確認用に別のモデルを作り、4枚を使って**1回だけ**重みを更新します。
    `backward()` は勾配を計算する操作、`step()` はその勾配を使って重みを変える操作です。
    PyTorchは勾配を加算して記録するため、計算前に `zero_grad()` で前の勾配を消します。
    最後に隠れ層の重みの変化を調べ、0ではない値が出ることを確認しましょう。
    1回の更新であらゆる画像の損失が必ず下がる、という意味ではありません。
    """)
    return


@app.cell
def _(MLP, images, labels, learning_rate, loss_fn, torch):
    _model = MLP()  # 部品の確認用。最後のまとまった学習では新しく作り直す。
    _optimizer = torch.optim.Adam(_model.parameters(), lr=learning_rate)
    _before = _model.hidden.weight.detach().clone()  # 更新前の値をコピーする。

    _optimizer.zero_grad()  # 前の勾配をリセット。
    _scores = _model(images[:4])  # 予測。
    _loss = loss_fn(_scores, labels[:4])  # 正解と比べて損失を計算。
    _loss.backward()  # 各重みの勾配を計算。この行ではまだ重みは変わらない。
    _optimizer.step()  # 勾配を使って重みを更新。

    _change = (_model.hidden.weight.detach() - _before).abs().max().item()
    print("隠れ層の重みの変化（絶対値の最大）:", _change)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 検証の計算を関数にする

    学習の各周で同じ確認を行うので、先に `evaluate` という関数を作っておきます。
    関数を定義するだけでは評価は実行されません。後でモデル・データ・損失関数を渡して呼びます。

    `model.eval()` は評価用のモード、`torch.no_grad()` は勾配を記録しない指定です。
    どちらも使います。このMLPにはモードで動作が変わる層はありませんが、Dropoutなどを使うモデルでも必要になる基本の形です。
    `eval()` だけでは勾配の記録は止まりません。

    `argmax(dim=1)` は、各画像の10個のスコアから最大のものの位置、つまり予測した数字を取り出します。
    正解と等しいものを数え、全体の枚数で割ると**正解率**になります。
    損失は学習に使うずれの大きさ、正解率は当たった割合で、別の指標です。
    バッチの平均損失には枚数を掛けてから足します。最後の小さなバッチも公平に集計するためです。
    """)
    return


@app.cell
def _(torch):
    def evaluate(model, loader, loss_fn):
        model.eval()  # 層を評価用のモードにする。これだけでは勾配の記録は止まらない。
        loss_sum = 0.0
        correct = 0
        total = 0
        # 評価では勾配を記録せず、optimizerによる重みの更新も行わない。
        with torch.no_grad():
            for x, y in loader:
                scores = model(x)
                # バッチの平均損失に枚数を掛け、最後の小さなバッチも公平に集計する。
                loss_sum += loss_fn(scores, y).item() * len(y)
                predictions = scores.argmax(dim=1)  # 各画像で最大スコアの数字を選ぶ。
                correct += (predictions == y).sum().item()
                total += len(y)
        return {"loss": loss_sum / total, "accuracy": correct / total}

    return (evaluate,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 検証結果から、残す重みを選ぶ

    検証の計算も用意できました。次は、その結果をどう使うかです。
    毎周の検証損失を比べ、これまでで最も小さければ、その時点の重みを残します。
    `state_dict()` で重みの名前と値を取り出し、`deepcopy` でコピーしておくと、
    その後の学習で値が変わっても、よかった時点の重みを保てます。
    次の周を始める前には `train()` で学習用のモードへ戻します。

    ## 5. 部品をつなげて、実際に学習してみよう

    モデル、データ、DataLoader、損失関数、optimizer、検証の仕組みが揃いました。
    **下のセルには、importからデータの準備・学習・検証まで、必要な部品を全て書いてあります。**
    前のセルで作った変数に頼らず、このまとまりで学習できます。外部から読むモデル定義は、最初に見た `model.py` です。

    `train_mnist` という関数にまとめ、中のコメント1〜6で、これまでの部品と対応付けています。
    関数の中身を上から読み、その下の `train_mnist(...)` が全体を実行する入口だと確かめてください。
    `for epoch ...` は5周、内側の `for x, y ...` はバッチごとの繰り返しです。
    先ほど1回だけ試した「予測→損失→勾配→更新」を、全ての訓練画像に繰り返します。

    **再実行すると、データを用意してモデルを初期値から学習し直します。**
    ダウンロード済みの画像は再利用します。部品を確かめた前のセルを書き換えても、この全体コードの設定は変わりません。
    学習の設定を試すときは、**このセルの最後にある `train_mnist(epochs=5, learning_rate=0.001, seed=42)` の数値**を変更します。
    戻り値は順に、学習したモデル、検証で選んだ重み、採用したepoch、学習履歴です。次の保存や作図で使います。
    """)
    return


@app.cell
def _():
    def train_mnist(epochs=5, learning_rate=0.001, seed=42):
        import time
        from copy import deepcopy
        from pathlib import Path
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset, Subset
        from torchvision.datasets import MNIST
        from model import MLP

        # 1. 設定と乱数の出発点を揃える。
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        batch_size = 128

        # 2. 訓練用画像を取得し、モデルへ渡すTensorにする。取得済みなら再利用する。
        mnist = MNIST(root=Path("data"), train=True, download=True)
        images = mnist.data.unsqueeze(1).float() / 255.0
        labels = mnist.targets.long()
        dataset = TensorDataset(images, labels)

        # 3. 検証5,000枚と訓練40,000枚を重ならないように分ける。テストは使わない。
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))
        validation_data = Subset(dataset, indices[:5_000].tolist())
        train_data = Subset(dataset, indices[5_000:45_000].tolist())
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed))
        validation_loader = DataLoader(validation_data, batch_size=512, shuffle=False)

        # 4. モデル、損失関数、重みを更新するoptimizerを用意する。
        trained_model = MLP()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(trained_model.parameters(), lr=learning_rate)

        # 5. 学習に使わなかった画像で確認する関数も、この中に定義する。
        def evaluate(model, loader, loss_fn):
            model.eval()  # 層を評価用のモードにする。これだけでは勾配の記録は止まらない。
            loss_sum = 0.0
            correct = 0
            total = 0
            # 評価では勾配を記録せず、optimizerによる重みの更新も行わない。
            with torch.no_grad():
                for x, y in loader:
                    scores = model(x)
                    # バッチの平均損失に枚数を掛け、最後の小さなバッチも公平に集計する。
                    loss_sum += loss_fn(scores, y).item() * len(y)
                    predictions = scores.argmax(dim=1)  # 各画像で最大スコアの数字を選ぶ。
                    correct += (predictions == y).sum().item()
                    total += len(y)
            return {"loss": loss_sum / total, "accuracy": correct / total}


        # 6. 予測・損失・勾配・更新を繰り返し、検証でよい重みを選ぶ。
        history = []
        best_loss = float("inf")  # 最初の検証結果を必ず採用できるよう、無限大から始める。
        best_weights = None
        best_epoch = 0
        started = time.perf_counter()

        for epoch in range(1, epochs + 1):
            # 前の周の検証でeval()にしたモデルを、学習用のモードへ戻す。
            trained_model.train()
            loss_sum = 0.0
            correct = 0
            total = 0
            for x, y in train_loader:  # xは画像のバッチ、yは対応する正解ラベル。
                optimizer.zero_grad()  # 前のバッチの勾配を消す（放置すると加算される）。
                scores = trained_model(x)  # 予測：forwardを通して10個のスコアを計算。
                loss = loss_fn(scores, y)  # 損失：予測と正解のずれを一つの数にする。
                loss.backward()  # 逆伝播：勾配を計算する。まだ重みは変わらない。
                optimizer.step()  # 更新：計算した勾配を使って重みを変える。
                # グラフ用に、この周の損失と正解数を集計する。
                loss_sum += loss.item() * len(y)
                correct += (scores.argmax(dim=1) == y).sum().item()
                total += len(y)

            # 一周が終わったら、訓練に使わなかった検証画像で確認する。テストはまだ使わない。
            validation = evaluate(trained_model, validation_loader, loss_fn)
            history.append({"epoch": epoch, "train_loss": loss_sum / total,
                            "train_accuracy": correct / total, **validation})
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                # この後の学習で値が変わらないよう、最もよかった時点の重みをコピーする。
                best_weights = deepcopy(trained_model.state_dict())
                best_epoch = epoch
            print(f"epoch {epoch}/{epochs}: train loss={loss_sum / total:.4f}, "
                  f"validation loss={validation['loss']:.4f}, "
                  f"validation accuracy={validation['accuracy']:.2%}")

        training_seconds = time.perf_counter() - started
        print(f"学習・検証: {training_seconds:.1f}秒 / 残す重み: epoch {best_epoch}")

        return trained_model, best_weights, best_epoch, history

    # ここで実行する。設定を試すときは、この呼び出しの数値を変える。
    trained_model, best_weights, best_epoch, history = train_mnist(
        epochs=5, learning_rate=0.001, seed=42
    )
    return best_epoch, best_weights, history, train_mnist, trained_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 学習の経過をグラフで読む

    横軸は何周目か、縦軸は損失です。訓練と検証の両方で下がれば、見ていない画像への予測も改善していると考えられます。
    訓練の損失だけ下がり、検証の損失が上がり始めたら、過学習の兆候です。
    そのため、最後の周が必ず最もよいとは限りません。今回の短い学習では、上がる様子が出ない場合もあります。

    青の訓練損失は、重みが更新されている一周の途中の平均です。
    検証損失は一周が終わった時点のモデルで測ります。訓練の線が常に下になるとは限りません。
    """)
    return


@app.cell
def _(history, plt):
    curve_fig, curve_ax = plt.subplots(figsize=(7, 3))
    curve_ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], "o-", label="Train")
    curve_ax.plot([r["epoch"] for r in history], [r["loss"] for r in history], "o--", label="Validation")
    curve_ax.set(xlabel="Epoch", ylabel="Loss", xticks=[r["epoch"] for r in history])
    curve_ax.legend()
    curve_fig.tight_layout()
    curve_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. 学習した重みをファイルに保存する

    今の重みは、動いているPythonのメモリにあります。終了すると消えるため、次回も使うにはファイルに残します。

    `model.py` は**層の構造と計算方法**を書いたファイルです。
    学習して得られた**重み・バイアスの数値**は別で、`state_dict()` に層名と値の組として入ります。
    例えば `hidden.weight` は隠れ層の重みです。先ほどの `best_weights` は、この組をコピーしたものです。

    下のセルで値の名前と形を確認し、`torch.save` で保存します。
    保存先はこのセルで指定する `outputs/05-pytorch/notebook/mnist_mlp.pt` です。
    `mkdir(parents=True, exist_ok=True)` は途中のフォルダも作り、すでにあってもそのまま使います。
    `Path` の `/` はパスをつなぐ記号です。同じ場所で再実行すると、前の重みを上書きします。
    """)
    return


@app.cell
def _(Path, best_epoch, best_weights, torch):
    # 層の名前と重みの形を確認する。クラスの定義そのものは保存しない。
    for name, value in best_weights.items():
        print(name, tuple(value.shape))

    weights_path = Path("outputs/05-pytorch/notebook/mnist_mlp.pt")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    # 最終epochとは限らず、検証で選んだ重みを保存する。同名ファイルは上書きする。
    torch.save(best_weights, weights_path)
    print(f"epoch {best_epoch} の重みを保存:", weights_path.resolve())
    return (weights_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. 保存した重みを、新しいモデルへ読み込む

    学習済みのモデルを使うときは、同じ構造の空のモデルを `MLP()` で作り、保存した数値を `load_state_dict` で入れます。
    下では新しい `restored_model` を作るので、学習セルの `trained_model` をそのまま評価しているわけではありません。
    **構造の読み込み（import）と、重みの読み込み（load）は別の操作**です。
    層の大きさを変えると保存した数値の形が合わなくなるため、同じ構造が必要です。

    `torch.load` がファイルを読み、`load_state_dict` がそれを各層へ反映します。
    `weights_only=True` は重みの読み込みに使う指定、`map_location="cpu"` はCPUへ読み込む指定です。
    別の日に新しくPythonを起動しても、`MLP` のimportとこの読み込みを行えば、学習をやり直さずに使えます。
    """)
    return


@app.cell
def _(MLP, torch, weights_path):
    # 保存した重みと同じ形の層を用意し、ファイルの数値を各層へ戻す。
    restored_model = MLP()
    saved_weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    restored_model.load_state_dict(saved_weights)
    restored_model.eval()
    print(restored_model)
    return (restored_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. 初めて見るテスト画像で、正解率を調べる

    ここで初めて公式のテスト用10,000枚を用意します。
    取得の指定を `train=False` に変え、画素値は訓練時と同じように0〜1にします。
    `evaluate` を呼ぶだけなので、重みの更新はありません。
    全てのテスト画像で調べ、正解した枚数の割合を表示します。

    この結果は、用意されたMNISTの手書き数字に対する成績です。
    自分で撮った写真や、背景・文字の大きさが違う画像でも同じ精度になるとは限りません。
    まずは、今回のデータに対して学習から評価まで一巡できたことを確認しましょう。
    """)
    return


@app.cell
def _(
    DataLoader,
    MNIST,
    TensorDataset,
    data_dir,
    evaluate,
    loss_fn,
    restored_model,
):
    # ここで初めて公式テスト画像を使う。訓練・設定選びには渡していないデータ。
    mnist_test = MNIST(root=data_dir, train=False, download=True)
    # 入力の形と画素値の範囲は、訓練時と同じにする。
    test_images = mnist_test.data.unsqueeze(1).float() / 255.0
    test_labels = mnist_test.targets.long()
    test_data = TensorDataset(test_images, test_labels)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False)
    # メモリ上の学習直後のモデルではなく、ファイルから復元したモデルを評価する。
    test_result = evaluate(restored_model, test_loader, loss_fn)
    print(f"テスト正解率: {test_result['accuracy']:.2%}（{len(test_data):,}枚）")
    print(f"テスト損失: {test_result['loss']:.4f}")
    return test_images, test_labels, test_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 画像を見ながら、予測と正解を比べる

    正解率という一つの数字に加え、どんな画像に何と答えたかも見ます。
    下ではテストの先頭8枚を取り出し、`restored_model(...)` で予測します。
    `argmax(dim=1)` が取り出した数字と正解を並べます。表示の `true` が正解、`pred` が予測です。
    """)
    return


@app.cell
def _(plt, restored_model, test_images, test_labels, torch):
    # 先頭8枚の予測を求め、画像・正解・予測を見比べる。
    with torch.no_grad():
        predictions = restored_model(test_images[:8]).argmax(dim=1)
    prediction_fig, prediction_axes = plt.subplots(2, 4, figsize=(8, 4))
    for i, prediction_axis in enumerate(prediction_axes.flat):
        prediction_axis.imshow(test_images[i, 0].numpy(), cmap="gray")
        prediction_axis.set_title(f"true: {test_labels[i].item()} / pred: {predictions[i].item()}")
        prediction_axis.axis("off")
    prediction_fig.tight_layout()
    prediction_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 間違えた画像も確認する

    最初の8枚が全問正解でも、全体で間違いがないとは限りません。
    下はテスト全体から予測と正解が違う番号を探し、その先頭8枚を表示するコードです。
    似た形の数字をどう間違えたか観察しましょう。

    ここで見つけた間違いを参考に設定を選び直すと、このテストは開発にも使ったことになります。
    設定を試す練習では、まず検証結果を使って比べ、最後の評価用データは分けておく、という流れを覚えてください。
    """)
    return


@app.cell
def _(plt, restored_model, test_images, test_labels, test_loader, torch):
    with torch.no_grad():
        # バッチごとの予測を、テスト画像と同じ順番で一つに連結する。
        all_predictions = torch.cat([restored_model(x).argmax(dim=1) for x, y in test_loader])
    # 予測と正解が違う画像の番号を探し、先頭8枚を表示する。
    wrong_ids = (all_predictions != test_labels).nonzero().flatten()[:8]
    error_fig, error_axes = plt.subplots(2, 4, figsize=(8, 4))
    for axis in error_axes.flat:
        axis.axis("off")
    for index, axis in zip(wrong_ids.tolist(), error_axes.flat):
        axis.imshow(test_images[index, 0].numpy(), cmap="gray")
        axis.set_title(f"true: {test_labels[index].item()} / pred: {all_predictions[index].item()}")
    error_fig.tight_layout()
    error_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. 自分で学習を組むときの流れ

    この章で動かした順序は、別の画像分類でも基本になります。

    1. 画像と正解を用意し、訓練・検証・テストの役割を分ける。
    2. 入力の形と値の範囲を揃え、DataLoaderでバッチにする。
    3. モデル・損失関数・optimizerを用意する。
    4. 予測 → 損失 → 勾配 → 重みの更新を繰り返し、検証結果を確認する。
    5. 選んだ重みを保存し、同じ構造のモデルへ読み込んでテストする。

    コードを読み返して、**どの行が重みを変えたか**、**どのデータでは重みを変えなかったか**、
    **model.pyと重みファイルはそれぞれ何を保存しているか**を、自分の言葉で説明してみてください。

    発展として、全体コードの最後の `train_mnist(...)` に渡す `epochs` や `learning_rate` を編集して、訓練・検証の曲線を比べられます。
    変更後は学習セルから実行し直します。今回のテストを見た後の比較は実習として扱い、未知データに対する最終成績とは区別します。

    本編のコードはこのnotebookにあります。毎回まとめて実行したくなったら、同じ手順を通常のPythonファイルへ移せます。
    フォルダ内の `train.py`・`evaluate.py` はターミナル実行用の別例です。追加の実行記録や、数字の比率を保つ分割も含むため、本編と重みは一致しません。
    まずはここに表示したコードだけで一巡できれば十分です。

    詳しく確認するときは、[PyTorchの入門](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)、
    [重みの保存と読み込み](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)、
    [Pythonのモジュール](https://docs.python.org/ja/3/tutorial/modules.html)、
    [データ分割と検証](https://scikit-learn.org/stable/modules/cross_validation.html)、
    [marimoの実行設定](https://docs.marimo.io/guides/configuration/runtime_configuration/)を参照してください。
    """)
    return


if __name__ == "__main__":
    app.run()
