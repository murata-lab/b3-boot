# 06 特徴空間と次元削減（任意）

05のMNISTモデルで作られた128個の特徴量を、PCA・Isomap・UMAPを通して観察する章です。提出物はありません。

共通の入口 `uv run python app.py` から06を開くか、単独で次を実行します。

```sh
uv run marimo run 06-feature-space/notebook.py
```

章の図は、MNISTテスト画像600枚（各数字60枚）と、05で学習・保存したモデルの重みを使って事前計算した結果です。実行時に学習や次元削減はしません。元画像と正解ラベルはMNISTから採りました。学習前は同じモデル構造に乱数シード6で初期値を入れています。各表現のUMAPはラベルを入力せずに計算しました。

図を再生成する場合は、先に05で `outputs/05-pytorch/notebook/mnist_mlp.pt` を保存し、次を実行します。再生成にはscikit-learnとumap-learnが一時的に必要です。

```sh
uv run --with scikit-learn --with umap-learn python 06-feature-space/build_assets.py
```

再生成すると、重みやライブラリの版によって数値・配置が変わる可能性があります。本文中の数値と例も再確認してください。

MNISTの原データ：[The MNIST Database](https://yann.lecun.com/exdb/mnist/)。手法の説明は章末の公式資料も参照してください。
