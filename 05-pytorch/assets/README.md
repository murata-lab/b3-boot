# MNIST・配布モデル・実測記録

`reference/` は、同じ章の `model.py`・`data.py`・`train.py`・`evaluate.py` で作った学習済みモデルと記録です。CLIによる学習・評価の参考例として同梱しています。学習者が作る `outputs/05-pytorch/` の結果とは区別します。

## データの出典と画像

MNIST：Yann LeCun、Corinna Cortes、Christopher J. C. Burges。

- [MNISTの説明](https://yann.lecun.org/exdb/mnist/index.html)
- 取得はTorchvisionの `MNIST` APIを使用。[APIの説明](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
- 取得元：OSSciのMNIST配布ミラー。元gzipのSHA-256と配布画像の選び方は [manifest.json](manifest.json) に収録。
- 画像の利用条件は [TensorFlowのMNIST案内](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data) に従い、Yann LeCun・Corinna Cortesの著作権表記と [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) を付す。同梱画像もこの条件で配布。

表示画像は公式テストの先頭8枚と、固定済みモデルが誤分類した画像のID順で先頭8枚。28×28の元画素を変えず、可逆なグレースケールPNGに変換して `evaluation.json` 内にbase64で格納しています。拡大は表示時のみです。画像ごとに元のテストID・正解・予測を記録しています。

## 配布モデルの条件

MLP：784→128→10、ReLU、画素値を255で割る前処理。CPU、Adam、学習率0.001、batch 128、5 epoch、初期化・shuffleのseed 42。公式訓練60,000枚から検証5,000枚を固定して分け、残りからラベルの比率を保って訓練40,000枚を選んでいます。分割seedは42、検証・テストbatchは512、CPUスレッドは2です。

採用epochは検証損失が最小の5。検証正解率96.30%、設定固定後の公式テスト正解率 **96.43%（9,643/10,000）**。設定選択にはテストを使用していません。

- [mnist_mlp.pt](reference/mnist_mlp.pt)：`state_dict`。構造は `model.py` が定義。
- [run.json](reference/run.json)：設定、元データの訓練・検証ID、クラス数、環境、時間、重みのSHA-256。
- [history.json](reference/history.json)：epochごとの訓練・検証結果。
- [evaluation.json](reference/evaluation.json)：テスト全体の結果と表示画像。
- [benchmarks.json](benchmarks.json)：訓練20,000・30,000・40,000・55,000枚、各seed 42・43・44の検証精度と時間。採用40,000枚は全seedで95%以上かつ55,000枚との差1ポイント以内。

環境により再生成時の数値に小さな違いが出る場合があります。再現条件の詳細は `run.json` で確認できます。

## 再生成する

ルートから実行します。配布素材の再生成用であり、通常の学習者は実行不要です。

```sh
uv run python 05-pytorch/data.py
uv run python 05-pytorch/train.py --train-size 40000 --epochs 5 --seed 42 --output-dir 05-pytorch/assets/reference
uv run python 05-pytorch/evaluate.py --weights 05-pytorch/assets/reference/mnist_mlp.pt
```

上のコマンドは `reference/` の重み・記録を更新します。更新後は `manifest.json` の重みハッシュも更新してください。画像は評価スクリプトが自動的に同じ規則で選びます。

比較を再現する場合は、`train.py` の `--train-size` を20000・30000・40000・55000、`--seed` を42・43・44とし、別々の `--output-dir` を指定します。各 `run.json` の検証結果で比べ、`evaluate.py` は設定を決めたモデルにだけ実行します。性能の計測では同時に複数の学習を走らせないでください。
