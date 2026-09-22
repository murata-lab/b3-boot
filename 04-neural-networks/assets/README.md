# 手書き数字画像の出典

MNIST（Yann LeCun、Corinna Cortes、Christopher J. C. Burges）の訓練画像から、元の並び順でラベル3の最初の6枚、ラベル5の最初の6枚を採用。モデルの予測結果による選定はしていません。分類精度の比較や最終評価には使いません。

- [MNISTの説明](https://yann.lecun.org/exdb/mnist/index.html)
- [配布ミラー（CVDF）](https://github.com/cvdfoundation/mnist)
- 取得日：2026-09-11
- [ライセンスの案内（TensorFlow）](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data)：Yann LeCun・Corinna Cortesの著作権表記、[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)。同梱画像もこの条件で配布します。

元の28×28の画素値を変更せず、グレースケールPNGへ可逆変換しています。拡大は表示時だけ行います。元の画像番号・ラベル・配布元・元gzipのSHA-256は `manifest.json` に記録しています。

画像を再生成する場合は、配布元の `train-images-idx3-ubyte.gz` と `train-labels-idx1-ubyte.gz` を展開し、元の並び順でラベル3・5それぞれの先頭6枚を取り出します。各画像の28×28画素を変更せずグレースケールPNGに保存し、`manifest.json` の画像番号・ラベル・元gzipのSHA-256と照合します。画像は同梱済みのため、教材を読む際の生成処理や専用Pythonスクリプトは不要です。
