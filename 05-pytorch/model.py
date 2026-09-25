from torch import nn


class MLP(nn.Module):
    def __init__(self):
        # nn.Moduleを初期化し、このモデルが持つ層・重みを管理できるようにする。
        super().__init__()
        # 1枚の画像（1×28×28）を784個の値に並べる。バッチの軸は残す。
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(784, 128)
        # 非線形な変換を挟み、全結合層だけでは表せない特徴を学べるようにする。
        self.relu = nn.ReLU()
        # 0〜9の各数字に対応する、10個のスコアを出す。
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        # model(x)で呼ばれる計算。Bは一度に渡した画像の枚数。
        x = self.flatten(x)
        h = self.relu(self.hidden(x))  # [B, 784] → [B, 隠れ層のニューロン数]
        # [B, 10]のスコアを返す。CrossEntropyLossへ渡すためsoftmaxはしない。
        return self.output(h)
