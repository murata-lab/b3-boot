import marimo

__generated_with = "0.24.0"
app = marimo.App(
    width="medium",
    app_title="01 Python入門",
    css_file="notebook.css",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    def check(choice, answer, explanation):
        # 選択肢の答え合わせ。選ぶまでは解説を表示しない。
        if choice.value is None:
            return None
        correct = choice.value == answer
        head = '**正解です。**' if correct else f'**正解は「{answer}」です。**'
        return mo.callout(mo.md(head + ' ' + explanation), kind='success' if correct else 'warn')

    return (check,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 01 Python入門

    Pythonの基本構文を、コードを動かしながら学びます。最後にNumPyの配列とグラフを扱います。

    **この章の目標：** 関数の戻り値を計算に使い、配列の形を読めるようになることです。

    ### この画面の使い方

    コードを入力する区切りを**セル**と呼びます。セルを編集し、右上の ▶ で実行します（`Ctrl+Enter`、Macでは `⌘+Enter` でも実行できます）。結果はセルの下に出ます。

    marimoでは、あるセルが使う変数を変更すると、関係するセルも再計算されます。
    同じ変数を別のセルで再定義せず、**元のセルを書き換えて**ください。
    実行待ちのセルがあれば、そのセルも実行してください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. 表示と計算

    `print(...)` は、括弧の中の値を表示する関数です。
    `'Hello, world!'` のように引用符で囲んだものは**文字列**、`2` や `3` は数値です。
    引用符は文字列の境界なので、表示には含まれません。
    """)
    return


@app.cell
def _():
    print('Hello, world!')
    return


@app.cell
def _():
    print(2 + 3)  # 足し算の結果を表示
    # print(100)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    文字の出力は `Hello, world!`、足し算の結果は `5` です。
    文字列の外に書いた `#` から行末までは**コメント（comment）**です。
    `print(2 + 3)` の右側にも、何をする行かをコメントで書いています。

    Pythonは行末のセミコロンを必要としません。
    掛け算は `*`、割り算は `/`、累乗は `**` で書きます。
    例えば `2 ** 3` は $2^3 = 8$ です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. 変数：値に名前を付ける

    `=` は右辺の値を左辺の名前に代入します。数学の等式と違い、左右は交換できません。
    """)
    return


@app.cell
def _():
    x = 3
    y = 2 * x + 1
    print(y)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `x` は `3` なので、`2 * 3 + 1` を計算して `y` は `7` になります。
    上のセルの `x = 3` を `x = 5` に変えて実行すると、出力は `11` になります。
    試したら `3` に戻せます。後の章では、このような式で入力から予測を計算します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. リスト：複数の値をまとめる

    `[...]` で値をカンマ区切りに並べると**リスト**になります。
    一つ取り出すときも角括弧を使い、位置を表す整数（添字）を指定します。
    添字は **0から**始まります。
    """)
    return


@app.cell
def _():
    values = [1, 2, 3]
    print(values[0])
    print(values[2])
    print(len(values))
    return (values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    順に `1`、`3`、`3` が表示されます。`len(values)` は要素の個数を返します。
    3個の要素の添字は `0, 1, 2` なので、`values[3]` は範囲外のエラーになります。

    リストの中にリストを入れると、表のように行と列で値を読めます。
    """)
    return


@app.cell
def _():
    rows = [[10, 20], [30, 40]]
    print(rows[0])
    print(rows[1][0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `rows[0]` は先頭の行 `[10, 20]` です。
    `rows[1][0]` は2行目の `[30, 40]` を取り出してから、その先頭の `30` を取り出します。
    この例では「行、列」の順で、どちらも0から数えます。
    後の章の配列を読むときにも、この位置の考え方を使います。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. 条件分岐：インデントで範囲を表す

    `if` の後ろに条件を書き、行末に `:` を付けます。
    条件が成り立つ場合の処理は、次の行からスペース4個分下げます。
    この字下げを**インデント**と呼びます。Pythonでは処理のまとまりを決める文法です。
    `elif` は前の条件が成り立たなかったときに調べる次の条件、
    `else` はどの条件にも当てはまらない場合です。上から調べ、最初に成り立った分岐だけを実行します。
    """)
    return


@app.cell
def _():
    number = -2
    if number > 0:
        print('正の数')
    elif number == 0:
        print('ゼロ')
    else:
        print('負の数')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `-2` は0より大きくも0と等しくもないので、出力は `負の数` です。
    `number` を `0` にすると `ゼロ`、`2` にすると `正の数` が出ます。
    値が等しいかを比べるのは `==`、代入は `=` です。
    `number > 0` のような条件式の値は `True`（真）か `False`（偽）になります。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. ループ：処理を繰り返す

    ### `for`：要素を順番に取り出す

    `for 名前 in リスト:` は、リストの要素を順にその名前で受け取り、
    インデントした部分を繰り返します。ここでは先ほどの `values = [1, 2, 3]` を使います。
    """)
    return


@app.cell
def _(values):
    for _value in values:
        print(2 * _value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `_value` に `1`、`2`、`3` が順に入り、出力は1行ずつ `2`、`4`、`6` です。
    繰り返すのは字下げした `print(...)` の行です。
    `values` の元のセルで最後に `4` を追加すると、この出力にも `8` が加わります。

    先頭の `_` は、marimoでは「このセルだけで使う名前」にするためのものです。
    Python一般でループの変数名に `_` が必須という意味ではありません。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### `while`：条件が成り立つ間繰り返す

    `while 条件:` は、条件が `True` の間、インデントした部分を繰り返します。
    次の例では残り回数を表示し、繰り返すたびに1ずつ減らします。
    """)
    return


@app.cell
def _():
    _remaining = 3
    while _remaining > 0:
        print(_remaining)
        _remaining = _remaining - 1
    print('スタート！')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    出力は `3`、`2`、`1`、`スタート！` の順です。
    `_remaining` が `0` になると条件が `False` になり、ループを抜けます。
    条件がいつまでも `True` のままだと繰り返しが終わらないため、
    `while` では条件を変化させる処理があることを確認します。

    要素を順に処理するときは `for`、条件が成り立つ間続けるときは `while` を使います。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. 関数：処理に名前を付ける

    同じ計算を別の値にも使いたいときは、処理を関数にまとめます。
    ここでは「受け取った値を2倍する」関数を作ります。
    `def` で関数を定義します。括弧内の `value` は入力を受け取る**引数**です。
    字下げした部分が処理で、`return` で計算結果（**戻り値**）を呼び出し元へ返します。
    """)
    return


@app.cell
def _():
    def double(value):
        result = 2 * value
        return result

    doubled = double(5)
    print(doubled)
    print(double(8))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    定義しただけでは関数の中身は実行されません。`double(5)` と呼ぶと `value` に `5` が入り、
    `result` は `10` になります。その戻り値を `doubled` に代入し、表示しています。
    続く `double(8)` では同じ処理に `8` を渡すので、`16` が表示されます。
    出力は順に `10` と `16` です。計算の中身をもう一度書かずに、入力を変えて使えます。
    `value` と `result` は関数の中で使う名前です。
    `return` は値を返す操作、`print` は表示する操作です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. import：用意された機能を使う

    **ライブラリ**は再利用できる機能の集まり、**モジュール**は `import` で読み込む単位です。
    `import` は、ほかの場所にある機能をこのコードから使えるようにします。

    `math` はPythonに付属する**標準ライブラリ**です。追加インストールは必要ありません。
    """)
    return


@app.cell
def _():
    import math

    print(math.sqrt(9))
    print(math.pi)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `math.sqrt(9)` は平方根を求める関数を呼び、`3.0` を返します。
    `math.pi` は円周率を表す値で、約 `3.14159` です。
    `math.sqrt` は「`math` の中の `sqrt`」、`math.pi` は「`math` の中の `pi`」です。
    この `.` は、モジュールに含まれる名前を指定しています。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. 属性とメソッド：データの情報を読む・用意された処理を使う

    ### メソッド：持ち物の中に薬がいくつあるか数える

    ゲームの持ち物をリストで表します。`'薬'` の個数を知りたいとき、
    ここまでに出てきた `for` と `if` で書くと、次のようになります。
    """)
    return


@app.cell
def _():
    inventory = ['薬', '鍵', '薬', '地図']
    _medicine_count = 0
    for _item in inventory:
        if _item == '薬':
            _medicine_count = _medicine_count + 1
    print(_medicine_count)
    return (inventory,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    出力は `2` です。持ち物を一つずつ調べ、薬を見つけるたびに個数を1増やしています。

    リストには、指定した値の個数を数える `count` という処理が用意されています。
    これを使えば、数える部分は1行で書けます。
    """)
    return


@app.cell
def _(inventory):
    medicine_count = inventory.count('薬')
    print(medicine_count)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    こちらも出力は `2` です。`inventory.count('薬')` は、
    「`inventory` の中にある `'薬'` を数える」と読みます。

    - `inventory`：処理する対象のリスト
    - `.count`：そのリストに用意されている「数える」処理
    - `('薬')`：処理を呼び出し、何を数えるかを引数で渡す

    このように、**対象のデータに結び付いた関数をメソッド**と呼びます。
    普通の関数で `double(5)` と書いたのに対し、ここでは `対象.メソッド(引数)` と書いています。
    戻り値を変数に受け取れる点は同じです。

    `count()` を使うと、個数を0から始める処理や、ループ・条件分岐を自分で書かずに済みます。
    「何を数えたいか」がコードに直接表れます。
    最初のリストに `'薬'` を一つ追加すると、どちらの出力も `3` になります。

    ### 属性：日付から「年」を取り出す

    Pythonでは、リストや日付などの値を**オブジェクト**と呼びます。
    それぞれの種類に応じた情報や処理が用意されています。
    ここでは、レポートの提出日を日付のデータとして扱います。
    `datetime` は日付や時刻を扱う標準ライブラリで、
    `datetime.date(2026, 4, 7)` は2026年4月7日を表す日付を作ります。
    """)
    return


@app.cell
def _():
    import datetime

    deadline = datetime.date(2026, 4, 7)
    print(deadline.year)
    print(deadline.isoformat())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    出力は順に `2026` と `2026-04-07` です。同じ `deadline` に対して、
    情報を読む書き方と、処理を呼ぶ書き方を比べてみましょう。

    | コード | していること | 得られる値 |
    |---|---|---|
    | `deadline.year` | 日付が持つ「年」の情報を読む | 整数の `2026` |
    | `deadline.isoformat()` | 日付を `年-月-日` の形式の文字列にする処理を呼ぶ | 文字列の `'2026-04-07'` |

    **オブジェクトに結び付いた値などを属性**と呼びます。
    `year` は日付の「年」を表す属性です。
    ここでは値を読むので `()` を付けません。
    一方、`isoformat` はメソッドです。渡す引数がなくても、処理を呼ぶための `()` を付けます。
    月と日を2桁にそろえたり、ハイフンでつないだりする処理は、このメソッドに任せられます。

    使える属性やメソッドはデータの種類によって決まります。
    今回は用意されたものを使いましたが、自分で作るための**クラス**の定義は05で扱います。

    ゲームのキャラクターでたとえるなら、HPという情報を読むのが属性、
    回復などの処理を呼ぶのがメソッドです。
    次の描画例の `ax.plot(...)` も、描画先の `ax` に用意されたメソッドを呼んでいます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9. NumPy：数の並びをまとめて計算する

    機械学習では、何人分・何枚分ものデータに同じ計算をします。
    **NumPy**は、数の並び（**配列**）をまとめて計算するための追加ライブラリです。
    この教材の環境には導入済みなので、`import` するだけで使えます。

    `import numpy as np` の `as np` は、長いモジュール名に `np` という別名を付けます。
    `np.array([...])` で、リストから配列を作ります。
    """)
    return


@app.cell
def _():
    import numpy as np

    hours = np.array([1.0, 2.0, 3.0])
    print(hours * 2)
    print(hours + 1)
    return hours, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    出力は `[2. 4. 6.]` と `[2. 3. 4.]` です。
    `* 2` や `+ 1` が、**各要素に**同じように計算されます。
    5節では `for` で一つずつ2倍しましたが、配列ならループを書かずに済みます。
    `2.` の点は、小数（浮動小数点数）であることを表しています。

    リストでは、同じ書き方が別の意味になります。`[1, 2, 3] * 2` はリストを2回つなげた
    `[1, 2, 3, 1, 2, 3]` です。数としてまとめて計算したいときは配列を使います。

    ### 予測と誤差を、3人分まとめて計算する

    02では、入力 $x$ から数値を $\hat y = wx + b$ の形の式で予測します。
    ここでは小さな例として、学習時間1・2・3時間の3人の得点を予測します。次のセルは、3人分の予測を一度に計算し、
    実際の得点との差（誤差）と、その二乗の平均を求めています。
    """)
    return


@app.cell
def _(hours, np):
    w = 1.5
    b = 0.5
    predictions = w * hours + b
    scores = np.array([2.0, 3.0, 5.0])
    errors = predictions - scores
    print(predictions)
    print(errors)
    print(np.mean(errors ** 2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    予測は `[2.  3.5 5. ]`、誤差は `[0.  0.5 0. ]` です。
    最後の行は誤差を二乗して平均した値で、約 `0.0833` です。
    `np.mean` は配列の平均を返す関数です。この「二乗誤差の平均」は、02で**MSE**として詳しく扱います。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 配列の形（shape）

    表のように行と列を持つ配列も作れます。
    `.shape` は配列の**形**を表す属性で、各方向にいくつ要素があるかを示します。
    """)
    return


@app.cell
def _(np):
    table = np.array([[1, 2, 3], [4, 5, 6]])
    print(table.shape)
    print(table[1, 0])
    print(table.mean(axis=0))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    `table.shape` は `(2, 3)` で、「2行3列」という意味です。
    `table[1, 0]` は2行目の先頭の `4` です。3節の `rows[1][0]` と同じく、行・列の順に0から数えます。
    `table.mean(axis=0)` は縦方向に平均を取り、列ごとの平均 `[2.5 3.5 4.5]` を返します。

    機械学習では、この形を読む場面がとても多くあります。
    例えば05では、28×28画素の白黒画像128枚をまとめて、形が `[128, 1, 28, 28]` の配列として扱います。
    **形が合わない配列同士を計算するとエラーになる**ことが多いので、
    うまく動かないときは、まず `.shape` を表示して確かめます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 10. Matplotlibで3点を結ぶ

    Matplotlibはグラフを描く**追加ライブラリ**です。NumPyと同じく、この環境には導入済みです。

    `import matplotlib.pyplot as plt` の `as plt` も、長いモジュール名に別名を付けています。
    `plt.subplots()` が返す二つの値を `fig, ax` で受け取ります。
    `fig` は図全体、`ax` はグラフを描く領域です。
    `ax.plot(...)` はその領域の描画メソッドを呼びます。
    最初のリストが横軸、次のリストが縦軸の値で、`marker='o'` は点を丸印にする指定です。
    このように `名前=値` で渡す指定をキーワード引数と呼びます。
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [2, 4, 6], marker='o')
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    初期状態では $(1, 2)$、$(2, 4)$、$(3, 6)$ を結ぶ直線が表示されます。
    最後の `fig` は、図をセルの出力として表示するための式です。
    縦軸のリストだけを `[2, 5, 6]` に変えると、中央の点が上がって折れ線になります。
    横軸と縦軸のリストは同じ長さにします。元に戻すには `[2, 4, 6]` に戻します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 11. Pythonファイルをターミナルで実行する

    今見ているmarimoは、コードと説明・結果を一緒に読むための道具です。
    Pythonのコードを `.py` ファイルに保存して、ターミナルから実行する方法もあります。
    この教材の `01-python/hello.py` には、次のコードが入っています。

    ```python
    print('Hello, world!')
    print(2 + 3)
    ```

    **別のターミナルを開き、教材ルートで**ファイル名を指定して実行します。

    ```sh
    uv run python 01-python/hello.py
    ```

    ターミナルには次のように表示されます。

    ```text
    Hello, world!
    5
    ```

    Pythonはファイルのコードを上から順に実行します。`print(...)` の結果が、ターミナルに1行ずつ表示されています。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 確認問題

    迷ったら、コードを実行して確かめてください。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_index=mo.ui.radio(['3が表示される','エラーになる','何も表示されない'],label='**Q1.** `values = [1, 2, 3]` のとき、`print(values[3])` を実行するとどうなりますか？')
    quiz_index
    return (quiz_index,)


@app.cell(hide_code=True)
def _(check, quiz_index):
    check(quiz_index,'エラーになる',
          '添字は0から数えるので、3個の要素の添字は `0, 1, 2` です。`values[3]` は範囲外なので `IndexError` になります。最後の要素は `values[2]` です。')
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_array=mo.ui.radio(['[2 4 6]','[1 2 3 1 2 3]','エラーになる'],label='**Q2.** `np.array([1, 2, 3]) * 2` の結果はどれですか？')
    quiz_array
    return (quiz_array,)


@app.cell(hide_code=True)
def _(check, quiz_array):
    check(quiz_array,'[2 4 6]',
          'NumPyの配列では、`* 2` が各要素に計算されます。`[1, 2, 3, 1, 2, 3]` になるのは、配列ではなくリストの `[1, 2, 3] * 2` のときです。')
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_shape=mo.ui.radio(['1枚','28枚','128枚'],label='**Q3.** 白黒画像を `[画像の枚数, チャンネル数, 高さ, 幅]` の順に並べた配列の `.shape` が `(128, 1, 28, 28)` です。画像は何枚ありますか？')
    quiz_shape
    return (quiz_shape,)


@app.cell(hide_code=True)
def _(check, quiz_shape):
    check(quiz_shape,'128枚',
          '先頭の軸が画像の枚数、次が色のチャンネル数（白黒なので1）、残りの `28, 28` が縦・横の画素数です。')
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_return = mo.ui.radio(
        ['計算結果を返すので、double(5) + 3 は13になる',
         '画面に10を表示するだけなので、double(5) + 3 はエラーになる',
         '関数を定義しただけで10が表示され、呼び出すと13になる'],
        label='**Q4.** `double` が `return 2 * value` を実行する関数です。`double(5) + 3` はどうなりますか？')
    quiz_return
    return (quiz_return,)


@app.cell(hide_code=True)
def _(check, quiz_return):
    check(quiz_return, '計算結果を返すので、double(5) + 3 は13になる',
          '`return` で返した10を、その後の `+ 3` に使えます。`print(10)` は表示する操作で、計算に使う値を返す操作とは異なります。')
    return


@app.cell(hide_code=True)
def _(mo):
    quiz_member = mo.ui.radio(
        ['table.shape で形を読み、inventory.count(\'薬\') で個数を数える',
         'table.shape() で形を読み、inventory.count で個数を数える',
         'table.shape と inventory.count はどちらも括弧なしで結果を読む'],
        label='**Q5.** 配列 `table` の形と、リスト `inventory` にある「薬」の個数を知りたいとき、どの書き方ですか？')
    quiz_member
    return (quiz_member,)


@app.cell(hide_code=True)
def _(check, quiz_member):
    check(quiz_member, 'table.shape で形を読み、inventory.count(\'薬\') で個数を数える',
          '`.shape` は形の情報を読む属性なので括弧は付けません。`.count(\'薬\')` は数える処理を呼ぶメソッドなので、調べたい値を括弧に入れます。')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## この章で使った書き方と次の章

    - `values[0]`：リストから位置を指定して値を取り出す。
    - `if`・`for`・`while`：条件や繰り返しに応じて処理を進める。インデントが処理の範囲を決める。
    - `double(5)`：関数に入力を渡し、戻り値を受け取る。
    - `deadline.year`・`inventory.count('薬')`：属性の値を読む・メソッドを呼ぶ。
    - `import math`：モジュールを読み込み、用意された機能を使えるようにする。
    - `np.array(...)`・`.shape`：配列をまとめて計算し、形を確かめる。

    次の02では、数値を予測する「回帰」を扱います。ここで計算したMSEも、予測のずれを測るために使います。

    **この章の確認：** 戻り値と表示の違い、属性とメソッドの違いを自分の言葉で説明し、配列の形を読めれば完了です。
    """)
    return


if __name__ == "__main__":
    app.run()
