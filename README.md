# ポケモンデータの単回帰分析 (Pokémon Linear Regression)

ポケモンの「高さ」と「重さ」のデータを用いて、Pythonで単回帰分析と回帰診断を行うスクリプトです。

## ファイルの概要

* **`poke_linearmodel1.py`**
  * ポケモンの「高さ(m)」を説明変数、「重さ(kg)」を目的変数とした通常の単回帰分析を行います。
  * 「大きいポケモンほど重い」という仮説を検証し、回帰直線と回帰診断図（Residuals vs Fitted, Normal Q-Q, Scale-Location, Residuals vs Leverage）を出力します。

* **`poke_linearmodel2.py`**
  * 「体重は高さの3乗に比例する」という仮説に基づき、両変数を対数変換した上で両対数単回帰分析を行います（`log(高さ)` vs `log(重さ)`）。
  * よりデータにフィットしたモデルの検証と、外れ値（影響力の大きいポケモン）の確認を回帰診断図を通して行います。

## 使用している主なライブラリ
* `pandas`, `numpy` (データ加工)
* `statsmodels` (単回帰分析・回帰診断)
* `matplotlib`, `japanize_matplotlib`, `seaborn` (可視化)

## データの出典
データは [ポケモン種族値データ](http://blog.game-de.com/pokedata/pokemon-data/) より取得・加工して使用しています。