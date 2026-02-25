#データの読み取り
import pandas as pd
url = 'http://blog.game-de.com/pokedata/pokemon-data/'
dfs = pd.read_html(url, encoding='utf-8')
df=dfs[0].copy()

#データの加工
import numpy as np
#謎の空白
df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
# '重さ (kg)'カラムから数字部分を抽出し、float型に変換
df['重さ (kg)'] = df['重さ (kg)'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
# カラム名をより分かりやすい名前に変更
df = df.rename(columns={
    '計': '合計種族値',
    '高さ (m)':'高さ',
    '重さ (kg)':'重さ',
    '捕 獲':'捕獲率',
    '性別 ♂:♀': '性別',
    '経 験 値':'経験値',
    'な つ き':'初期なつき度'
})
def calculate_male_ratio(value):
    val_str = str(value).strip()

    # 1. 性別不明の場合
    if val_str == '不明' or val_str == 'nan':
        return np.nan

    # 2. オスのみ、メスのみの場合
    if val_str == '♂':
        return 1.0
    if val_str == '♀':
        return 0.0

    # 3. "m:w" 表記の場合
    if ':' in val_str:
        try:
            m, f = map(float, val_str.split(':'))
            return m / (m + f)
        except:
            return np.nan

    return np.nan
df['雄率'] = df['性別'].apply(calculate_male_ratio)

print(df)

#体重は高さの3乗に比例することに対しての両対数単回帰分析
#回帰診断
import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
data_log = df[['高さ', '重さ']]
data_log['log_height'] = np.log(data_log['高さ'])
data_log['log_weight'] = np.log(data_log['重さ'])
y = data_log['log_weight']
X = data_log['log_height']
X_const = sm.add_constant(X)
model_log = sm.OLS(y, X_const).fit()
print(model_log.summary())
#可視化
plt.figure(figsize=(8, 6))
plt.scatter(data_log['log_height'], data_log['log_weight'], alpha=0.5, label='実測値')
plt.plot(data_log['log_height'], model_log.predict(X_const), color='red', linewidth=2, label='回帰直線')
plt.xlabel('log(高さ)')
plt.ylabel('log(重さ)')
plt.title('対数変換後の単回帰分析: log(高さ) vs log(重さ)')
plt.legend()
plt.grid(True) # グリッド線を表示
plt.show()
#診断
names = df['名前']

def plot_lm_diagnostics_enhanced(model, labels=None):
    # 統計量の取得
    fitted = model.fittedvalues
    residuals = model.resid
    norm_residuals = model.get_influence().resid_studentized_internal
    norm_residuals_abs_sqrt = np.sqrt(np.abs(norm_residuals))
    leverage = model.get_influence().hat_matrix_diag
    cooks = model.get_influence().cooks_distance[0]

    # プロット枠の作成
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # --- 共通処理: 上位の外れ値にラベルを貼る関数 ---
    def annotate_outliers(x, y, ax, n=3):
        if labels is None: return
        # Cookの距離が大きい順にn個選ぶ
        top_indices = np.argsort(cooks)[-n:]
        for i in top_indices:
            ax.annotate(labels.iloc[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 0), textcoords='offset points',
                        color='black', fontsize=9)

    # 1. Residuals vs Fitted
    sns.residplot(x=fitted, y=residuals, lowess=True,
                  scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1}, ax=axes[0, 0])
    axes[0, 0].set_title('Residuals vs Fitted')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    # 残差の絶対値が大きい順にラベル表示
    top_resid = np.argsort(np.abs(residuals))[-3:]
    for i in top_resid:
        axes[0, 0].annotate(labels.iloc[i], xy=(fitted[i], residuals[i]))

    # 2. Normal Q-Q
    QQ = ProbPlot(norm_residuals)
    QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1, ax=axes[0, 1])
    axes[0, 1].set_title('Normal Q-Q')
    axes[0, 1].set_xlabel('Theoretical Quantiles')
    axes[0, 1].set_ylabel('Standardized Residuals')

    # 3. Scale-Location
    axes[1, 0].scatter(fitted, norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(x=fitted, y=norm_residuals_abs_sqrt, scatter=False, ci=False, lowess=True,
                line_kws={'color': 'red', 'lw': 1}, ax=axes[1, 0])
    axes[1, 0].set_title('Scale-Location')
    axes[1, 0].set_xlabel('Fitted values')
    axes[1, 0].set_ylabel('$\sqrt{|Standardized\ Residuals|}$')

    # 4. Residuals vs Leverage (Cook's Distance Contours)
    axes[1, 1].scatter(leverage, norm_residuals, alpha=0.5)
    sns.regplot(x=leverage, y=norm_residuals, scatter=False, ci=False, lowess=True,
                line_kws={'color': 'red', 'lw': 1}, ax=axes[1, 1])

    # --- Cookの距離の等高線を描画 ---
    p = len(model.params) # パラメータ数
    x_pos = np.linspace(min(leverage)+0.001, max(leverage), 100)

    # 等高線の数式: Standardized Residuals = sqrt(CookD * p * (1-h)/h)
    def r_cooks(D, x, p):
        return np.sqrt(D * p * (1 - x) / x)

    # 0.5 と 1.0 のラインを描画
    for D in [0.5, 1.0]:
        y_pos = r_cooks(D, x_pos, p)
        axes[1, 1].plot(x_pos, y_pos, linestyle='--', color='gray', linewidth=0.8)      # 上側
        axes[1, 1].plot(x_pos, -y_pos, linestyle='--', color='gray', linewidth=0.8)     # 下側
        # 端っこにテキスト表示
        if not np.isnan(y_pos[-1]):
             axes[1, 1].text(x_pos[-1], y_pos[-1], f'Cook\'s D={D}', fontsize=8, color='gray')

    axes[1, 1].set_title('Residuals vs Leverage')
    axes[1, 1].set_xlabel('Leverage')
    axes[1, 1].set_ylabel('Standardized Residuals')
    axes[1, 1].set_ylim(min(norm_residuals)-1, max(norm_residuals)+1)

    # 外れ値（影響力が大きい点）のラベル表示
    annotate_outliers(leverage, norm_residuals, axes[1, 1], n=5)

    plt.show()
plot_lm_diagnostics_enhanced(model_log, names)