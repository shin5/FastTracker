"""
Build docs/manual.html with KaTeX math and embedded base64 screenshots.
Uses placeholder replacement instead of f-strings to avoid brace conflicts.
"""
import base64, os

DOCS = os.path.dirname(os.path.abspath(__file__))
SS_DIR = os.path.join(DOCS, 'screenshots')

def img_b64(filename):
    path = os.path.join(SS_DIR, filename)
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

# Pre-load all images
imgs = {}
for name in ['01_full_gui','02_map_points','03_target_panel','04_sensor_panel',
             '05_tracker_panel','06_trajectory_3d','07_tracking_3d',
             '08_evaluation','09_timeline','10_full_after_sim']:
    imgs[name] = img_b64(name + '.png')
    print(f'  Loaded {name}: {len(imgs[name])//1024} KB')

def img_html(key, alt, caption):
    tag = '<img src="data:image/png;base64,' + imgs[key] + '" alt="' + alt + '" style="width:100%;border-radius:6px;border:1px solid var(--border);">'
    return '<div class="figure">' + tag + '<div class="figure-caption">' + caption + '</div></div>'

# Read the current manual.html as base template
with open(os.path.join(DOCS, 'manual.html'), 'r', encoding='utf-8') as f:
    html = f.read()

# ============================================================
# 1. Add KaTeX CDN to <head>
# ============================================================
katex_tags = """<!-- KaTeX Math Rendering -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}]});"></script>
"""
html = html.replace('<style>', katex_tags + '<style>')

# ============================================================
# 2. Add math-block CSS
# ============================================================
math_css = """/* KaTeX display math styling */
.katex-display { margin: 16px 0 !important; }
.katex { color: var(--text) !important; }
.math-block {
    background: var(--code-bg); border: 1px solid var(--border);
    border-radius: 6px; padding: 16px 20px; margin: 14px 0;
    text-align: center;
}
"""
html = html.replace('/* Responsive */', math_css + '/* Responsive */')

# ============================================================
# 3. Replace text-based math formulas with KaTeX
# ============================================================

# Section 5.1 - State vector
old_state = '''<div class="diagram"><strong>x</strong> = [x, y, z, v<sub>x</sub>, v<sub>y</sub>, v<sub>z</sub>, a<sub>x</sub>, a<sub>y</sub>, a<sub>z</sub>]<sup>T</sup>

  x, y, z    : 位置 [m]     (目標原点のメートル座標系)
  v_x, v_y, v_z : 速度 [m/s]
  a_x, a_y, a_z : 加速度 [m/s&sup2;]</div>'''
new_state = '''<div class="math-block">
$$
\\mathbf{x} = \\begin{bmatrix} x \\\\ y \\\\ z \\\\ v_x \\\\ v_y \\\\ v_z \\\\ a_x \\\\ a_y \\\\ a_z \\end{bmatrix} \\in \\mathbb{R}^9
$$
</div>
<table>
    <tr><th>成分</th><th>意味</th><th>単位</th></tr>
    <tr><td>$ x, y, z $</td><td>位置（目標原点メートル座標系）</td><td>m</td></tr>
    <tr><td>$ v_x, v_y, v_z $</td><td>速度</td><td>m/s</td></tr>
    <tr><td>$ a_x, a_y, a_z $</td><td>加速度</td><td>m/s&sup2;</td></tr>
</table>'''
html = html.replace(old_state, new_state)

# Section 5.1 - Measurement vector
old_meas = '''<div class="diagram"><strong>z</strong> = [r, &theta;, &phi;, v<sub>r</sub>]<sup>T</sup>

  r     : 距離 [m]       (センサーからの直線距離)
  &theta;     : 方位角 [rad]   (atan2系: 0=East, &pi;/2=North)
  &phi;     : 仰角 [rad]     (水平面からの角度)
  v_r   : ドップラー速度 [m/s] (視線方向の速度成分)</div>'''
new_meas = '''<div class="math-block">
$$
\\mathbf{z} = \\begin{bmatrix} r \\\\ \\theta \\\\ \\phi \\\\ v_r \\end{bmatrix} \\in \\mathbb{R}^4
$$
</div>
<table>
    <tr><th>成分</th><th>意味</th><th>単位</th></tr>
    <tr><td>$ r $</td><td>距離（センサーからの直線距離）</td><td>m</td></tr>
    <tr><td>$ \\theta $</td><td>方位角（atan2系: 0=East, $ \\pi/2 $=North）</td><td>rad</td></tr>
    <tr><td>$ \\phi $</td><td>仰角（水平面からの角度）</td><td>rad</td></tr>
    <tr><td>$ v_r $</td><td>ドップラー速度（視線方向の速度成分）</td><td>m/s</td></tr>
</table>'''
html = html.replace(old_meas, new_meas)

# Sigma points
html = html.replace(
    '<p>UKFのシグマポイント数は <code>2n+1 = 19</code> 個です。</p>',
    '<p>UKFのシグマポイント数は $ 2n+1 = 19 $ 個（$ n = 9 $）です。</p>'
)

# Section 5.2 - Motion model predictions
old_cv = '''<td>x(t+dt) = x + v&middot;dt<br>v(t+dt) = v<br>a(t+dt) = 0</td>'''
new_cv = '''<td>$ \\mathbf{x}(t{+}\\Delta t) = \\mathbf{x} + \\mathbf{v}\\Delta t $ <br>$ \\mathbf{v}(t{+}\\Delta t) = \\mathbf{v} $ <br>$ \\mathbf{a}(t{+}\\Delta t) = \\mathbf{0} $</td>'''
html = html.replace(old_cv, new_cv)

old_ballistic = '''<td>RK4積分（重力 + 大気抗力）<br>密度モデル: &rho; = &rho;<sub>0</sub> &middot; exp(-z/H)</td>'''
new_ballistic = '''<td>RK4積分: $ \\ddot{\\mathbf{x}} = \\mathbf{g} + \\mathbf{f}_{\\text{drag}} $ <br>大気密度: $ \\rho(z) = \\rho_0 \\, e^{-z/H} $</td>'''
html = html.replace(old_ballistic, new_ballistic)

old_ct = '''<td>2D水平面旋回 + 鉛直CV<br>旋回率: &omega; = |a| / |v|</td>'''
new_ct = '''<td>2D水平旋回 + 鉛直CV <br>旋回率: $ \\omega = |\\mathbf{a}| / |\\mathbf{v}| $</td>'''
html = html.replace(old_ct, new_ct)

# UKF params
old_ukf = '''<tr><td>&alpha;</td><td>0.5</td><td>スケーリング（シグマポイントの広がり）</td></tr>
    <tr><td>&beta;</td><td>2.0</td><td>分布形状（ガウスに最適）</td></tr>
    <tr><td>&kappa;</td><td>0.0</td><td>補助スケーリング</td></tr>
    <tr><td>&lambda;</td><td>&alpha;&sup2;(n+&kappa;) - n</td><td>複合パラメータ（n=9）</td></tr>'''
new_ukf = '''<tr><td>$ \\alpha $</td><td>0.5</td><td>スケーリング（シグマポイントの広がり）</td></tr>
    <tr><td>$ \\beta $</td><td>2.0</td><td>分布形状（ガウス分布に最適）</td></tr>
    <tr><td>$ \\kappa $</td><td>0.0</td><td>補助スケーリング</td></tr>
    <tr><td>$ \\lambda $</td><td>$ \\alpha^2(n + \\kappa) - n $</td><td>複合パラメータ（$ n = 9 $）</td></tr>'''
html = html.replace(old_ukf, new_ukf)

# IMM cycle - add model probability update formula
old_imm_list = '''    <li><strong>モデル確率更新</strong> &mdash; 各モデルの観測尤度からモデル確率をベイズ更新</li>
    <li><strong>統合</strong> &mdash; モデル確率で重み付けし、最終的な状態推定・共分散を算出</li>'''
new_imm_list = '''    <li><strong>モデル確率更新</strong> &mdash; 各モデルの観測尤度 $ L_j $ からベイズ更新:</li>
</ol>
<div class="math-block">
$$
\\mu_j(k) = \\frac{L_j(k) \\cdot \\bar{\\mu}_j(k)}{\\sum_{i=1}^{3} L_i(k) \\cdot \\bar{\\mu}_i(k)}
$$
</div>
<ol start="5">
    <li><strong>統合</strong> &mdash; モデル確率 $ \\mu_j $ で重み付け統合:</li>'''
html = html.replace(old_imm_list, new_imm_list)

# Add combination formula after the IMM cycle list's closing </ol>
old_after_imm = '''</ol>

<h3 id="sec5-assoc">'''
new_after_imm = '''</ol>
<div class="math-block">
$$
\\hat{\\mathbf{x}} = \\sum_{j=1}^{3} \\mu_j \\, \\hat{\\mathbf{x}}_j
\\qquad
\\hat{P} = \\sum_{j=1}^{3} \\mu_j \\left[ P_j + (\\hat{\\mathbf{x}}_j - \\hat{\\mathbf{x}})(\\hat{\\mathbf{x}}_j - \\hat{\\mathbf{x}})^T \\right]
$$
</div>

<h3 id="sec5-assoc">'''
html = html.replace(old_after_imm, new_after_imm)

# Section 5.3 - Mahalanobis distance
old_maha = '''<div class="diagram">d<sub>M</sub> = &radic;( &sum;<sub>i</sub> ((z<sub>pred,i</sub> - z<sub>meas,i</sub>) / &sigma;<sub>i</sub>)&sup2; )</div>'''
new_maha = '''<div class="math-block">
$$
d_M = \\sqrt{ \\sum_{i=1}^{4} \\left( \\frac{ \\hat{z}_i - z_i }{ \\sigma_i } \\right)^2 }
$$
</div>
<p>ここで $ \\sigma_i $ は各観測次元のノイズ標準偏差（距離、方位角、仰角、ドップラー）です。</p>'''
html = html.replace(old_maha, new_maha)

# Remove the now-redundant paragraph
html = html.replace('<p><strong>マハラノビス距離</strong>は、各観測次元のノイズ標準偏差で正規化した距離です。</p>', '<p><strong>マハラノビス距離:</strong></p>')

# Section 5.4 - Initial state estimation (polar to cartesian)
old_init = '<p><strong>初期状態推定:</strong> 極座標の観測値をセンサー位置を基準にデカルト座標に変換し、初期速度・加速度はゼロに設定します。</p>'
new_init = '''<p><strong>初期状態推定:</strong> 極座標の観測 $ (r, \\theta, \\phi) $ からデカルト座標に変換:</p>
<div class="math-block">
$$
\\begin{aligned}
x &= r \\cos \\phi \\cos \\theta + s_x \\\\
y &= r \\cos \\phi \\sin \\theta + s_y \\\\
z &= r \\sin \\phi + s_z
\\end{aligned}
$$
</div>
<p>ここで $ (s_x, s_y, s_z) $ はセンサー位置です。初期速度・加速度はゼロに設定されます。</p>'''
html = html.replace(old_init, new_init)

# Section 5.5 - Coordinate conversion formula
old_coord_conv = '''<ul>
    <li>X軸: 東方向 [m]</li>
    <li>Y軸: 北方向 [m]</li>
    <li>Z軸: 高度 [m]</li>
</ul>

<h4>センサー相対極座標</h4>'''
new_coord_conv = '''<ul>
    <li>$ x $ 軸: 東方向 [m]</li>
    <li>$ y $ 軸: 北方向 [m]</li>
    <li>$ z $ 軸: 高度 [m]</li>
</ul>
<p>緯度・経度からの変換:</p>
<div class="math-block">
$$
\\Delta x = R \\cos(\\varphi_{\\text{ref}}) \\cdot \\Delta\\lambda, \\qquad
\\Delta y = R \\cdot \\Delta\\varphi
$$
</div>
<p>ここで $ R = 6\\text{,}371\\text{,}000 $ m（地球半径）、$ \\varphi $ は緯度、$ \\lambda $ は経度です。</p>

<h4>センサー相対極座標</h4>'''
html = html.replace(old_coord_conv, new_coord_conv)

# Add polar coord formulas
old_polar = '''<ul>
    <li>距離 (range): センサーからの直線距離 [m]</li>
    <li>方位角 (azimuth): atan2系 (0=East, &pi;/2=North) [rad]</li>
    <li>仰角 (elevation): 水平面からの角度 [rad]</li>
</ul>'''
new_polar = '''<div class="math-block">
$$
r = \\sqrt{\\Delta x^2 + \\Delta y^2 + \\Delta z^2}, \\quad
\\theta = \\operatorname{atan2}(\\Delta y,\\, \\Delta x), \\quad
\\phi = \\arcsin\\!\\left(\\frac{\\Delta z}{r}\\right)
$$
</div>'''
html = html.replace(old_polar, new_polar)

# Section 5.6 - RMSE
old_rmse = '''<div class="diagram">RMSE = &radic;( (1/N) &sum; ||x<sub>est</sub> - x<sub>true</sub>||&sup2; )</div>'''
new_rmse = '''<div class="math-block">
$$
\\text{RMSE} = \\sqrt{ \\frac{1}{N} \\sum_{i=1}^{N} \\| \\hat{\\mathbf{x}}_i - \\mathbf{x}_i^{\\text{true}} \\|^2 }
$$
</div>'''
html = html.replace(old_rmse, new_rmse)

# OSPA
old_ospa_desc = '''<ul>
    <li>カットオフ距離: トラックが真値から離れすぎた場合のペナルティ上限</li>
    <li>次数 (p): デフォルト = 2</li>
</ul>'''
new_ospa_desc = '''<div class="math-block">
$$
\\bar{d}_p^{(c)}(X, Y) = \\left( \\frac{1}{n} \\left[ \\min_{\\pi \\in \\Pi_n} \\sum_{i=1}^{m} d^{(c)}(x_i, y_{\\pi(i)})^p + c^p (n - m) \\right] \\right)^{1/p}
$$
</div>
<p>ここで $ c $ はカットオフ距離、$ p $ は次数（デフォルト = 2）、$ n = \\max(|X|, |Y|) $、$ m = \\min(|X|, |Y|) $ です。</p>'''
html = html.replace(old_ospa_desc, new_ospa_desc)

# Precision / Recall / F1 table -> formula
old_prf = '''<table>
    <tr><th>指標</th><th>定義</th><th>意味</th></tr>
    <tr><td>Precision</td><td>TP / (TP + FP)</td><td>トラックのうち正しいものの割合</td></tr>
    <tr><td>Recall</td><td>TP / (TP + FN)</td><td>真目標のうち追尾できたものの割合</td></tr>
    <tr><td>F1</td><td>2 &middot; P &middot; R / (P + R)</td><td>PrecisionとRecallの調和平均</td></tr>
</table>'''
new_prf = '''<div class="math-block">
$$
\\text{Precision} = \\frac{TP}{TP + FP}, \\qquad
\\text{Recall} = \\frac{TP}{TP + FN}, \\qquad
F_1 = \\frac{2 \\cdot P \\cdot R}{P + R}
$$
</div>'''
html = html.replace(old_prf, new_prf)

# ============================================================
# 4. Insert screenshots
# ============================================================
# After "1.2 システム構成" heading, before the diagram
html = html.replace(
    '<h3 id="sec1-features">1.1 主要機能</h3>',
    img_html('10_full_after_sim', 'FastTracker Web GUI 全体画面',
             '図1.1: FastTracker Web GUI — 追尾シミュレーション実行後の全体画面')
    + '\n\n<h3 id="sec1-features">1.1 主要機能</h3>'
)

# Section 3.1 - after layout table
html = html.replace(
    '<p>可視化パネルには4つのタブがあります。</p>',
    img_html('01_full_gui', 'FastTracker 初期画面',
             '図3.1: 初期画面 — 左上: Map、右上: Visualization、下部: Parameters &amp; Control')
    + '\n\n<p>可視化パネルには4つのタブがあります。</p>'
)

# Section 3.2 - map
html = html.replace(
    '<div class="info-box">\nセンサー位置を設定しない場合',
    img_html('02_map_points', '地図上の3点設定',
             '図3.2: 地図上に発射点（赤）・目標（青）・センサー（緑）を配置した状態')
    + '\n\n<div class="info-box">\nセンサー位置を設定しない場合'
)

# Section 3.4 - TARGET panel
html = html.replace(
    '<h4>ミサイル種類</h4>',
    img_html('03_target_panel', 'TARGET パラメータパネル',
             '図3.3: TARGET パネル — ミサイル種類、飛行パラメータ、Auto-Adjust、クラスタ設定')
    + '\n\n<h4>ミサイル種類</h4>'
)

# Section 3.5 - SENSOR panel
html = html.replace(
    '<h4>レーダー覆域</h4>',
    img_html('04_sensor_panel', 'SENSOR パラメータパネル',
             '図3.4: SENSOR パネル — レーダー覆域、性能、観測ノイズ、ビームステアリング')
    + '\n\n<h4>レーダー覆域</h4>'
)

# Section 3.6 - TRACKER panel
html = html.replace(
    '<h4>データアソシエーション</h4>',
    img_html('05_tracker_panel', 'TRACKER パラメータパネル',
             '図3.5: TRACKER パネル — アソシエーション、トラック管理、プロセスノイズ、モンテカルロ')
    + '\n\n<h4>データアソシエーション</h4>'
)

# Section 3.7 - Trajectory tab
html = html.replace(
    '<ul>\n    <li><strong>色分け</strong> &mdash; 飛行フェーズ別に着色',
    img_html('06_trajectory_3d', 'Trajectory 3D表示',
             '図3.6: Trajectory タブ — 弾道ミサイル軌道の3D表示（フェーズ別色分け）')
    + '\n<ul>\n    <li><strong>色分け</strong> &mdash; 飛行フェーズ別に着色'
)

# Section 3.7 - Tracking tab
html = html.replace(
    '<p>センサー原点（0,0,0）基準の3D座標で以下を同時表示します。</p>',
    img_html('07_tracking_3d', 'Tracking 3D表示',
             '図3.7: Tracking タブ — 真値(黄), 検出(橙), トラック(緑), クラッタ(ピンク)の3D表示')
    + '\n<p>センサー原点（0,0,0）基準の3D座標で以下を同時表示します。</p>'
)

# Section 3.7 - Evaluation tab
html = html.replace(
    '<p>追尾性能の定量評価を表示します。',
    img_html('08_evaluation', 'Evaluation 評価結果',
             '図3.8: Evaluation タブ — モンテカルロ10回の集計結果と詳細指標')
    + '\n<p>追尾性能の定量評価を表示します。'
)

# Section 3.7 - Timeline tab
html = html.replace(
    '<p>フレーム単位の詳細時系列データをマルチパネルで表示します。</p>',
    img_html('09_timeline', 'Timeline タイムライン',
             '図3.9: Timeline タブ — 目標ごとの検出・追尾状態の時系列表示')
    + '\n<p>フレーム単位の詳細時系列データをマルチパネルで表示します。</p>'
)

# ============================================================
# 5. Write output
# ============================================================
out_path = os.path.join(DOCS, 'manual.html')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

sz = os.path.getsize(out_path)
print(f'\nGenerated: {out_path}')
print(f'Size: {sz/1024:.0f} KB ({sz/1024/1024:.1f} MB)')

# Verify
import re
hrefs = re.findall(r'href="#([^"]+)"', html)
ids = re.findall(r'id="([^"]+)"', html)
missing = [h for h in hrefs if h not in ids]
if missing:
    print(f'WARNING: Broken links: {missing}')
else:
    print(f'All {len(hrefs)} internal links OK')

img_count = html.count('data:image/png;base64,')
katex_count = html.count('$$') // 2 + html.count('$ ')
print(f'Embedded images: {img_count}')
print(f'Math expressions: ~{katex_count}')
