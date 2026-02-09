# FastTracker 可視化ツール - 使用ガイド

## 📋 概要

FastTrackerには6つの可視化ツールが用意されており、シミュレーション結果を様々な角度から分析できます。

## 🚀 クイックスタート

### 前提条件

Pythonライブラリのインストール:
```bash
# 基本ライブラリ（必須）
pip install pandas numpy matplotlib plotly

# リアルタイムプレイヤー用（オプション）
pip install pyqtgraph PyQt5
```

### 基本的な使用フロー

1. **シミュレーション実行**
   ```bash
   fasttracker.exe --scenario ballistic --num-targets 1000 --duration 10
   ```

2. **可視化の生成**
   - すべての可視化を一括生成: `visualize_all.bat` をダブルクリック
   - または、個別に実行（下記参照）

---

## 📊 可視化ツール一覧

### 1. パフォーマンスダッシュボード
**ファイル**: `visualize_dashboard.bat`

**機能**:
- FPS（処理速度）の時系列
- 処理時間の推移
- トラック数の変化
- 観測数 vs トラック数
- 位置誤差とOSPA距離
- 統計サマリー

**出力**: `dashboard.png` (高解像度: 300dpi)

**使い方**:
```bash
# バッチファイル実行
visualize_dashboard.bat

# または直接Pythonで実行
python -m python.visualization.performance_dashboard
```

---

### 2. トラック品質レポート
**ファイル**: `visualize_quality.bat`

**機能**:
- トラック連続性の分析
- 確定率の推移
- トラック寿命の分布
- トラック状態の分布
- False Positive/Negative分析
- 品質メトリクスのサマリー

**出力**:
- `quality_report.png` (図)
- `quality_report.txt` (テキストレポート)

**使い方**:
```bash
# バッチファイル実行
visualize_quality.bat

# または直接Pythonで実行
python -m python.visualization.track_quality_report
```

---

### 3. 3D軌道ビューア
**ファイル**: `visualize_3d.bat`

**機能**:
- インタラクティブな3D軌道表示
- 3つのカラーリングモード:
  - **State**: トラック状態別（Tentative/Confirmed/Lost）
  - **Track ID**: トラックID別
  - **Model Probability**: IMM優勢モデル別
- 真値軌道の表示（破線）
- ズーム、回転、パン操作

**出力**: `trajectory_3d.html` (ブラウザで開く)

**使い方**:
```bash
# バッチファイル実行（対話型）
visualize_3d.bat

# または直接Pythonで実行
python -m python.visualization.trajectory_3d --color-by state
python -m python.visualization.trajectory_3d --color-by model_prob --no-ground-truth
```

**高度な使い方**:
```bash
# 特定の時間範囲のみ表示
python -m python.visualization.trajectory_3d --time-start 2.0 --time-end 8.0

# 特定のトラックを比較
python -m python.visualization.trajectory_3d --compare 1,2,3
```

---

### 4. IMM分析ツール
**ファイル**: `visualize_imm.bat`

**機能**:
- IMMモデル確率の時系列（積み上げグラフ）
- 加速度プロファイル
- 優勢モデルの切り替え可視化
- モデル統計レポート:
  - 各モデルでの滞在時間
  - モデル切替回数
  - 平均加速度

**出力**:
- `imm_analysis.png` (図)
- `imm_stats.txt` (統計レポート)

**使い方**:
```bash
# バッチファイル実行（対話型）
visualize_imm.bat

# または直接Pythonで実行
# 全トラック分析
python -m python.visualization.imm_analyzer

# 特定トラック分析
python -m python.visualization.imm_analyzer --track-id 5
```

---

### 5. リアルタイム軌道プレイヤー
**ファイル**: `visualize_player.bat`

**機能**:
- GUIベースのリアルタイム再生
- 再生コントロール（Play/Pause/Stop）
- 速度調整（0.25x～5x）
- 軌跡トレイル表示
- 速度ベクトル矢印
- 真値との比較

**使い方**:
```bash
# バッチファイル実行（対話型）
visualize_player.bat

# または直接Pythonで実行
python -m python.visualization.trajectory_player --speed 2.0 --trail 50
```

**操作方法**:
- **Play/Pause**: 再生/一時停止の切り替え
- **Stop**: 最初のフレームに戻る
- **Slider**: 特定のフレームにジャンプ
- **Speed**: 再生速度を変更

---

### 6. 一括可視化
**ファイル**: `visualize_all.bat`

**機能**:
- 上記1～4の可視化を一括生成（プレイヤーは除く）
- タイムスタンプ付きファイル名で保存
- エラーがあっても継続実行

**出力**: 以下のファイルが`YYYYMMDD_HHMMSS`のタイムスタンプ付きで生成
- `dashboard_<timestamp>.png`
- `quality_report_<timestamp>.png`
- `quality_report_<timestamp>.txt`
- `trajectory_3d_<timestamp>.html`
- `imm_analysis_<timestamp>.png`
- `imm_stats_<timestamp>.txt`

**使い方**:
```bash
# バッチファイル実行
visualize_all.bat
```

---

## 🔧 トラブルシューティング

### エラー: "results.csv not found"
**原因**: シミュレーションがまだ実行されていない
**解決策**:
```bash
fasttracker.exe --scenario default --num-targets 100 --duration 5
```

### エラー: "ModuleNotFoundError: No module named 'plotly'"
**原因**: 必要なPythonライブラリがインストールされていない
**解決策**:
```bash
pip install pandas numpy matplotlib plotly pyqtgraph PyQt5
```

### エラー: "No IMM data found"
**原因**: IMM分析に必要なモデル確率データがtrack_details.csvに含まれていない
**解決策**: シミュレーションがIMM有効で実行されているか確認

### プレイヤーが重い
**原因**: トラック数が多すぎる（1000+）
**解決策**:
- 再生速度を遅くする（0.5x）
- トレイル長を短くする（`--trail 10`）
- より少ないトラック数でテスト実行

---

## 📁 必要なファイル

各可視化ツールが必要とするCSVファイル:

| ツール | results.csv | evaluation_results.csv | track_details.csv | ground_truth.csv |
|--------|-------------|------------------------|-------------------|------------------|
| Dashboard | ✅ 必須 | ⚠️ 推奨 | - | - |
| Quality Report | ✅ 必須 | ⚠️ 推奨 | ⚠️ 推奨 | - |
| 3D Viewer | ⚠️ 推奨 | - | ✅ 必須 | ⚠️ 推奨 |
| IMM Analyzer | - | - | ✅ 必須 | - |
| Player | ✅ 必須 | - | ✅ 必須 | ⚠️ 推奨 |

**凡例**:
- ✅ 必須: ツールが動作するために必要
- ⚠️ 推奨: なくても動作するが、一部機能が制限される
- `-`: 不要

---

## 💡 使用例

### 例1: 弾道ミサイルシナリオの完全解析
```bash
# 1. シミュレーション実行
fasttracker.exe --scenario ballistic --num-targets 100 --duration 10

# 2. すべての可視化を生成
visualize_all.bat

# 3. リアルタイムで確認
visualize_player.bat
```

### 例2: 特定トラックの詳細分析
```bash
# 1. 3D軌道で興味深いトラックを発見
visualize_3d.bat
# → トラックID 42が面白い動きをしている

# 2. そのトラックのIMM分析
python -m python.visualization.imm_analyzer --track-id 42 --output-plot track42_imm.png
```

### 例3: 論文用の高品質な図
```bash
# 高解像度ダッシュボード（デフォルト300dpi）
python -m python.visualization.performance_dashboard --output paper_dashboard.png

# 特定の時間範囲の3D可視化
python -m python.visualization.trajectory_3d --time-start 5.0 --time-end 8.0 --output paper_3d.html
```

---

## 🎨 カスタマイズ

### 出力ファイル名の変更
```bash
python -m python.visualization.performance_dashboard --output my_dashboard.png
```

### カラースキームの選択
```bash
# State別（緑=Confirmed、黄=Tentative、赤=Lost）
python -m python.visualization.trajectory_3d --color-by state

# IMM優勢モデル別（青=CV、赤=High-Accel、緑=Med-Accel）
python -m python.visualization.trajectory_3d --color-by model_prob
```

### 真値表示のオン/オフ
```bash
# 真値を非表示
python -m python.visualization.trajectory_3d --no-ground-truth
```

---

## 📚 詳細情報

各Pythonモジュールのヘルプ:
```bash
python -m python.visualization.performance_dashboard --help
python -m python.visualization.track_quality_report --help
python -m python.visualization.trajectory_3d --help
python -m python.visualization.imm_analyzer --help
python -m python.visualization.trajectory_player --help
```

---

## 🐛 問題報告

可視化ツールに問題がある場合は、以下の情報と共に報告してください:
- 使用したバッチファイルまたはコマンド
- エラーメッセージ全文
- Pythonバージョン (`python --version`)
- インストール済みライブラリ (`pip list`)
