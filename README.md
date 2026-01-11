# Semiconductor Quality Prediction (VM Process)

**VM工程における品質（OV: 欠陥数）予測モデル**

本リポジトリは、大学の「生産管理」講義課題における最終成果物です。
時系列データの特性を考慮した厳密な検証（Rolling TimeSplit）と、実務運用を想定した「説明変数の最小化」を目的としています。

👉 **[最終レポート本文はこちら (REPORT.md)](REPORT.md)**

---

## 📂 ディレクトリ構成

| ファイル/ディレクトリ | 役割 |
| :--- | :--- |
| `REPORT.md` | **最終レポート本文**（検証結果、考察、モデル選定根拠） |
| `reproduce_final_report.py` | **再現スクリプト**。これを実行するとレポートの数値・図表が生成されます。 |
| `vm_quality/` | モデルの実装コード（前処理、特徴量生成、評価ロジック） |
| `docs/images/` | レポートで使用している図表画像 |
| `artifacts/` | スクリプト実行時に生成される成果物（CSV, PNGなど） |

## 🚀 実行手順 (Reproducibility)

本リポジトリのコードを実行し、レポートの結果を再現する手順は以下の通りです。

### 1. データの準備
プロジェクトルートに `data` ディレクトリを作成し、課題データの `kadai.xlsx` を配置してください。
（※データファイルはGit管理外です）

```bash
mkdir data
# data/kadai.xlsx を配置
```

### 2. 環境構築
Python 3.x 環境が必要です。必要なライブラリをインストールしてください。

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl
```

### 3. スクリプト実行
以下のコマンドで、実験の全工程（CV検証、最終テスト、Ablation Study、ドリフト実験）が一括実行されます。

```bash
python3 reproduce_final_report.py
```

### 4. 結果の確認
実行が完了すると `artifacts/` ディレクトリ内に以下のファイルが生成されます。

*   `metrics.csv`: 最終的なRMSEスコア詳細
*   `experiments.csv`: ドリフト対策実験の結果一覧
*   `coefficients.csv`: 最終モデルの係数
*   `*.png`: 時系列推移、残差プロットなどの可視化画像

---

## 🛠 実装のポイント

*   **No Leakage (DataGuard)**: 未来のデータが学習に混入しないよう、`vm_quality/data.py` で厳密な時刻チェックを行っています。
*   **Stability**: ドリフト（環境変化）に強いモデルを作るため、交互作用項（Interaction）を排除し、単純な時間項と安定した4つの監視パラメータのみを使用しています。
*   **Refactor**: `train_and_predict` 関数により、CVと本番予測のロジックを統一し、実装ミスを防いでいます。
