[English](workflow.md) | **日本語**

# ワークフローエンジン — 自動化マルチステージ学習

ワークフローエンジンは、aiohttp（バックエンド）と Vue 3 CDN（フロントエンド）で構築された WebUI + CLI 自動学習パイプラインです。スキーマ駆動の動的フォームによる設定可能なマルチステージ学習ワークフロー、SSE によるリアルタイム進捗フィードバック、ステージ間チェックポイントの引き継ぎをサポートしています。

## インストール

### Python 依存関係

Python の依存関係はすべて `pyproject.toml` に記載されています。以下のコマンドでインストールしてください：

```bash
uv sync
```

主な依存関係：
- `aiohttp >= 3.13.5` — HTTP サーバーおよび REST API
- `pywebview >= 5.0` — デスクトップウィンドウモード（任意、未導入時はブラウザにフォールバック）

### Node.js（開発時のみ）

ワークフローフロントエンドは CDN 経由で Vue 3 を使用しています（本番利用にビルドステップは不要）。**Node.js はフロントエンドの JavaScript を変更する場合にのみ必要です。**

[nodejs.org](https://nodejs.org/)（LTS 推奨）からインストールするか、パッケージマネージャーを使用してください：

```bash
# Windows (winget)
winget install OpenJS.NodeJS.LTS

# macOS (Homebrew)
brew install node

# Linux (nvm - 推奨)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
nvm install --lts
```

フロントエンド開発では、ホットリロード付きのローカル開発サーバーを利用できます：

```bash
cd workflow/web
npx serve .    # または: python -m http.server 3000
```

### pywebview のシステム依存関係

Windows では、pywebview に **Microsoft Edge WebView2 Runtime** が必要です。Windows 10 (1903+) および Windows 11 にはプリインストールされています。未導入の場合は [Microsoft](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) からダウンロードしてください。

Linux では、pywebview に `python3-gi` または `python3-pyqt5` が必要です。詳しくは [pywebview ドキュメント](https://pywebview.flowrl.com/guide/installation.html)を参照してください。

### 起動

```bash
# デスクトップウィンドウモード（デフォルト）
python -m workflow

# ブラウザモード（pywebview 不要）
python -m workflow --no-gui

# ポートとワークフロールートを指定
python -m workflow --port 8765 --workflows-root /path/to/workflows
```

## クイックスタート：シングルステージ学習

この例では、基本的な1ステージの LoKR 学習ワークフローを作成する手順を説明します。

### 1. ワークフロー UI を起動

```bash
python -m workflow
```

デスクトップウィンドウが `http://localhost:8765` で開きます。

### 2. 新しいワークフローを作成

**"New Workflow"** をクリックし、名前を付けます（例：`my_first_training`）。

### 3. 前処理ステージを追加

1. **"Add Stage"** をクリック → **Preprocess** を選択
2. **Source directory** に `image_dataset/` フォルダを設定
3. **Bucket family** を選択 — まずは `L`（1.03 MP、品質と速度のバランスが良い）をお試しください
4. **Min pixels** はデフォルト（500,000）のままにします

前処理ステージでは、選択したバケットファミリーに合わせて画像をリサイズし、VAE 潜在表現とテキスト埋め込みをキャッシュします。

### 4. 学習ステージを追加

1. **"Add Stage"** をクリック → **Train** を選択
2. **Method** を選択 — 例：**LoKR**
3. スキーマ駆動フォームでパラメーターを設定（network_dim、learning_rate、max_train_epochs など）
4. **Dataset** フィールドは上流の Preprocess ステージの出力を自動的に参照します

### 5. 実行

**"Run"** をクリックします。ワークフローはステージを順番に実行します：

1. **Preprocess** — 画像のリサイズ、VAE 潜在表現とテキスト埋め込みのキャッシュ
2. **Train** — LoKR アダプターの学習

### 6. 学習成果物の確認

学習出力はワークフローディレクトリ以下に整理されます：

```
.anima_workflow/my_first_training/
  runs/
    20260530-120000/          ← タイムスタンプ付きの実行ディレクトリ
      preprocess_1/
        post_image_dataset/   ← リサイズ済み画像とキャッシュ
      train_1/
        output/
          *.safetensors       ← 学習済みアダプター
        command.txt           ← 実行された正確なコマンド
        config.toml           ← 解決済み設定
      status.json             ← 実行ステータスのスナップショット
      run.log                 ← 完全なログ
    latest → 20260530-120000/ ← 最新の実行へのジャンクションリンク
```

**最新のアダプターを見つける3つの方法：**

1. **`runs/latest/train_1/output/`** — `latest` ジャンクションが常に最新の実行を指しています
2. **History タブ** — 完了した任意の実行で "Open directory" ボタンをクリック
3. **System ログ** — 学習完了時に safetensors パスが表示されます

## シングルステージの詳細な使い方

### 前処理ステージ

| 設定 | 説明 |
|------|------|
| **Source directory** | 生の学習画像（`.txt` キャプションサイドカー付き）へのパス |
| **Bucket families** | 使用する解像度ファミリー。詳しくは [バケットファミリーガイド](bucket-families.ja.md) を参照してください。 |
| **Min pixels** | この画素数未満の画像はスキップされます（デフォルト：500,000） |

前処理ステージは以下の3つのサブステップを順番に実行します：
1. **Resize** — 選択したバケットファミリーに合わせて画像を拡大縮小・クロップ
2. **VAE cache** — 画像を潜在空間にエンコード
3. **TE cache** — テキストキャプションを埋め込みにエンコード

### 学習ステージ

学習ステージは、選択したメソッドに応じて変化するスキーマ駆動フォームを表示します：

- **Method セレクター** — LoRA、LoKR、LoHA などを切り替えるドロップダウン
- **共通パラメーター** — 学習率、エポック数、バッチサイズ、オプティマイザー
- **メソッド固有パラメーター** — 例：LoKR の `lokr_factor`、LoRA の `network_dim`

フォームは `workflow/schemas/train_{method}.yaml` と `workflow/schemas/train_common.yaml` から生成されます。

## マルチステージでの使い方

マルチステージワークフローでは、[低解像度での事前学習後に高解像度で微調整する](bucket-families.ja.md#multi-stage-training-strategy)といった高度な学習戦略が可能です。

### マルチステージオーケストレーションの仕組み

ステージは `depends_on` 宣言に基づく**トポロジカル順序**で実行されます。スケジューラーは循環依存を検出し、エラーを報告します。

各ステージの出力は、後続ステージから以下の方法で利用できます：
- **自動参照** — システムが上流出力から `network_weights` や `datasets` を自動入力
- **プレースホルダー構文** — 設定値内の `${stage_id.output_key}` が実行時に解決

### 複数の前処理ステージ

各前処理ステージには異なる設定を指定できます：

| 設定 | Preprocess 1 | Preprocess 2 |
|------|-------------|-------------|
| **Bucket families** | `S1`（低解像度、0.26 MP） | `L`（高解像度、1.03 MP） |
| **Source directory** | `image_dataset/` | `image_dataset/`（同じでも別でも可） |

これにより、異なる解像度の2セットのキャッシュデータが、それぞれ別のサブディレクトリに生成されます。

### 複数の学習ステージ

#### `stop_epoch` — 中断と保存

学習ステージに `stop_epoch` を設定すると、指定したエポックで学習を停止し、チェックポイントが確実に保存されます：

```
stop_epoch = 6
```

これにより `max_train_epochs` と `save_every_n_epochs` が指定値に設定され、epoch-6 のチェックポイント保存直後に学習が停止します。

#### チェックポイントの引き継ぎ

ある学習ステージの後に別の学習ステージが実行される場合、以下の処理が自動的に行われます：

1. 上流ステージの `safetensors_path` 出力を見つける
2. `--network_weights` にそのパスを設定
3. LoRA の場合：`--dim_from_weights` を設定し、チェックポイントからランクを自動推論
4. LyCORIS（lokr/loha/locon）の場合：`dim_from_weights = false` を設定（次元は設定と一致している必要あり）

#### 典型的なマルチステージフロー

```
Preprocess S1 → Train S1 (epoch 6 で停止) → Preprocess L → Train L (S1 チェックポイントから)
```

1. **Preprocess S1**：S1 ファミリー（0.26 MP）でリサイズ + キャッシュ
2. **Train S1**：LoKR アダプターを学習、epoch 6 で停止
3. **Preprocess L**：L ファミリー（1.03 MP）でリサイズ + キャッシュ
4. **Train L**：S1 の epoch-6 チェックポイントから継続、S1 と L の両方のキャッシュを使用

2番目の学習ステージはプレースホルダー経由で1番目の出力を参照します：`${train_1.safetensors_path}` → 実際のパスに解決されます。

## ログビューアー

下部パネルには3つのタブがあります：

### System ログ

ワークフローレベルのイベントを表示します：ステージの開始/終了、チェックポイントの保存、エラーなど。SSE（Server-Sent Events）によりリアルタイムで更新されます。

### Script 出力

サブプロセスの標準出力を表示します：
- **TQDM プログレスバー** — ステップ数、経過時間、残り時間、メトリクス（loss、lr）を含む視覚的なプログレスバーとして解析・表示
- **ステージフィルタリング** — ドロップダウンでステージごとに出力をフィルタリング
- **自動スクロール** — 最新の出力に自動的にスクロール、スクロールロックボタンで一時停止/再開
- **バッファ制限** — ステージごとに500行、超過時は古い行から切り詰め

### 実行履歴

すべての過去の実行を新しい順に一覧表示します。各エントリには以下が表示されます：
- **タイムスタンプ** と **所要時間**
- **ステータス**：ok / stopped / error / running
- **ステージチェーン** とカラーコード付きステータスインジケーター
- **アクション**："View log" と "Open directory"

**履歴から最新の学習成果物を見つけるには：**
1. **History** タブを開く
2. 最新の実行が一番上に表示されます
3. **"Open directory"** をクリックして実行フォルダを開く
4. `{train_stage_id}/output/` に移動して `.safetensors` ファイルを見つける

または、`runs/latest` が常に最新の実行ディレクトリへのジャンクション/シムリンクとなっています。

### 検索とハイライト

ログビューアーは、表示中のすべてのログ行にわたるテキスト検索とハイライト機能をサポートしています。

## 設定

### 言語

UI はブラウザの言語設定を自動検出し、3つの言語をサポートしています：
- **English** (en)
- **中文** (zh-CN)
- **日本語** (ja)

手動で切り替えるには、右上の言語セレクターを使用してください。設定は `localStorage` に保存されます。

すべてのスキーマラベル、フィールド説明、ヘルプテキスト、選択肢ラベルは i18n オーバーレイシステムにより翻訳されます。

### モデル設定

**Settings** ダイアログでモデルパスを設定します：

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| **DiT model** | `models/diffusion_models/anima-base-v1.0.safetensors` | ベースモデルのパス |
| **Qwen3 text encoder** | `models/text_encoders/qwen_3_06b_base.safetensors` | テキストエンコーダーのパス |
| **VAE** | `models/vae/qwen_image_vae.safetensors` | VAE のパス |

パスはリポジトリルート（`ANIMA_HOME`）を基準に解決されます。環境変数 `ANIMA_DIT`、`ANIMA_VAE`、`ANIMA_TEXT_ENCODER` を設定してオーバーライドできます。

### ハードウェア設定

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| **Mixed precision** | `bf16` | 学習精度 |
| **Attention mode** | `flex` | アテンションの実装方式 |

### オーバーライドの優先順位

設定は以下の順序で適用されます（後のものが前のものをオーバーライド）：

1. **インフラデフォルト** — `library.env.resolve_under_home()` から解決
2. **インフラ設定** — `workflow.yaml` に保存されたワークフローごとの設定
3. **ステージ設定** — ステージごとの TOML オーバーライド
4. **自動導出** — `network_weights`、`datasets` など、上流出力から自動入力される値

グローバル設定（ワークフロールートなど）は、プロジェクトルートの `.anima_workflow_config.json` に保存されます。
