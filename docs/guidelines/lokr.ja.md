[English](lokr.md) | **日本語**

# LoKR — 低ランク Kronecker 積適応

LoKR は LyCORIS ファミリのアダプタであり、重みの次元を分解し、Kronecker 積を用いて合成します。これにより、単一のランク値ではなく分解形状に依存するパラメータ数で、構造化された高ランク近似が得られます。

数学的な詳細（順伝播の数式、次元解析、重みキーの命名規則、スカラーのベイク処理）については [docs/methods/lycoris-variants.md](../methods/lycoris-variants.md) を参照してください。本ガイドでは使用方法と設定について説明します。

## クイックスタート


### GUI

Anima GUI のメソッドドロップダウンから **LoKR** を選択します。専用設定ファイル `configs/gui-methods/lokr.toml` により、バリアント固有のパラメータが自動的に設定されます。

## 設定パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|---------|-------------|
| `network_dim` | 8 | 低ランク因子のランク |
| `network_alpha` | 8 | LoRA alpha（スケール = alpha / dim） |
| `decompose_both` | `true` | W1 と W2 の両方を低ランクペアに分解します。小さいランクではパラメータ数が大幅に削減されます。 |
| `lokr_factor` | -1 | 次元分解のターゲット因子サイズ。`-1` = 自動（推奨）。 |
| `scale_weight_norms` | 1.0 | 最大ノルムスケーリングのターゲット値 |
| `weight_decompose` | `false` | DoRA スタイルの重み分解を有効化（LoKR のみ対応 — LoHA/LoCON では使用不可） |
| `use_scalar` | `false` | 固定スカラー=1 の代わりに学習可能スカラー（ゼロ初期化）を使用 |
| `full_matrix` | `false` | フル（非分解）行列を強制 |
| `conv_dim` | 4 | Conv2d レイヤーのランク（**Anima では無効** — DiT に Conv2d は存在しません） |
| `conv_alpha` | 4 | Conv2d レイヤーの alpha（**Anima では無効**） |
| `use_tucker` | `true` | Conv2d の Tucker コア（**Anima では無効**） |

## Anima 固有の注意事項

### QKV 融合と `lokr_factor`

Anima-base-v1.0 DiT は Q/K/V を単一の `qkv_proj` Linear レイヤー（形状 `[6144, 2048]`）に融合しています。ComfyUI 互換性のためにチェックポイントを保存する際、この融合モジュールは個別の `q_proj`/`k_proj`/`v_proj` に分割する必要があります。この分割には `factorization(6144, factor)` が 3 で割り切れる `out_l` を生成することが求められます。

**推奨 `lokr_factor` 値：**

| `lokr_factor` | そのままの `out_l` | 調整後の `out_l` | チェックポイントサイズ |
|---------------|------------|------------------|-----------------|
| **-1**（自動） | 自動 | 自動 | ~10 MB |
| **6** | 6 | 6（変更なし） | ~13 MB |
| **12** | 12 | 12（変更なし） | ~7 MB |
| **24** | 24 | 24（変更なし） | ~4 MB |
| 4 | 4 | **3**（調整あり） | ~10 MB |
| 8 | 8 | **6**（調整あり） | ~13 MB |
| 16 | 16 | **12**（調整あり） | ~7 MB |

いずれの値でも正しいサイズのチェックポイントが生成されます。3 の倍数（6、12、24）である因子を使用すると、調整が不要になります。最もバランスの良い分解には `lokr_factor = -1`（自動）を使用してください。

### Anima DiT に Conv2d は存在しない

Anima-base-v1.0 DiT のすべての LoRA 対象レイヤーは `nn.Linear` です。`conv_dim`、`conv_alpha`、`use_tucker` パラメータはエラーなく受け付けられますが、効果はありません。

## 推論

### CLI — 静的マージ

`inference.py` は safetensors のキープレフィックス（`lokr_*`）を検査することで LoKR を自動検出します。重みの差分は Kronecker 積の公式を用いて計算され、ノイズ除去の前にベースモデルの重みにマージされます。

LoKR チェックポイントは、通常の LoRA ファイルと同じ `--lora_weight` リスト内に共存できます。各ファイルは独立してマージされます。

### ComfyUI

ComfyUI の LyCORIS ローダーノードは LoKR 重み形式をネイティブにサポートしています。このトレーナーが生成する safetensors ファイルは、sd-scripts / LyCORIS エコシステムと同じキー命名規則を使用しているため、変換なしで直接読み込み可能です。

## 互換性

| 組み合わせ可能 | 備考 |
|-------------|-------|
| **T-LoRA** | 再構成された重みに対する `make_weight` 後にタイムステップランクマスクが適用される |
| **Spectrum** | 相互作用なし — キャッシュ済みステップはブロック全体をスキップ |
| **モジュレーションガイダンス** | 直交 — AdaLN のみに作用 |
| **ReFT** | 直交するサイドチャネル |
| **P-GRAFT** | カットオフステップで `network.enabled` を切り替え |
| **HydraLoRA** | **非対応** — 標準的な BA 構造が必要 |
| **OrthoLoRA** | **非対応** — Cayley 再パラメータ化は標準的な BA のみに定義 |

## 推奨設定

### 小規模データセット（20 枚以下）

```toml
network_type = "lokr"
network_dim = 8
network_alpha = 8
decompose_both = true
lokr_factor = -1
scale_weight_norms = 1.0
learning_rate = 2e-5
max_train_epochs = 4
```

### 大規模データセット（100 枚以上）

```toml
network_type = "lokr"
network_dim = 16
network_alpha = 16
decompose_both = true
lokr_factor = 12
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 8
```
