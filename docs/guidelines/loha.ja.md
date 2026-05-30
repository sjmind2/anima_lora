[English](loha.md) | **日本語**

# LoHA — Low-Rank Hadamard Product Adaptation

LoHA は LyCORIS 系アダプタの一種で、2つの低ランク行列の Hadamard（要素ごとの）積を利用し、標準 LoRA の約2倍のパラメータで実効ランク r² を実現します。カスタム autograd 関数により、Hadamard 積に対する正確な勾配が提供されます。

数式の解説（フォワード公式、次元解析、重みキーの命名規則）については [docs/methods/lycoris-variants.md](../methods/lycoris-variants.md) を参照してください。本ガイドでは使用方法と設定について説明します。

## クイックスタート

### GUI

Anima GUI のメソッドドロップダウンから **LoHA** を選択します。専用設定ファイル `configs/gui-methods/loha.toml` により、バリアント固有のパラメータが自動的に設定されます。

## 設定パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|---------|------|
| `network_dim` | 32 | ランク r（実効ランク ≈ r²） |
| `network_alpha` | 16 | LoRA alpha。**スケール = alpha / dim** — 推奨範囲は 0.1–0.5。 |
| `scale_weight_norms` | 1.0 | 最大ノルムスケーリングの目標値 |
| `conv_dim` | 4 | Conv2d 層のランク（**Anima では無効** — DiT に Conv2d は存在しない） |
| `conv_alpha` | 1 | Conv2d 層の alpha（**Anima では無効**） |
| `use_tucker` | `true` | Conv2d の Tucker モード（**Anima では無効**） |

### スケールについて

スケール係数 `s = alpha / dim` は、重み更新量 ΔW の大きさを制御します。LoHA の場合：

- `dim=32, alpha=16` → スケール = 0.5（良い出発点）
- `dim=32, alpha=8` → スケール = 0.25（より控えめ）
- `dim=16, alpha=8` → スケール = 0.5（同じスケール、低い実効ランク）

推奨スケール範囲：**0.1–0.5**。スケールが高いほど適応力は強くなりますが、不安定になるリスクもあります。

## LoKR との比較

| 項目 | LoHA | LoKR |
|------|------|------|
| 主要演算 | Hadamard（要素ごとの）積 | Kronecker 積 |
| 実効ランク | r² | rank(W1) × rank(W2)（適応的） |
| パラメータ数（同じ r） | LoRA の約2倍 | 適応的；LoRA より少ない場合もあり |
| DoRA 対応 | 非対応 | 対応（`weight_decompose=true`） |
| カスタム autograd | `HadaWeight` / `HadaWeightTucker` | `KronLinearFn` / `KronLinearTwoStageFn` |
| 適している場面 | 与えられた r から高い実効ランクを得たい場合 | ファクタ調整による柔軟なパラメータ数の制御 |

## 推論

### CLI — 静的マージ

`inference.py` は safetensors のキープレフィックス（`hada_*`）を検査して LoHA を自動検出します。重みデルタは Hadamard 積の公式を用いて計算され、ベースモデルの重みにマージされます。

LoHA チェックポイントは、通常の LoRA や LoKR ファイルと同じ `--lora_weight` リスト内に共存できます。

### ComfyUI

ComfyUI の LyCORIS ローダーノードは LoHA の重みフォーマットをネイティブでサポートしています。変換なしで直接読み込み可能です。

## 互換性

LoKR と同じ相互排他ルールが適用されます：

| 組み合わせ可否 | 備考 |
|--------------|------|
| **T-LoRA** | タイムステップランクマスキングが `make_weight` の後に適用される |
| **Spectrum** | 相互作用なし |
| **Modulation guidance** | 直交 |
| **ReFT** | 直交するサイドチャネル |
| **P-GRAFT** | カットオフステップで `network.enabled` を切り替え |
| **HydraLoRA** | **非対応** |
| **OrthoLoRA** | **非対応** |

## 推奨設定

### 標準トレーニング

```toml
network_type = "loha"
network_dim = 32
network_alpha = 16
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 4
```

### T-LoRA とタイムステップマスキングを使用する場合

```toml
network_type = "loha"
network_dim = 32
network_alpha = 16
use_timestep_mask = true
min_rank = 8
scale_weight_norms = 1.0
learning_rate = 1e-4
max_train_epochs = 6
```
