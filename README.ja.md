# anima_lora

[English](README.md) | [中文](README.zh.md) | **日本語**

[Anima](https://huggingface.co/circlestone-labs/Anima) 拡散モデル（DiTベース・フローマッチング）向けの LoRA / T-LoRA 学習・推論エンジンです。

このリポジトリが重視しているのは次の4点です：

1. **高速な LoRA 学習** — コンシューマー GPU で、ブロックごとの `torch.compile` を少数の固定シェイプセット（トークンカウントファミリごとに1つのブロックグラフ）に対してエンドツーエンドで実行します。
2. **堅牢な従来手法の実装** — LoRA、OrthoLoRA、T-LoRA をスタック可能で、ロスレスに単体 DiT チェックポイントへマージできます。
3. **Anima 向けに最適化された最新手法** — Spectrum 推論、DCW & SMC-CFG サンプラー、OrthoHydraLoRA、モデュレーションガイダンスなど、それぞれ Anima のコンパイル契約に合わせてエンドツーエンドで実装されており、単なるトイポートではありません。
4. **幅広い実験的機能** — SPD、ChimeraHydra、Soft Tokens、Turbo 蒸留、ReFT、IP-Adapter、EasyControl、DirectEdit、埋め込み反転。

> **各手法の概要図**（DiT 内部構造、LoRA、OrthoLoRA、T-LoRA、HydraLoRA、ReFT、Spectrum、モデュレーション、コンパイル最適化）は [`docs/structure_images/`](docs/structure_images/) にあり、[`docs/structure/`](docs/structure/) の解説文と対になっています。

## 新着情報

| 機能 | 説明 | ガイド |
|---------|-------------|-------|
| **LoKR** | 低ランク Kronecker 積による適応 — 構造化された高ランクと適応的パラメータ数を実現 | [docs/guidelines/lokr.ja.md](docs/guidelines/lokr.ja.md) |
| **LoHA** | 低ランク Hadamard 積による適応 — 実効ランク r² を LoRA のわずか2倍のパラメータで達成 | [docs/guidelines/loha.ja.md](docs/guidelines/loha.ja.md) |
| **CAME オプティマイザ** | 完全行列の二次モーメントを置き換える因数分解オプティマイザ — 大幅なメモリ削減 | [docs/guidelines/came.ja.md](docs/guidelines/came.ja.md) |
| **バケットファミリ** | 面積による解像度バケット分け → アスペクト比マッチングをトークンカウントグループに対応させ、コンパイル性能を最適化 | [docs/guidelines/bucket-families.ja.md](docs/guidelines/bucket-families.ja.md) |
| **ワークフローエンジン** | WebUI + CLI のマルチステージ学習パイプライン。リアルタイム進捗表示とスキーマ駆動フォームを搭載 | [docs/guidelines/workflow.ja.md](docs/guidelines/workflow.ja.md) |

---

## 始め方

1行で完了 — [uv](https://astral.sh/uv) が未導入なら自動インストールし、最新リリースを取得して `uv sync` を実行します（git 不要）。インストーラはチェックサム署名付きのリリースアセットとして公開されています：

```bash
# Linux / macOS
curl -LsSf https://github.com/sorryhyun/anima_lora/releases/latest/download/install.sh | sh
```
```powershell
# Windows (PowerShell)
irm https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 | iex
```

`./anima_lora/` にインストールされます（`ANIMA_DIR` で変更可能）。Windows ではデスクトップに **"Anima LoRA GUI"** ショートカットも作成されます。

<details>
<summary><b>より安全なインストール</b> — スクリプトを事前に確認・検証する場合</summary>

各リリースには `checksums.txt`（インストーラ + ソースアーカイブの SHA-256）が同梱されています。ダウンロードして検証後、実行してください：

```bash
# Linux / macOS
curl -fLO https://github.com/sorryhyun/anima_lora/releases/latest/download/install.sh
curl -fLO https://github.com/sorryhyun/anima_lora/releases/latest/download/checksums.txt
grep install.sh checksums.txt | sha256sum -c -    # "install.sh: OK" と表示されることを確認
less install.sh                                    # 内容を確認
sh install.sh
```
```powershell
# Windows (PowerShell)
iwr https://github.com/sorryhyun/anima_lora/releases/latest/download/install.ps1 -OutFile install.ps1
iwr https://github.com/sorryhyun/anima_lora/releases/latest/download/checksums.txt -OutFile checksums.txt
(Get-FileHash install.ps1 -Algorithm SHA256).Hash.ToLower()   # checksums.txt の値と照合
notepad install.ps1                                            # 内容を確認
powershell -ExecutionPolicy Bypass -File .\install.ps1
```
</details>

**再現可能な固定バージョンインストール** — `ANIMA_VERSION` を設定すると、最新版ではなく指定タグをインストールします（既知の安定環境が必要な場合に推奨）：

```bash
ANIMA_VERSION=v1.4.0 sh install.sh       # または: $env:ANIMA_VERSION='v1.4.0'; irm ... | iex
```

その後、認証してモデルをダウンロードします：

```bash
cd anima_lora
hf auth login
make download-models      # DiT + Qwen3 TE + QwenImage VAE（+ マスク・画像条件付け用の SAM3 / MIT / PE）を models/ にダウンロード
make gui                  # 推奨 — 設定エディタ + データセットブラウザ + 学習モニタ
```

後からアップデートする場合は `make update` を使用します（リリース tarball のマージ、git 不要）。リポジトリをクローンしたい場合は [セットアップ → 手動](#manual-from-a-clone) を参照してください。

---

## 1. 高速な学習

**ピーク VRAM 13.4 GB · 1.1 s/step** で RTX 5060 Ti 単体で **rank=32・1MP 解像度の LoRA 学習** を実現 — データパイプライン・アテンション・コンパイラスタックを協調設計し、Dynamo がラン全体で少数の固定シェイプセット（トークンカウントファミリごとに1つのブロックグラフ）のみを扱うようにしています。

| 手法 | 概要 |
|---|---|
| 定数トークンバケット | バケットは2つのトークンカウントファミリ — 4032 と 4200 パッチ — に属し、各解像度がカウントを正確に埋めるため、バケット内のパディングはゼロです。フォワードはネイティブのトークンカウントで実行されるため、`torch.compile` は固有カウントごとに1つのブロックグラフ（計2つ）のみをトレースします。従来の静的パディングパスは削除されました（Flash Self-Attention にパディングが漏洩し、4200 > 4096 のためこのテーブルを実行できませんでした）。 |
| 最大パディング付きテキストエンコーダ | テキスト出力は 512 にパディングされゼロ埋めされます — 事前学習済み DiT はゼロキーをクロスアテンションのシンクとして使用するため、トリミングすると破綻します。コンパイラにとってもう1つの固定次元となります。 |
| ブロックごとの `torch.compile` | 各 DiT ブロックを Inductor で個別にコンパイル（`compile_blocks()`）。ネイティブトークンバケットとの組み合わせにより、トレースは2つのブロックグラフに固定され、ガードの再コンパイルを排除します。 |
| コンパイルフレンドリーなホットパス | Dynamo がクリーンにトレースできないパターンをすべて監査 — `einops.rearrange` を明示的な `.unflatten()/.permute()` チェーンに置換、`torch.autocast` コンテキストマネージャを直接の `.to(dtype)` キャストに置換、辞書の `.items()` ループをコンパイル対象領域外に巻き上げ、FA4 を `@torch.compiler.disable` でラップしてクリーンなグラフブレイクを実現。 |
| Flash Attention 2 | `flash_attn` 2.x と SDPA フォールバック。FA4 は評価済みで削除 — [fa4.md](docs/optimizations/fa4.md) を参照してください。 |

コンパイルパイプラインの詳細は [docs/optimizations/for_compile.md](docs/optimizations/for_compile.md) にあります。

---

## 2. 堅牢な従来手法の実装

デフォルトの学習設定では **LoRA + OrthoLoRA + T-LoRA** をスタックします。3つすべてが保存時の thin-SVD エクスポートによりロスレスに単体 DiT チェックポイントに畳み込まれるため、アダプタローダーの依存なしで ComfyUI 互換の `*_merged.safetensors` を配布できます。

| バリアント | 特徴 | 詳細 |
|---|---|---|
| **LoRA** | クラシックな低ランク、ランク 16–32。 | — |
| **OrthoLoRA** | 直交性正則化付きの SVD パラメータ化。プレーンな LoRA としてエクスポート可能。 | [psoft-integrated-ortholora.md](docs/methods/psoft-integrated-ortholora.md) |
| **T-LoRA** | タイムステップ依存のランクマスク — 高ノイズでは低ランク、低ノイズではフルランク。学習時のみのマスクのため、マージはビット等価。 | [timestep_mask.md](docs/methods/timestep_mask.md) |

**並べ替え比較** — 同一プロンプト、`er_sde` 30ステップ、`cfg=4.0`、1024²。各 LoRA はランク 16 で2エポック、20%サブセット、学習シード 42 で学習。推論シード `{41, 42, 43}`。`python _archive/bench_methods.py` で再現可能。

|  | **LoRA** | **OrthoLoRA + T-LoRA** |
|:---:|:---:|:---:|
| seed 41 | <img src="docs/side_by_side/lora/20260423-154854-014_41_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155545-258_41_.png" width="320"> |
| seed 42 | <img src="docs/side_by_side/lora/20260423-154938-584_42_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155631-762_42_.png" width="320"> |
| seed 43 | <img src="docs/side_by_side/lora/20260423-155024-080_43_.png" width="320"> | <img src="docs/side_by_side/ortho_tlora/20260423-155718-280_43_.png" width="320"> |

<details>
<summary>ベースモデルと各バリアント（プレーン、OrthoLoRA、T-LoRA）</summary>

|  | **プレーン（ベース）** | **OrthoLoRA** | **T-LoRA** |
|:---:|:---:|:---:|:---:|
| seed 41 | <img src="docs/side_by_side/plain/20260423-160513-382_41_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155109-338_41_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155327-834_41_.png" width="240"> |
| seed 42 | <img src="docs/side_by_side/plain/20260423-160556-697_42_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155155-526_42_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155413-304_42_.png" width="240"> |
| seed 43 | <img src="docs/side_by_side/plain/20260423-160640-759_43_.png" width="240"> | <img src="docs/side_by_side/ortholora/20260423-155241-905_43_.png" width="240"> | <img src="docs/side_by_side/tlora/20260423-155458-996_43_.png" width="240"> |

</details>

**マージ**：

```bash
make merge                                  # 最新 LoRA をマージ（multiplier 1.0）
make merge ADAPTER_DIR=output/ckpt MULTIPLIER=0.8
```

非線形デルタのバリアント（ReFT / HydraLoRA `_moe`）はデフォルトで拒否されます。`--allow-partial` を指定するとこれらを除外し、LoRA 部分のみをマージします。

---

## 3. Anima 向けに最適化された最新手法

5つの最新論文を取り上げ、Anima にエンドツーエンドで実装し、実際に使用可能なエンジニアリングを施して提供しています — 単なるトイ再実装ではありません。

| 手法 | 概要 | エンジニアリングノート | ドキュメント |
|---|---|---|---|
| **Spectrum 推論** | Chebyshev 多項式による特徴予測を用いた学習不要な高速化（Han et al., CVPR 2026） — デフォルト設定で約1.75倍、より積極的なスケジュールでは最大約5倍（品質とのトレードオフ）。キャッシュされたステップでは全トランスフォーマーブロックがスキップされ — `t_embedder` + `final_layer` + `unpatchify` のみが実行されます。 | `final_layer` の `register_forward_pre_hook` により、モデルにモンキーパッチを当てることなくブロック出力をキャプチャ。適応ウィンドウスケジュールにより、実際のフォワードを初期の高ノイズステップに集中させます。安定版 ComfyUI ノードは別リポジトリで提供：[ComfyUI-Spectrum-KSampler](https://github.com/sorryhyun/ComfyUI-Spectrum-KSampler)。 | [spectrum.md](docs/methods/spectrum.md) |
| **DCW キャリブレータ** | サンプラーレベルの SNR-t バイアス補正（Yu et al., CVPR 2026） — 各 Euler ステップの `prev_sample` をモデルの `x0_pred` に向けて LL Haar バンドに沿ってミックスします。スカラー `λ`（オフライン調整）と **v4 学習型** プロンプトごとのキャリブレータの2つのモードを搭載。 | v4 ヘッドは `(アスペクト比, プロンプト, 観測プレフィックスギャップ)` で条件付けされ、`k=7` ウォームアップステップ後に発動します。バイアス方向は Anima で **(CFG × アスペクト比) に依存** することが判明 — CFG=4 非正方形では論文方向、CFG=1 / 1024² では論文と逆方向。`make dcw` でチェックポイントごとに学習。 | [dcw.md](docs/methods/dcw.md) |
| **SMC-CFG** | 速度空間での学習不要なスライディングモード CFG 補正（Wang et al., CFG-Ctrl） — cond/uncond の結合を残差 `e = v_cond − v_uncond` に適用される制御問題として扱います。追加の DiT フォワードは不要。 | **α 適応バリアント** を搭載：論文の固定ゲイン `k`（Anima の CFG=4 で約14倍ズレ、視覚的にチャタリングが見られる）をステップごとの `k_t = α·mean(\|e_t\|)` に置換。`make test-smc-cfg`（λ=5, α=0.2）。Spectrum およびモデュレーションガイダンスと組み合わせ可能。 | [smc_cfg.md](docs/methods/smc_cfg.md) |
| **OrthoHydraLoRA** | 直交化されたエキスパートとレイヤローカルルーティングによる MoE スタイルのマルチヘッド LoRA — 共有 `lora_down`、エキスパートごとの `lora_up_i`、学習されたサンプルごとのルーター。単一の低ランク部分空間で生じるクロススタイルの干渉なしに、マルチスタイル学習を実現します。元論文：[arXiv:2605.03252](https://arxiv.org/abs/2605.03252)。 | 2つのファイルを並べて保存：`anima_hydra.safetensors`（マージ済み LoRA、ComfyUI でそのまま使用可能）と `anima_hydra_moe.safetensors`（フルマルチヘッド）。バンドルされた **Anima Adapter Loader** ノード（`custom_nodes/comfyui-hydralora/`）経由で ComfyUI でライブルーティングが可能で、`HydraLoRAModule.forward` を再現する Linear ごとのフォワードフックをインストールします。 | [hydra-lora.md](docs/methods/hydra-lora.md) |
| **モデュレーションガイダンス** | `pooled_text_proj` MLP を蒸留し、AdaLN モデュレーション係数を品質向上方向に誘導（Starodubcev et al., ICLR 2026）。教師は実際のクロスアテンションを、生徒はゼロ化されたクロスアテンションを見ますが、プール済みテキストをモデュレーション経由で受け取ります。 | 凍結した DiT に対して `make distill-mod` で学習。推論時は AdaLN のタイミングで射影を適用するため、任意の LoRA バリアントと組み合わせ可能。`make test MOD=1` で有効化してサンプル生成（`SPECTRUM=1` との組み合わせも可能）。 | [mod-guidance.md](docs/methods/mod-guidance.md) |

---

## 4. 実験的機能

各機能にはドキュメントが用意されています — 使用方法、フラグ、注意事項はリンク先を参照してください。

| 機能 | 概要 | ドキュメント |
|---|---|---|
| **SPD** | Spectral Progressive Diffusion（Xiao et al., 2026） — 学習不要なマルチ解像度推論（`--spd`）。ノイズ支配の初期ステップを低解像度で実行し、スペクトルノイズ展開で高周波ディテールを注入します。オプションの軌道アダプタのファインチューンも可能（`make exp-spd`）。 | [spd.md](docs/experimental/spd.md) |
| **ChimeraHydra** | デュアルプール加法 MoE：コンテンツプール（レイヤローカルルーター）+ 周波数プール（FEI + σ 特徴上のネットワークルーター）。それぞれが互いに素な SVD 部分空間上の非対称 HydraLoRA。HydraLoRA + TimeStep Master + FeRA を統合。`make exp-chimera`。 | [chimera-hydra.md](docs/experimental/chimera-hydra.md) |
| **Soft Tokens** | SoftREPA（Lee et al., NeurIPS 2025） — レイヤごと × t ごとの学習可能テキストトークン（約1M パラメータ）を `crossattn_emb` にスプライス。DiT は凍結。`make exp-soft-tokens`。 | [soft_tokens.md](docs/experimental/soft_tokens.md) |
| **Turbo** | 28ステップの教師からの Decoupled DMD 蒸留（Liu et al., 2025）。4〜8ステップのジェネレータを生成。出力は通常の LoRA — `--infer_steps 4 --cfg 1.0` で推論。`make exp-turbo`。 | [turbo_anima_dmd_lora.md](docs/proposal/turbo_anima_dmd_lora.md) |
| **DirectEdit** | フロー反転による画像編集（Yang & Ye, 2026） — ノイズまで反転し、編集条件付けをスワップし、V-injection で再ノイズ除去。ソースキャプションは **Anima Tagger**（画像 → Anima 形式のタグ）から取得。`make exp-test-directedit`。 | [directedit_editing_v3.md](docs/experimental/directedit_editing_v3.md) |
| **ReFT** | ブロックレベルの残差ストリーム介入（LoReFT, NeurIPS 2024）。任意の LoRA バリアントと組み合わせ可能。 | [reft.md](docs/methods/reft.md) |
| **IP-Adapter** | 分離型画像クロスアテンション（Ye et al. 2023）。DiT は凍結。Perceiver リサンプラー + ブロックごとの `to_k_ip`/`to_v_ip` を学習。 | [ip-adapter.md](docs/experimental/ip-adapter.md) |
| **EasyControl** | 拡張セルフアテンション画像条件付け。DiT は凍結。セルフアテンション + FFN 上のブロックごとの cond LoRA とスカラー `b_cond` ゲートを学習。 | [easycontrol.md](docs/experimental/easycontrol.md) |
| **埋め込み反転** | 凍結した DiT を通してターゲット画像に一致するようテキスト埋め込みを最適化。 | [invert.md](docs/methods/invert.md) |

> **コントリビュートに興味がありますか？** 外部の協力が特に大きなインパクトを持つ2つの領域：**IP-Adapter のプロダクション化**（テスト、公開参照チェックポイント、より軽量なビジョンエンコーダ）と **EasyControl アダプタ**（canny / depth / pose / … — 各コントロールタイプが1つの自己完結した PR になります）。[CONTRIBUTING.md → Priority areas](CONTRIBUTING.md#priority-areas) を参照してください。

---

## セットアップ

> クイックインストールは上記の [始め方](#how-to-start) を参照してください。以下は手動クローンの手順です。

### 手動（クローンから）

```bash
uv sync                   # Python 3.13 とビルド済み Flash Attention 2
hf auth login
make download-models      # DiT + Qwen3 TE + QwenImage VAE（+ マスク・画像条件付け用の SAM3 / MIT / PE）を models/ にダウンロード
# 学習画像を image_dataset/ に配置（.txt キャプションサイドカー付き）
make gui                  # 推奨 — 設定エディタ + データセットブラウザ + 学習モニタ
```

`uv sync` は **torch 2.12 + CUDA 13.2** に解決されます。

> **Anima は uv でロックされたアプリケーション環境として提供されており、一般的な pip パッケージではありません。** `pyproject.toml` は `python ==3.13.*`、特定の torch / flash-attn ホイール URL、および `index-strategy = "unsafe-best-match"` を固定しています — これらはメンテナーが選択した動作確認済みビルドです。コミットされた `uv.lock` に対して `uv sync` でインストールしてください。`pyproject.toml` から `pip install` しないでください（pip は uv のインデックス戦略やビルド済み flash-attn ホイールを尊重しません）。

CLI の場合：

```bash
make preprocess           # VAE 互換のリサイズとバリデーション
make lora                 # または: PRESET=fast_16gb make lora / PRESET=low_vram make lora / make exp-chimera
make test                 # 最新の学習済み LoRA でサンプル生成
```

設定チェーン：`configs/base.toml → configs/presets.toml[<preset>] → configs/methods/<method>.toml → CLI args`。`PRESET=low_vram make lora` や `--network_dim 32 --max_train_epochs 64` で上書き可能。全フラグのリファレンスは [docs/guidelines/training.md](docs/guidelines/training.md) および [docs/guidelines/inference.md](docs/guidelines/inference.md) にあります。

---

## ドキュメント

| ドキュメント | 内容 |
|-----|----------|
| [guidelines/training.md](docs/guidelines/training.md) | 学習フラグ、LoRA バリアント、キャプションシャッフル、マスク付きロス、データセット設定 |
| [guidelines/inference.md](docs/guidelines/inference.md) | 推論フラグ、P-GRAFT、プロンプトファイル、LoRA 形式変換 |
| [optimizations/](docs/optimizations/) | コンパイルパイプライン、FA4 事後分析、CUDA 13.2 |
| [methods/](docs/methods/) | 手法ごとのドキュメント — HydraLoRA、ReFT、Spectrum、反転、モデュレーションガイダンス、T-LoRA、OrthoLoRA |

---

## ライセンス

ツールキットコード：[MIT](LICENSE)。

Anima / CircleStone の **ベースモデルウェイト** は **CircleStone Labs Non-Commercial License v1.0** の下で提供されており、このリポジトリによって再ライセンスされるものではありません。これらのウェイトから学習された LoRA、ファインチューン、またはマージ済みチェックポイントはすべて派生物であり、非商用条件を継承します。[NOTICE](NOTICE) を参照してください。
