# qe — Qwen3.8-27B を Tenstorrent P100A 1 枚で動かす B=1 推論エンジン

27B の dense モデル (Gated DeltaNet 48 層 + full attention 16 層) を、Blackhole **P100A 1 枚 (28GB、448GB/s)** で
バッチ 1 のまま速く動かすための、tt-metal / ttnn 上の小さな推論エンジンです。
重みは全部 bfp4、モデル付属の MTP 層で投機的デコードを行い、OpenAI 互換サーバーとして使えます。

| | 速度 |
|---|---|
| デコード (コード生成、貪欲法) | **47 tok/s** |
| デコード (日本語の説明文 / 英語の物語) | 32 / 28 tok/s |
| prefill (短〜中程度のプロンプト) | 約 300 tok/s |
| 最大文脈 | **163,840 トークン** (KV 用に DRAM 残り 10GB 強) |
| 12.4 万トークンのプロンプト | prefill 466 秒 (265 tok/s)、デコード 34 tok/s、needle テスト正解 |

- 投機的デコードの出力は、通常の貪欲法 (1 トークンずつ) の出力と一致します (コードのプロンプトで確認)。
- 短いプロンプトでは最初のトークンまで約 0.6 秒です。

## 仕組み

1 トークンあたりの時間の大半は重みの読み出しなので、(1) 読む量を減らし、(2) 1 回の読み出しで複数トークンを処理し、
(3) 行列積以外の小さな op の固定費を削る、の 3 つで速くしています。

- **bfp4 の DRAM シャーディング行列積** (`qe/ops.py`): 重みを 7 バンクに分けて流し読み。
  ワーカー数 3・in0_block_w=2 で公称帯域の 93% (約 415GB/s)。全部 bfp4 で行列積だけ約 38ms/ステップ。
- **GDN 融合カーネル** (`kernels/gdn_fused_*.cpp`、`qe/gdn_kernel.py`): conv 以降の
  L2 正規化 → T トークン分の DeltaNet 再帰 → gated RMSNorm を 1 本のカーネルで行う (値ヘッド 1 つに 1 コア)。
  各トークン後の状態をスロットに残すので、投機的デコードで「確定した位置」から次を再開できる。
- **T 行モード** (`forward_spec`): T 行のトークンをまとめて流す。検証は T=4 (下書き 3 個)、prefill は T=32。
  conv の履歴はホストが書く 0/1 の選択行列との行列積でずらすので、トレースを固定したまま採択数に追従できる。
- **MTP 投機的デコード** (`MTPHead`、`qe/generator.py`): 付属の MTP 層で 3 トークン先まで下書きし、本体で一括検証。
  下書き用には lm_head の bfp4 の複製を使う。
- **GQA split-K アテンション** (`kernels/gqa_attn_*.cpp`、`gqa_comb_*.cpp`、`qe/gqa_attn.py`):
  KV ヘッド 4 × 区間 27 = 108 コアが、担当区間の KV を 1 回だけ読み、同じ KV ヘッドを使う 6 q ヘッド × T 行を
  まとめて計算 (online softmax)。部分結果を別の op で合成する。区間とチャンク数は実行時に位置から決まるのでトレース可能。
  12.4 万トークンの文脈で 4 行の検証が 1 層 1.7ms (DRAM 帯域の下限 1.27ms 付近)。
  ttnn の decode SDPA に T 行を「T ユーザー」として渡す方式 (5.6ms) や chunked SDPA (32 行で 15ms) の置き換え。
- **KV 書き込みカーネル** (`kernels/kv_write.cpp`): T 行の K/V をページ KV に競合なく書く。
- 残差ストリームは 40 コアの幅方向シャードで持ち、RMSNorm もシャード版 (89µs → 6.5µs)。argmax はデバイス上で行う。

## 必要なもの

- Blackhole P100A (P150 でも動くはず。未確認)
- tt-metal (開発時は `v0.79.0-dev20260916-3-g5f02a6718f4`) と同じ venv の ttnn、`transformers` (5.12 で確認)、
  サーバーには `fastapi` と `uvicorn`
- モデル: Qwen3.8-27B の HF 形式 (bf16、約 56GB)
- 変換済み重みのキャッシュ用に約 16GB (bfp4)

## 使い方

```bash
source /path/to/tt-metal/python_env/bin/activate      # ttnn が入っている環境
export TT_METAL_HOME=/path/to/tt-metal PYTHONPATH=/path/to/tt-metal
export QE_MODEL_DIR=/path/to/Qwen3.8-27B             # HF 形式のモデル
export QE_CACHE_DIR=/path/to/qe_cache/Qwen3.8-27B    # 変換済み重みのキャッシュ (初回に作られる、約 8 分)

# OpenAI 互換サーバー (127.0.0.1:8010)
python server.py --port 8010 [--max-seq 163840]

# 1 プロンプトで速度と位置ごとの採択率を見る
python run_mtp.py --prompt "Pythonでフィボナッチ数列を返す関数を書いてください。"
```

サーバー:

- `POST /v1/chat/completions` (stream 対応)、`GET /v1/models`、`GET /health`。1 リクエストずつ処理。
- 思考は `chat_template_kwargs.enable_thinking` (既定 false)。`<think>` の中身は `reasoning_content` に分ける。
- 起動は約 20 秒 (キャッシュがページキャッシュにあるとき)。

環境変数:

| 変数 | 意味 |
|---|---|
| `QE_MODEL_DIR` / `QE_CACHE_DIR` | モデルと変換済み重みの場所 (`qe/paths.py`) |
| `QE_GQA_ATTN=0` | 自作の GQA アテンションを使わず、ttnn の SDPA に戻す |
| `QE_CHUNKED_PREFILL=0` | (GQA を切ったとき) prefill で chunked SDPA を使わない |
| `QE_FUSED_GDN=0` | GDN 融合カーネルを使わず、元の op の並びで動かす |

## 制限

- **貪欲法のみ**。temperature などは受け取りますが無視します (投機的サンプリングは未実装)。
- **品質は未評価**です。重みを全部 bfp4 にしたときの劣化 (perplexity) をまだ測っていません。
  bfp4 の量子化誤差は llama.cpp の Q4_0 の約 1.4 倍でした (`quantstudy.py`)。
- 会話の続きの再利用 (前回の KV/状態を使い回す仕組み) はあるものの、チャットテンプレートの差で 2 ターン目に一致せず、
  実際には毎回全体を prefill しています。
- 短い文脈では、GQA カーネルの固定費 (約 0.1ms/層) のぶん、ttnn の SDPA より 1 ステップ 1ms ほど遅くなります。
- デバイスが固まった場合は `tt-smi -r 0` を 2 回、15 秒ほど待つと戻ることが多いです。

## ファイル

| ファイル | 内容 |
|---|---|
| `qe/model.py` | エンジン本体 (`Qwen35Engine`、`MTPHead`、T 行モード、トレース、prefill) |
| `qe/generator.py` | MTP 投機的デコードの生成ループ |
| `qe/ops.py` | DRAM シャーディングの bfp4/bfp8 行列積、重みの変換とキャッシュ |
| `qe/gdn.py`、`qe/gdn_kernel.py` | Gated DeltaNet 層と融合カーネルの起動 |
| `qe/attn.py`、`qe/gqa_attn.py`、`qe/kv_kernel.py` | attention 層、GQA split-K アテンション、KV 書き込み |
| `kernels/*.cpp` | 自作カーネル (generic_op で起動) |
| `server.py` | OpenAI 互換サーバー |
| `run_mtp.py`、`run_spec_test.py`、`run_gen.py` | 生成と速度の確認 (MTP / T 行 / T=1) |
| `cmp_hf.py` | 先頭 N 層を HF (CPU fp32) と比較 |
| `test_*.py` | カーネルと層の単体テスト |
| `bench_*.py`、`mmbench.py`、`sdpabench.py`、`memprobe.py`、`longctx.py`、`prefill_agree.py`、`quantstudy.py` | 計測 |

## ライセンス

Apache-2.0
