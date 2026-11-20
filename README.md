# tsumi — Qwen3.8-27B を Tenstorrent P100A 1 枚で動かす B=1 推論エンジン

名前はツミ (*Tachyspiza gularis*)、日本で一番小さいタカから。小さな構成で大きなモデルを速く動かす、という意味を込めています。

27B の dense モデル (Gated DeltaNet 48 層 + full attention 16 層) を、Blackhole **P100A 1 枚 (28GB、448GB/s)** で
バッチ 1 のまま速く動かすための、tt-metal / ttnn 上の小さな推論エンジンです。
重みは全部 bfp4、モデル付属の MTP 層で投機的デコードを行い、OpenAI 互換サーバーとして使えます。

| | 速度 |
|---|---|
| デコード (コード生成、貪欲法) | **47 tok/s** |
| デコード (日本語の説明文 / 英語の物語) | 32 / 28 tok/s |
| prefill (数千トークンのプロンプト) | **約 1,500〜2,000 tok/s** (4,000 トークンで応答開始まで約 2 秒) |
| 最大文脈 | **155,648 トークン** (KV 用に DRAM 残り約 10GB) |
| 12.4 万トークンのプロンプト | prefill 130 秒 (957 tok/s)、デコード 34 tok/s、needle テスト正解 |

- 投機的デコードの出力は、通常の貪欲法 (1 トークンずつ) の出力と一致します (コードのプロンプトで確認)。
- 短いプロンプトでは最初のトークンまで約 0.6 秒です。

## 仕組み

1 トークンあたりの時間の大半は重みの読み出しなので、(1) 読む量を減らし、(2) 1 回の読み出しで複数トークンを処理し、
(3) 行列積以外の小さな op の固定費を削る、の 3 つで速くしています。

- **bfp4 の DRAM シャーディング行列積** (`tsumi/ops.py`): 重みを 7 バンクに分けて流し読み。
  ワーカー数 3・in0_block_w=2 で公称帯域の 93% (約 415GB/s)。全部 bfp4 で行列積だけ約 38ms/ステップ。
- **GDN 融合カーネル** (`kernels/gdn_fused_*.cpp`、`tsumi/gdn_kernel.py`): conv 以降の
  L2 正規化 → T トークン分の DeltaNet 再帰 → gated RMSNorm を 1 本のカーネルで行う (値ヘッド 1 つに 1 コア)。
  各トークン後の状態をスロットに残すので、投機的デコードで「確定した位置」から次を再開できる。
- **T 行モード** (`forward_spec`): T 行のトークンをまとめて流す。検証は T=4 (下書き 3 個)、prefill は T=32。
  conv の履歴はホストが書く 0/1 の選択行列との行列積でずらすので、トレースを固定したまま採択数に追従できる。
- **MTP 投機的デコード** (`MTPHead`、`tsumi/generator.py`): 付属の MTP 層で 3 トークン先まで下書きし、本体で一括検証。
  下書き用には lm_head の bfp4 の複製を使う。
- **GQA split-K アテンション** (`kernels/gqa_attn_*.cpp`、`gqa_comb_*.cpp`、`tsumi/gqa_attn.py`):
  KV ヘッド 4 × 区間 27 = 108 コアが、担当区間の KV を 1 回だけ読み、同じ KV ヘッドを使う 6 q ヘッド × T 行を
  まとめて計算 (online softmax)。部分結果を別の op で合成する。区間とチャンク数は実行時に位置から決まるのでトレース可能。
  12.4 万トークンの文脈で 4 行の検証が 1 層 1.7ms (DRAM 帯域の下限 1.27ms 付近)。
  ttnn の decode SDPA に T 行を「T ユーザー」として渡す方式 (5.6ms) や chunked SDPA (32 行で 15ms) の置き換え。
- **大きい T の prefill** (`tsumi/prefill.py`): プロンプトは最大 1024 行ずつ、`minimal_matmul` (同じ bfp4 重み)、
  チャンク並列の GDN (`ttnn.transformer.chunk_gated_delta_rule`)、自作の因果 conv カーネル (`kernels/conv_*.cpp`)、
  chunked SDPA で流す。終わったら GDN の状態と conv 履歴を T 行モードの形式に戻して、そのままデコードに入る。
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
export TSUMI_MODEL_DIR=/path/to/Qwen3.8-27B             # HF 形式のモデル
export TSUMI_CACHE_DIR=/path/to/qe_cache/Qwen3.8-27B    # 変換済み重みのキャッシュ (初回に作られる、約 8 分)

# OpenAI 互換サーバー (127.0.0.1:8010)。run_server.sh はハング時にデバイスをリセットして再起動する
./run_server.sh --port 8010 [--max-seq 155648]     # または python server.py ...
```

サーバー:

- `POST /v1/chat/completions` (stream 対応)、`GET /v1/models`、`GET /health`。1 リクエストずつ処理。
- 思考は `chat_template_kwargs.enable_thinking` (既定 false)。`<think>` の中身は `reasoning_content` に分ける。
- ツール呼び出し: `tools` / `tool_choice` / 履歴の `tool_calls` と `role: tool` に対応 (OpenAI 形式、ストリーミングも)。
  モデルが出す `<tool_call>` の XML を `tool_calls` に変換する (`tsumi/toolcall.py`)。
  `tool_choice` は auto / none。required と関数名の指定は thinking オフのときだけ効く (応答の先頭を `<tool_call>` にする)。
- 起動は約 20 秒 (キャッシュがページキャッシュにあるとき)。
- ハング検知: デバイスの処理が 30 秒進まないと (`TT_METAL_OPERATION_TIMEOUT_SECONDS`)、そのリクエストに 503 を返し、
  `/health` も 503 にして終了コード 75 で終わる。トークンが 300 秒出ないとき (`TSUMI_HANG_BACKSTOP`) も強制終了する。
  `run_server.sh` はそれを見て `tt-smi -r 0` を 2 回 → 再起動する (約 1.5 分で復帰。10 分に 3 回を超えたら諦める)。

環境変数 (旧名の `QE_...` も引き続き読めます):

| 変数 | 意味 |
|---|---|
| `TSUMI_MODEL_DIR` / `TSUMI_CACHE_DIR` | モデルと変換済み重みの場所 (`tsumi/paths.py`) |
| `TSUMI_LARGE_PREFILL=0` / `TSUMI_LARGE_T` | 大きい T の prefill を使わない (T=32 の旧経路) / 1 チャンクの最大行数 (既定 1024) |
| `TSUMI_MULTI_CONV=0` / `TSUMI_MAX_CONVS` | 複数の会話を保持しない / 保持する会話数の上限 (既定 8、ホストに 1 会話約 180MB) |
| `TSUMI_SNAPSHOT=0` | 会話の再利用用の状態スナップショットを取らない |
| `TSUMI_GQA_ATTN=0` | 自作の GQA アテンションを使わず、ttnn の SDPA に戻す |
| `TSUMI_CHUNKED_PREFILL=0` | (GQA を切ったとき) prefill で chunked SDPA を使わない |
| `TSUMI_FUSED_GDN=0` | GDN 融合カーネルを使わず、元の op の並びで動かす |

## 制限

- **貪欲法のみ**。temperature などは受け取りますが無視します (投機的サンプリングは未実装)。
- 全部 bfp4 の perplexity は bf16 比で日本語 +5%、英語 +7% (llama.cpp の IQ3_XXS 並み、Q4_0 は +2〜3%)。
  実務的なタスクでの品質は未評価。
- 会話の続きは再利用します (前回のプロンプト + 返答、またはアシスタントのターン開始位置の状態スナップショットから差分だけ prefill)。
  複数の会話 (本体とサブエージェントなど) は KV のブロックを分けて保持し、リクエストごとに切り替えます (約 0.1 秒)。
  KV の合計は最大文脈ぶん (155,648 トークン) で、足りなくなると最後に使ったのが最も古い会話から捨てます。
- 短い文脈では、GQA カーネルの固定費 (約 0.1ms/層) のぶん、ttnn の SDPA より 1 ステップ 1ms ほど遅くなります。
- デバイスが固まった場合は `tt-smi -r 0` を 2 回、15 秒ほど待つと戻ることが多いです。

## ファイル

| ファイル | 内容 |
|---|---|
| `tsumi/model.py` | エンジン本体 (`Qwen35Engine`、`MTPHead`、T 行モード、トレース、prefill) |
| `tsumi/generator.py` | MTP 投機的デコードの生成ループ |
| `tsumi/ops.py` | DRAM シャーディングの bfp4/bfp8 行列積、重みの変換とキャッシュ |
| `tsumi/gdn.py`、`tsumi/gdn_kernel.py` | Gated DeltaNet 層と融合カーネルの起動 |
| `tsumi/attn.py`、`tsumi/gqa_attn.py`、`tsumi/kv_kernel.py` | attention 層、GQA split-K アテンション、KV 書き込み |
| `tsumi/kvpager.py` | KV ブロックのプールとページ表 (複数の会話) |
| `tsumi/prefill.py`、`tsumi/conv_kernel.py`、`tsumi/page_copy.py` | 大きい T の prefill、因果 conv カーネル、ページコピー |
| `kernels/*.cpp` | 自作カーネル (generic_op で起動) |
| `server.py` | OpenAI 互換サーバー |
| `scripts/` | 開発用のテスト・計測・perplexity 評価 (使い方は [`scripts/README.md`](scripts/README.md)) |

## ライセンス

Apache-2.0
