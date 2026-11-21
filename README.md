# tsugumi — Qwen3.8-27B を Tenstorrent P100A 1 枚で動かす B=1 推論エンジン

名前はツグミ (*Turdus eunomus*) から。冬になるとシベリアから海を越えて日本に渡ってくる、身近な小さな鳥です。[[AUTHOR: 名前の由来の一文を確認]]

27B の dense モデル (Gated DeltaNet 48 層 + full attention 16 層) を、Blackhole **P100A 1 枚 (28GB、448GB/s)** で
バッチ 1 のまま速く動かすための、tt-metal / ttnn 上の小さな推論エンジンです。
重みは全部 bfp4、モデル付属の MTP 層で投機的デコードを行い、OpenAI 互換サーバーとして使えます。

| | 速度 |
|---|---|
| デコード (コード生成、貪欲法) | **47〜50 tok/s** |
| デコード (コードの修正・文章の校正など、プロンプトを写す応答) | **75〜115 tok/s** (n-gram 下書き) |
| デコード (日本語の説明文 / 英語の物語) | 32 / 28 tok/s |
| prefill (数千トークンのプロンプト) | **約 1,500〜2,000 tok/s** (4,000 トークンで応答開始まで約 2 秒) |
| 最大文脈 | **262,144 トークン** (モデルの上限。KV は bfp8 で約 9GiB) |
| 12.7 万トークンのプロンプト | prefill 110 秒 (1,150 tok/s)、デコード 36 tok/s、干し草山テスト正解 |
| 24.5 万トークンのプロンプト | prefill 307 秒 (800 tok/s)、デコード 30 tok/s、干し草山テスト正解 |

- 投機的デコードの出力は、通常の貪欲法 (1 トークンずつ) の出力と一致します (コードのプロンプトで確認)。サンプリングも同じ分布です。
- 短いプロンプトでは最初のトークンまで約 0.6 秒です。

## 仕組み

1 トークンあたりの時間の大半は重みの読み出しなので、(1) 読む量を減らし、(2) 1 回の読み出しで複数トークンを処理し、
(3) 行列積以外の小さな op の固定費を削る、の 3 つで速くしています。

- **bfp4 の DRAM シャーディング行列積** (`tsugumi/ops.py`): 重みを 7 バンクに分けて流し読み。
  ワーカー数 3・in0_block_w=2 で公称帯域の 93% (約 415GB/s)。全部 bfp4 で行列積だけ約 38ms/ステップ。
- **GDN 融合カーネル** (`kernels/gdn_fused_*.cpp`、`tsugumi/gdn_kernel.py`): conv 以降の
  L2 正規化 → T トークン分の DeltaNet 再帰 → gated RMSNorm を 1 本のカーネルで行う (値ヘッド 1 つに 1 コア)。
  各トークン後の状態をスロットに残すので、投機的デコードで「確定した位置」から次を再開できる。
- **T 行モード** (`forward_spec`): T 行のトークンをまとめて流す。検証は T=4 (下書き 3 個)、prefill は T=32。
  conv の履歴はホストが書く 0/1 の選択行列との行列積でずらすので、トレースを固定したまま採択数に追従できる。
- **MTP 投機的デコード** (`MTPHead`、`tsugumi/generator.py`): 付属の MTP 層で 3 トークン先まで下書きし、本体で一括検証。
  下書き用には lm_head の bfp4 の複製を使う。
- **GQA split-K アテンション** (`kernels/gqa_attn_*.cpp`、`gqa_comb_*.cpp`、`tsugumi/gqa_attn.py`):
  KV ヘッド 4 × 区間 27 = 108 コアが、担当区間の KV を 1 回だけ読み、同じ KV ヘッドを使う 6 q ヘッド × T 行を
  まとめて計算 (online softmax)。部分結果を別の op で合成する。区間とチャンク数は実行時に位置から決まるのでトレース可能。
  12.4 万トークンの文脈で 4 行の検証が 1 層 1.7ms (DRAM 帯域の下限 1.27ms 付近)。
  ttnn の decode SDPA に T 行を「T ユーザー」として渡す方式 (5.6ms) や chunked SDPA (32 行で 15ms) の置き換え。
- **大きい T の prefill** (`tsugumi/prefill.py`): プロンプトは最大 1024 行ずつ、`minimal_matmul` (同じ bfp4 重み)、
  チャンク並列の GDN (`ttnn.transformer.chunk_gated_delta_rule`)、自作の因果 conv カーネル (`kernels/conv_*.cpp`)、
  chunked SDPA で流す。終わったら GDN の状態と conv 履歴を T 行モードの形式に戻して、そのままデコードに入る。
- **KV 書き込みカーネル** (`kernels/kv_write.cpp`): T 行の K/V をページ KV に競合なく書く。
- 残差ストリームは 40 コアの幅方向シャードで持ち、RMSNorm もシャード版 (89µs → 6.5µs)。argmax はデバイス上で行う。

## 必要なもの

- Blackhole P100A (P150 でも動くはず。未確認)
- ホストの RAM: 32GB で動作確認
- tt-metal をソースからビルドしたもの (開発時は `v0.79.0-dev20260916-3-g5f02a6718f4`)。tt-metal の
  `models/` を import するので、`PYTHONPATH` に tt-metal のディレクトリが要る
- 同じ venv に ttnn、`torch`、`transformers` (5.12 で確認)、`safetensors`、`Pillow` (画像入力)、
  サーバー用の `fastapi` と `uvicorn`
- `tt-smi` (`run_server.sh` がハング後のリセットに使う。PATH になければ `TT_SMI=/path/to/tt-smi`)
- モデル: Qwen3.8-27B の HF 形式 (bf16、約 56GB)
- 変換済み重みのキャッシュ用に約 16GB (bfp4)

## 使い方

```bash
source /path/to/tt-metal/python_env/bin/activate      # ttnn が入っている環境
export TT_METAL_HOME=/path/to/tt-metal PYTHONPATH=/path/to/tt-metal
export TSUGUMI_MODEL_DIR=/path/to/Qwen3.8-27B             # HF 形式のモデル
export TSUGUMI_CACHE_DIR=/path/to/qe_cache/Qwen3.8-27B    # 変換済み重みのキャッシュ (初回に作られる、約 8 分)

# OpenAI 互換サーバー (127.0.0.1:8010)。run_server.sh はハング時にデバイスをリセットして再起動する
./run_server.sh --port 8010 [--max-seq 262144]     # または python server.py ...
```

試す:

```bash
curl -s http://127.0.0.1:8010/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": "こんにちは"}]}'
```

OpenAI の SDK や、OpenAI 互換の API を使えるクライアント (コーディングエージェントなど) からは、
base URL を `http://127.0.0.1:8010/v1` にして使えます。

### 認証 (別のマシンから使うとき)

既定では認証はなく、127.0.0.1 だけで待ち受けます。`--host 0.0.0.0` などでネットワークに出すときは、API キーを設定してください。

```bash
export TSUGUMI_API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')   # または --api-key KEY
./run_server.sh --host 0.0.0.0 --port 8010
# 複数のキー: TSUGUMI_API_KEY=key1,key2、またはファイルに 1 行 1 キー: --api-key-file keys.txt (TSUGUMI_API_KEY_FILE)
```

- キーを設定すると、`GET /` (Web UI のページ) と `GET /health` 以外は `Authorization: Bearer <キー>` が必要になり、
  ない・違うときは 401 を返します。OpenAI の SDK やクライアントは API キーの設定をこのヘッダで送ります。
- `/health` はキーなしでも答えますが、処理中のリクエストの詳細は出しません。
- Web UI は最初の送信で 401 になるとキーを聞き、そのブラウザに保存します。
- 通信は暗号化しません。インターネットに出すときは、TLS を終端するリバースプロキシ (Caddy、nginx など) の後ろに置いてください。
- キーなしで 127.0.0.1 以外で待ち受けると、起動時に警告を出します。

サーバー:

- `POST /v1/chat/completions`・`POST /v1/completions` (テンプレートなしの続き書き、プロンプトは文字列かトークン ID 列。どちらも stream 対応)、
  `POST /tokenize`・`/detokenize` (vLLM 形式)、`GET /v1/models`、`GET /health`。1 リクエストずつ処理。
- **Web UI**: ブラウザで `http://127.0.0.1:8010/` を開くと、試し用の最小限のチャット画面 (`webui/index.html`)。
  thinking、温度、最大トークン数、system プロンプトを切り替えられ、応答ごとにプロンプト側 (再利用、画像の読み込み、
  prefill のトークン数・秒・tok/s、応答開始) と生成側 (トークン数、tok/s、トークン/ステップ、思考) の数値を表示する。
  別のマシンから開くときは `--host 0.0.0.0` (API キーの設定を推奨、上の「認証」)。
- **画像入力**: `messages` の content に OpenAI 形式の `{"type": "image_url", "image_url": {"url": ...}}` (data: URL の base64、
  または http(s) の URL)。ViT (27 層、約 4.6 億パラメータ) はデバイスで動かす (重み bfp8 で DRAM 0.52 GiB、1024×1024 で
  0.17 秒。`TSUGUMI_VISION_DEVICE=0` で CPU、約 14 秒)。画像は
  100 万ピクセル (`TSUGUMI_IMAGE_MAX_PIXELS`) 以下に縮めて、32×32 ピクセルごとに 1 トークン (最大 1,024 トークン)。
  同じ画像 (バイト列のハッシュ) は 2 回目から符号化しない (ホストに 65,536 トークンぶん保持) し、会話の再利用も効く。
  画像トークンにはそれぞれの画像専用の仮想 ID (248,320 以上) を振るので、トークン数が同じ別の画像を誤って再利用しない。
  Web UI は「画像」ボタン・貼り付け・ドラッグ&ドロップで添付できる。動画は未対応。
- 思考は `chat_template_kwargs.enable_thinking` (既定 false)。`<think>` の中身は `reasoning_content` に分ける。
- 思考の上限: `thinking_budget` (トークン数。`chat_template_kwargs` 内でも可、既定は `--thinking-budget` / `TSUGUMI_THINKING_BUDGET`、
  -1 で上限なし)。上限に達すると「時間がないのでここまでの考えで答える」旨の一文と `</think>` を差し込んで答えに移る
  (ステップの切れ目で判定するので最大 7 トークン超える)。上限を指定したときは `max_tokens` は答えの分だけを数える。
- `logprobs` / `top_logprobs` (20 まで、completions は `logprobs: n`): デバイスの上位 64 の logits での log-softmax なので
  少し高めに出る (64 の外は差し込んだトークンだけで -9999)。chat では本文のトークンだけ (思考は含まない)。
- サンプリング: `temperature` > 0 で `top_p`、`top_k` (64 まで)、`min_p`、`seed` が効く。`temperature` がなければ貪欲法。
  MTP の投機的デコードのまま厳密 (1 トークンずつ引いたのと同じ分布)。Qwen の推奨は thinking オフで 0.7 / 0.8 / 20、オンで 0.6 / 0.95 / 20。
- ツール呼び出し: `tools` / `tool_choice` / 履歴の `tool_calls` と `role: tool` に対応 (OpenAI 形式、ストリーミングも)。
  モデルが出す `<tool_call>` の XML を `tool_calls` に変換する (`tsugumi/toolcall.py`)。
  `tool_choice` は auto / none。required と関数名の指定は thinking オフのときだけ効く (応答の先頭を `<tool_call>` にする)。
- 起動は約 40 秒 (重みがページキャッシュにあるとき。再起動直後は 2〜3 分)。
- ログ: `時刻 レベル [tsugumi] 本文` の形式で標準出力に出す。起動時は設定の一覧と段階ごとの所要時間・DRAM の空き、
  リクエストごとに受付 (プロンプト長、パラメータ、無視したパラメータ) と完了 (再利用、応答開始、tok/s、終了理由) の 2 行。
  `TSUGUMI_LOG_LEVEL=DEBUG` で詳しく。`run_server.sh` は tt-metal / ttnn のログを warning 以上に絞る (`TT_LOGGER_LEVEL`、`LOGURU_LEVEL`)。
- `finish_reason` は `stop` / `length` (max_tokens で打ち切り) / `tool_calls`。`stop` (停止文字列、4 つまで) は応答本文に効き、思考には効かない。
- モデル名: 既定はモデルのディレクトリ名 (`Qwen3.8-27B`)。`--served-model-name tsugumi,gpt-4o` (または `TSUGUMI_SERVED_MODEL_NAME`) で上書き、
  カンマで別名。リクエストの `model` は何でも受け付ける (別名ならそれを返す)。`/v1/models` の `meta` に量子化と最大文脈。
- 終了 (SIGTERM / Ctrl-C): 受け付けを止め、生成中なら次のトークンで終えて応答を閉じ、デバイスを閉じて終わる (1 秒前後)。
- `GET /health` は処理中かどうか・待ちの数・処理中のリクエスト (生成済みトークン数、経過秒) を返す。待っている間にクライアントが
  切断したリクエストは、順番が来ても生成しない。
- ハング検知: デバイスの処理が 30 秒進まないと (`TT_METAL_OPERATION_TIMEOUT_SECONDS`)、そのリクエストに 503 を返し、
  `/health` も 503 にして終了コード 75 で終わる。トークンが 300 秒出ないとき (`TSUGUMI_HANG_BACKSTOP`) も強制終了する。
  `run_server.sh` はそれを見て `tt-smi -r 0` を 2 回 → 再起動する (約 1.5 分で復帰。10 分に 3 回を超えたら諦める)。

環境変数 (旧名の `TSUMI_...`・`QE_...` も引き続き読めます):

| 変数 | 意味 |
|---|---|
| `TSUGUMI_MODEL_DIR` / `TSUGUMI_CACHE_DIR` | モデルと変換済み重みの場所 (`tsugumi/paths.py`) |
| `TSUGUMI_NGRAM=0` / `TSUGUMI_NGRAM_MIN` | n-gram 下書き (プロンプト参照) を使わない / 使う条件の一致長 (既定 6 トークン) |
| `TSUGUMI_VERIFY_T` | n-gram 下書きを検証する長いモードの行数 (既定 8、0 で T=4 のまま) |
| `TSUGUMI_DRAFT_VOCAB=0` | MTP の下書きを全語彙で出す (既定は頻出 63,168 語、`tsugumi/draft_vocab.pt`) |
| `TSUGUMI_KV_DTYPE=bf16` | KV キャッシュを bf16 にする (既定 bfp8。bf16 では max_seq 155,648 まで) |
| `TSUGUMI_PREFILL_SDPA=hifi4` | prefill の attention を q chunk 128 + HiFi4 に戻す (既定は 256 + HiFi2) |
| `TSUGUMI_LARGE_PREFILL=0` / `TSUGUMI_LARGE_T` | 大きい T の prefill を使わない (T=32 の旧経路) / 1 チャンクの最大行数 (既定 1024) |
| `TSUGUMI_MULTI_CONV=0` / `TSUGUMI_MAX_CONVS` | 複数の会話を保持しない / 保持する会話数の上限 (既定 8、ホストに 1 会話約 180MB) |
| `TSUGUMI_SNAPSHOT=0` | 会話の再利用用の状態スナップショットを取らない |
| `TSUGUMI_GQA_ATTN=0` | 自作の GQA アテンションを使わず、ttnn の SDPA に戻す |
| `TSUGUMI_CHUNKED_PREFILL=0` | (GQA を切ったとき) prefill で chunked SDPA を使わない |
| `TSUGUMI_FUSED_GDN=0` | GDN 融合カーネルを使わず、元の op の並びで動かす |
| `TSUGUMI_IMAGES=0` | 画像入力を切る (ViT を読み込まない。DRAM 0.52 GiB の節約) |
| `TSUGUMI_CAPTURE_GUARD=1` | (デバッグ) リクエストや画像の符号化のたびに、トレース取得の後に新しくコンパイルしたプログラムや増えた DRAM がないかを調べ、あれば WARNING を出す (後のトレース再生でハングする原因になる) |
| `TSUGUMI_VISION_DEVICE=0` | ViT を CPU で動かす (fp32、ホストの RAM 約 1.8GB、1024×1024 で約 14 秒) |
| `TSUGUMI_IMAGE_MAX_PIXELS` / `TSUGUMI_IMAGE_MIN_PIXELS` | 画像を縮める上限 / 広げる下限のピクセル数 (既定 1,048,576 / 65,536)。上限を上げると起動時のウォームアップと DRAM の作業領域が増える |
| `TSUGUMI_IMAGE_CACHE_TOKENS` | 符号化済みの画像を保持するトークン数 (既定 65,536、1 リクエストの画像はその半分まで) |
| `TSUGUMI_IMAGE_FETCH=0` | http(s) の画像 URL を取りに行かない (data: URL だけ受け付ける) |

## 制限

- 全部 bfp4 の perplexity は bf16 比で日本語 +5%、英語 +7% (llama.cpp の IQ3_XXS 並み、Q4_0 は +2〜3%)。
  タスクでの品質 (ゼロショット、貪欲法、thinking なし。リリース前の内部確認で、モデルの性能の主張ではありません):

  | ベンチマーク | tsugumi (全部 bfp4) |
  |---|---|
  | JCommonsenseQA (1,119 問) | 95.3% (同じスクリプトの llama.cpp Q4_K_M: 95.0%) |
  | JSQuAD (500 問) F1 | 0.933 |
  | MGSM-ja (250 問) | 86.4% |
  | HumanEval / HumanEval+ | 93.3 / 89.0 |
  | GPQA Diamond (先頭 50 問、thinking あり) | 84% |
  | BFCL v4 non-live / live / multi-turn | 84.3 / 77.9 / 58.1 (bf16 のサードパーティ計測、thinking あり: 85.8 / 78.3 / 60.2) |
  | Japanese MT-Bench / ELYZA-tasks-100 (採点は Gemini Decision 1) | 8.46 / 10、4.18 / 5 |

  評価のスクリプトは `scripts/bench_quality.py`、`run_bfcl.py`、`judge_bench.py`。
- バッチは 1 (リクエストは 1 つずつ処理)。動画入力、`response_format` (構造化出力)、`n` > 1 は未対応。
- 会話の続きは再利用します (前回のプロンプト + 返答、またはアシスタントのターン開始位置の状態スナップショットから差分だけ prefill)。
  複数の会話 (本体とサブエージェントなど) は KV のブロックを分けて保持し、リクエストごとに切り替えます (約 0.1 秒)。
  KV の合計は最大文脈ぶん (262,144 トークン) で、足りなくなると最後に使ったのが最も古い会話から捨てます。
- 短い文脈では、GQA カーネルの固定費 (約 0.1ms/層) のぶん、ttnn の SDPA より 1 ステップ 1ms ほど遅くなります。
- デバイスが固まった場合は `tt-smi -r 0` を 2 回、15 秒ほど待つと戻ることが多いです。

## ファイル

| ファイル | 内容 |
|---|---|
| `tsugumi/model.py` | エンジン本体 (`Qwen35Engine`、`MTPHead`、T 行モード、トレース、prefill) |
| `tsugumi/generator.py` | MTP 投機的デコードの生成ループ |
| `tsugumi/ops.py` | DRAM シャーディングの bfp4/bfp8 行列積、重みの変換とキャッシュ |
| `tsugumi/gdn.py`、`tsugumi/gdn_kernel.py` | Gated DeltaNet 層と融合カーネルの起動 |
| `tsugumi/attn.py`、`tsugumi/gqa_attn.py`、`tsugumi/kv_kernel.py` | attention 層、GQA split-K アテンション、KV 書き込み |
| `tsugumi/vision.py`、`tsugumi/vision_tt.py` | 画像入力 (前処理、画像トークンの仮想 ID と M-RoPE の位置、デバイスの ViT) |
| `tsugumi/kvpager.py` | KV ブロックのプールとページ表 (複数の会話) |
| `tsugumi/prefill.py`、`tsugumi/conv_kernel.py`、`tsugumi/page_copy.py` | 大きい T の prefill、因果 conv カーネル、ページコピー |
| `kernels/*.cpp` | 自作カーネル (generic_op で起動) |
| `server.py` | OpenAI 互換サーバー |
| `run_server.sh` | サーバーの起動とハング後の自動復帰 (デバイスのリセット → 再起動) |
| `webui/index.html` | 試し用のチャット画面 |
| `scripts/` | 開発用のテスト・計測・perplexity 評価 (使い方は [`scripts/README.md`](scripts/README.md)) |

## ライセンス

Apache-2.0
