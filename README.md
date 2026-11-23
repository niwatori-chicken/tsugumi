# Qwen3.8-27B を Tenstorrent P100A 1 枚で動かす推論エンジン「tsugumi」

名前はツグミ (*Turdus eunomus*) に由来します。冬になるとシベリアから海を越えて日本に渡ってくる、身近な小さな鳥です。

27B の dense モデル (Gated DeltaNet 48 層 + full attention 16 層) を、Blackhole **P100A 1 枚 (28GB、448GB/s)** でできるだけ速く動かすためのtt-metal / ttnn 上に構築された推論エンジンです。

HTTP経由でOpenAI 互換サーバーとして使うことができます。

重みは全部bfp4に量子化してカード上に収め、モデル付属の MTP 層で投機的デコードを行います。

| | 速度 |
|---|---|
| デコード (コード生成、貪欲法) | **40〜50 tok/s** |
| デコード (コードの修正・文章の校正など、プロンプトを写す応答) | **75〜115 tok/s** (n-gram 下書き) |
| デコード (日本語の説明文 / 英語の物語) | 30~40 tok/s |
| prefill (数千トークンのプロンプト) | **約 1,500〜2,000 tok/s** (4,000 トークンで応答開始まで約 2 秒) |
| 最大文脈 | **262,144 トークン** (モデルの上限。KV は bfp8 で約 9GiBで、ボード上に収まります) |
| 12.7 万トークンのプロンプト | prefill 110 秒 (1,150 tok/s)、デコード 36 tok/s、「干し草山(Needle-in-a-haystack)」テスト合格 |
| 24.5 万トークンのプロンプト | prefill 307 秒 (800 tok/s)、デコード 30 tok/s、「干し草山」テスト合格 |
| 複数リクエストの同時処理 (`TSUGUMI_BATCH=8`、**P150A で計測**、条件は「複数リクエストの同時処理」) | 8 並列で**合計 152 tok/s** (1 リクエストあたりのデコード 22.6 tok/s)、4 並列で合計 102 tok/s。同じ条件で 1 本だけなら合計 37.9 tok/s |

- 投機的デコードの出力は、通常の貪欲法 (1 トークンずつ) の出力と一致します (コードのプロンプトで確認)。サンプリングも同じ分布です。
- 短いプロンプトでは最初のトークンまで約 0.6 秒です。

## 注意
この推論エンジンはTenstorrent社製の"Blackhole"アクセラレータ世代専用の推論エンジンです。**CUDA/ROCm/Vulkan/SYCLのようなGPU向けではありません。**

Wormhole世代のアクセラレータには対応していません。

この推論エンジンが対応するモデルは、今のところ**Qwen 3.8 27Bのみです。** 他のモデルについては、アップデートで対応予定です。

## 必要なもの

- Blackhole P100A (P150A でも動作を確認。バッチデコードの数値は P150A で計測)
- ホストの RAM: 32GB で動作確認
- tt-metal をソースからビルドしたもの (開発時は `v0.79.0-dev20260916-3-g5f02a6718f4`)。tt-metal の
  `models/` を import するので、`PYTHONPATH` に tt-metal のディレクトリが要る
- Python の venv (3.12 で確認) に、tt-metal の ttnn (ビルドしたソースから入れたもの)、`torch`、`transformers`
  (5.12 で確認)、`safetensors`、`Pillow` (画像入力)、サーバー用の `fastapi` と `uvicorn`。tt-metal の
  `create_venv.sh` で作った venv に後のものを足すのが簡単
- `tt-smi` (`run_server.sh` がハング後のリセットに、サーバーが `/health` のボード情報に使う。PATH になければ
  tt-installer が作る `~/.tenstorrent-venv/bin/tt-smi` を使う。別の場所なら `TT_SMI=/path/to/tt-smi`)
- モデル: Qwen3.8-27B の HF 形式 (bf16、約 56GB)
- 変換済み重みのキャッシュ用に約 16GB (bfp4)

## 使い方

### 基本の使い方
```bash
source /path/to/venv/bin/activate                     # ttnn を入れた venv (場所は環境による)
export TT_METAL_HOME=/path/to/tt-metal PYTHONPATH=/path/to/tt-metal
export TSUGUMI_MODEL_DIR=/path/to/Qwen3.8-27B             # HF 形式のモデル
export TSUGUMI_CACHE_DIR=/path/to/tsugumi_cache/Qwen3.8-27B    # 変換済み重みのキャッシュ (初回に作られる、約 8 分)

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

既定では認証はなく、127.0.0.1 だけで待ち受けます。`--host 0.0.0.0` などでネットワークに出すときはセキュリティ上、API キーを設定することを推奨します。

```bash
export TSUGUMI_API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')   # または --api-key KEY
./run_server.sh --host 0.0.0.0 --port 8010
# 複数のキー: TSUGUMI_API_KEY=key1,key2、またはファイルに 1 行 1 キー: --api-key-file keys.txt (TSUGUMI_API_KEY_FILE)
```

- **通信は暗号化しません。インターネットに出すときは、TLS を終端するリバースプロキシ (Caddy、nginx など) の後ろに置いてください。**
- キーを設定すると、`GET /` (Web UI のページ) と `GET /health` 以外は `Authorization: Bearer <キー>` が必要になり、
  ない・違うときは 401 を返します。OpenAI の SDK やクライアントは API キーの設定をこのヘッダで送ります。
- キーなしで 127.0.0.1 以外で待ち受けると、起動時に警告を出します。
- `/health` はキーなしでも答えますが、処理中のリクエストの詳細とボード情報は出しません。
- Web UI は最初の送信で 401 になるとキーを聞き、そのブラウザに保存します。

サーバー:

- `POST /v1/chat/completions`・`POST /v1/completions` (テンプレートなしの続き書き、プロンプトは文字列かトークン ID 列。どちらも stream 対応)、
  `POST /tokenize`・`/detokenize` (vLLM 形式)、`GET /v1/models`、`GET /health`。既定では 1 リクエストずつ処理します。
  `TSUGUMI_BATCH=4` / `8` で複数のリクエストを同時に処理します (下の「複数リクエストの同時処理」)。
- **Web UI**: テスト用の最小限のチャット画面 (`webui/index.html`)が用意されています。ブラウザから `http://127.0.0.1:8010/` にアクセスすると開けます。
  thinking、温度、最大トークン数、system プロンプトを切り替えられ、応答ごとにプロンプト側 (再利用、画像の読み込み、
  prefill のトークン数・秒・tok/s、応答開始) と生成側 (トークン数、tok/s、トークン/ステップ、思考) の数値を表示します。
  別のマシンから開くときは `--host 0.0.0.0` を引数に追加してください。(API キーの設定を推奨、上の「認証」)
- **画像入力**: `messages` の content に OpenAI 形式の `{"type": "image_url", "image_url": {"url": ...}}` (data: URL の base64、
  または http(s) の URL)。ViT (27 層、約 4.6 億パラメータ) はアクセラレータ上で動かします (重み bfp8 で DRAM 0.52 GiB、1024×1024 で
  0.17 秒。`TSUGUMI_VISION_DEVICE=0` で CPU、約 14 秒)。画像は
  100 万ピクセル (`TSUGUMI_IMAGE_MAX_PIXELS`) 以下に縮めて、32×32 ピクセルごとに 1 トークンとしています (最大 1,024 トークン)。
  同じ画像 (バイト列のハッシュ) は 2 回目から符号化し直すことはありません(ホストに 65,536 トークンぶん保持)。会話の再利用も効きます。
  画像トークンにはそれぞれの画像専用の仮想 ID (248,320 以上) を付与するので、トークン数が同じ別の画像を誤って再利用することはありません。
  Web UI は「画像」ボタン・貼り付け・ドラッグ&ドロップで添付することができます。動画は今後対応予定です。
- 思考は `chat_template_kwargs.enable_thinking` で切り替えることができます (既定 false)。`<think>` の中身は `reasoning_content` に分けて切り出します。
- 思考の上限は `thinking_budget` から設定できます (トークン数。`chat_template_kwargs` 内でも可、既定は `--thinking-budget` / `TSUGUMI_THINKING_BUDGET`、
  -1 で上限なし)。上限に達すると「時間がないのでここまでの考えで答える」旨の一文と `</think>` を差し込んで答えに移る
  (ステップの切れ目で判定するので最大 7 トークン超える)。上限を指定したときは `max_tokens` は答えの分だけを数えます。
- `logprobs` / `top_logprobs` (20 まで、completions は `logprobs: n`): デバイスの上位 64 の logits での log-softmax なので
  少し高めに出ます (64 の外は差し込んだトークンだけで -9999)。chat では本文のトークンだけです (思考は含まない)。
- サンプリングは `temperature` > 0 で `top_p`、`top_k` (64 まで)、`min_p`、`seed` が有効になります。`temperature` がなければ貪欲法となります。
  MTP の投機的デコードのまま厳密です (1 トークンずつ引いたのと同じ分布)。Qwen の推奨は thinking オフで 0.7 / 0.8 / 20、オンで 0.6 / 0.95 / 20です。
- ツール呼び出し: `tools` / `tool_choice` / 履歴の `tool_calls` と `role: tool` に対応しています (OpenAI 形式、ストリーミングも)。
  モデルが出す `<tool_call>` の XML を `tool_calls` に変換します (`tsugumi/toolcall.py`)。
  `tool_choice` は auto / none。required と関数名の指定は thinking オフのときだけ有効です (応答の先頭を `<tool_call>` にする)。
- ログ: `時刻 レベル [tsugumi] 本文` の形式で標準出力に出力します。起動時は設定の一覧と段階ごとの所要時間・DRAM の空き、
  リクエストごとに受付 (プロンプト長、パラメータ、無視したパラメータ) と完了 (再利用、応答開始、tok/s、終了理由) の 2 行を出力します。
  `TSUGUMI_LOG_LEVEL=DEBUG` で詳細なログを得ることができます。`run_server.sh` は tt-metal / ttnn のログを warning 以上に絞っています (`TT_LOGGER_LEVEL`、`LOGURU_LEVEL`)。
- `finish_reason` は `stop` / `length` (max_tokens で打ち切り) / `tool_calls`。`stop` (停止文字列、4 つまで) は応答本文にのみ有効で、思考は対象外です。
- モデル名は既定はモデルのディレクトリ名です (`Qwen3.8-27B`)。`--served-model-name tsugumi,gpt-4o` (または `TSUGUMI_SERVED_MODEL_NAME`) で上書き、
  カンマで別名をつけることができます。リクエストの `model` は何でも受け付けます (別名ならそれを返す)。`/v1/models` の `meta` から量子化と最大文脈長を取得できます。
- 終了 (SIGTERM / Ctrl-C): 受け付けを止め、生成中なら次のトークンで終えて応答を閉じ、デバイスを閉じて終了します (1 秒前後)。
- `GET /health` は処理中かどうか・待ちの数・処理中のリクエスト (生成済みトークン数、経過秒) を返します。待っている間にクライアントが
  切断したリクエストは順番が来ても生成しません。
- `/health` の `board` は `tt-smi -s` から読み取ったボードの状態 (チップと GDDR の温度、チップの電力 (TDP。上限 150 W、ボード全体の上限 300 W とは別で、
  ボード全体の電力は取れない)、ファン、コアクロック、熱トリップの回数、GDDR のエラー数)。サーバーが 10 秒ごとに読みとります (`TSUGUMI_BOARD_POLL`、0 で切る)。`age_s` は読んだときからの秒数です。
  tt-smi はエンジンがデバイスを使っている間もデバイスの状態を読み取ることができます (生成の結果・速度は変わらない)。tt-smi は `TT_SMI`、なければ PATH の
  `tt-smi`、`~/.tenstorrent-venv/bin/tt-smi` (tt-installer が作る環境) の順に探します。Web UI は右上に温度・電力・ファンを出し、マウスを載せると詳細を出せるようにしています
  (スロットル温度の 10°C 手前、クロック低下、熱トリップ、訂正不能エラーで色が変わる)。
- ハング検知: デバイスの処理が 30 秒進まないと (`TT_METAL_OPERATION_TIMEOUT_SECONDS`)、そのリクエストに 503 を返し、
  `/health` も 503 にして終了コード 75 で終了します。トークンが 300 秒出ないとき (`TSUGUMI_HANG_BACKSTOP`) も強制終了します。
  `run_server.sh` はそれを見て `tt-smi -r 0` を 2 回 → 再起動します (約 1.5 分で復帰。10 分に 3 回を超えたら停止します)。

### 複数リクエストの同時処理 (バッチデコード)

```bash
TSUGUMI_BATCH=8 ./run_server.sh --port 8010 --max-seq 32768
```

- デコードの 1 ステップ (約 60ms) の大半は重みの読み出しで、1 リクエストの 4 行 (現在のトークン + MTP の下書き 3) は
  32 行のタイルの一部しか使っていません。`TSUGUMI_BATCH=N` (最大 8) にすると、同時に来たリクエストを 4 行ずつの「レーン」として
  同じステップに詰め、重みの読み出しを共有します。
- P150A での実測 (計測はそれぞれ 1 回):

  | 同時リクエスト | 合計 | 1 リクエストあたりのデコード |
  |---|---|---|
  | 1 | 37.9 tok/s | 40.6 tok/s |
  | 4 | 102.2 tok/s (1 本の 2.7 倍) | 29.7 tok/s |
  | 8 | 152.5 tok/s (1 本の 4.0 倍) | 22.6 tok/s |

  条件: `TSUGUMI_BATCH=8 --max-seq 32768`、`TSUGUMI_IMAGES=0 TSUGUMI_LARGE_PREFILL=0 TSUGUMI_VERIFY_T=0`、
  `scripts/bench_concurrent.py --clients 1,4,8 --max-tokens 512 --rounds 2` (各クライアントが固定のプロンプトで 2 リクエストずつ、貪欲法)。
  「合計」は生成トークン数の合計 / 実時間 (新しいプロンプトの prefill や待ちを含む)、「1 リクエストあたり」はサーバーが返す
  `decode_tps` (最初のトークン以降の速度) の平均なので、両者の比は並列数と一致しません。
  別の実行 (`TSUGUMI_BATCH=4`、`--max-tokens 256`) では 1 / 2 / 4 並列で合計 35.5 / 58.6 / 85.2 tok/s (4 並列で 2.4 倍)。
  ステップの時間 (`scripts/test_batch.py`、エンジン直接、256 トークン) は 1 レーン 65ms、2 レーン 71ms、4 レーン 84ms。
- 出力は 1 リクエストずつ処理したときと同じです (貪欲法で 4 並列の出力がトークン単位で一致することを確認。`scripts/test_batch.py`)。
  サンプリング (seed)・思考の上限・logprobs・停止文字列・`/v1/completions`・途中切断はバッチのモードで動作を確認しました。
  ツール呼び出しは応答テキストの解析なので変わらないはずですが、バッチのモードでは未確認です。
- 新しいリクエストのプロンプトは従来どおり 1 本ずつ prefill し、その間も 0.15 秒ごと (`TSUGUMI_PREFILL_SLICE`) に他のレーンの
  ステップを挟みます (大きい T の prefill は 8,192 トークンごと。ただし `TSUGUMI_LARGE_PREFILL=1` や画像入力との組み合わせはバッチのモードでは未確認)。
  終わった会話の状態はホストに保存し、続きのリクエストで再利用します。
- `--max-seq` は全リクエスト (と保持する会話) で共有する KV の総量です。レーン 1 つにつき GDN の状態のスナップショットで
  DRAM を約 0.6 GiB 使うので、最大文脈 262,144 とは両立しません (実測: 8 レーン + `--max-seq 32768` で DRAM の空き 9.3 GiB)。
- バッチのモードでは n-gram 下書き (T=8 の検証) は使いません (MTP の下書きのみ)。n-gram 下書きが効く応答 (プロンプトを写す応答) では
  1 本ずつの処理より遅くなる見込みです (未計測)。
- `/health` は処理中のリクエストの一覧 (`active`) とレーン数を返します。`/health?sched=1` でスケジューラの平均時間 (ステップ、受付、終了) を返します。
- 同時接続の計測: `python scripts/bench_concurrent.py --clients 1,4,8 --max-tokens 512 --rounds 2`

TSUGUMIサーバーが起動時に読み取る環境変数

| 変数 | 意味 |
|---|---|
| `TSUGUMI_MODEL_DIR` / `TSUGUMI_CACHE_DIR` | モデルと変換済み重みの場所 (`tsugumi/paths.py`) |
| `TSUGUMI_API_KEY` | APIにアクセスする際に利用するAPIキー。デフォルトでは未設定です。 |
| `TSUGUMI_KV_DTYPE=bf16` | KV キャッシュを bf16 にする (既定 bfp8。bf16 では max_seq 155,648 まで) |
| `TSUGUMI_BATCH` | 同時に処理するリクエストの数 (レーン数、1〜8)。既定 1 (1 リクエストずつ)。レーン 1 つにつき DRAM 約 0.6 GiB |
| `TSUGUMI_PREFILL_SLICE` | バッチのモードで、新しいプロンプトの prefill 中に他のレーンのステップを挟む間隔 (秒、既定 0.15) |
| `TSUGUMI_TRACE_MB` | トレース領域の大きさ (MiB、既定 512、バッチのモードでは 1024) |
| `TSUGUMI_MULTI_CONV=0` / `TSUGUMI_MAX_CONVS` | 複数の会話を保持しない / 保持する会話数の上限 (既定 8、ホストに 1 会話約 180MB) |
| `TSUGUMI_SNAPSHOT=0` | 会話の再利用用の状態スナップショットを取らない |
| `TSUGUMI_SERVED_MODEL_NAME` | サーバーが返すモデル名称を上書きする。デフォルトはモデルのディレクトリ名です。 |
| `TSUGUMI_HANG_BACKSTOP` | トークンがこの環境変数以上の秒数出力されない場合、強制終了する。デフォルトは300秒です。 |
| `TSUGUMI_BOARD_POLL` | サーバーが`tt-smi`からボードの状態を読み取る間隔。デフォルトは10秒です。 |
| `TSUGUMI_THINKING_BUDGET` | モデルが思考(CoT)に使えるトークン数の上限。デフォルトは無制限です。 |
| `TSUGUMI_IMAGES=0` | 画像入力を無効にする (ViT を読み込まない。DRAM 0.52 GiB の節約) |
| `TSUGUMI_VISION_DEVICE=0` | ViT を CPU で動かす (fp32、ホストの RAM 約 1.8GB、1024×1024 で約 14 秒と遅くなります) |
| `TSUGUMI_IMAGE_MAX_PIXELS` / `TSUGUMI_IMAGE_MIN_PIXELS` | 画像を縮める上限 / 広げる下限のピクセル数 (既定 1,048,576 / 65,536)。上限を上げると起動時のウォームアップと DRAM の作業領域が増えます。 |
| `TSUGUMI_IMAGE_CACHE_TOKENS` | 符号化済みの画像を保持するトークン数 (既定 65,536、1 リクエストの画像はその半分まで) |
| `TSUGUMI_IMAGE_FETCH=0` | http(s) の画像 URL を取りに行かないようにする (data: URL だけ受け付ける) |

デバッグ用の環境変数については、[debugging.md](./debugging.md)に記載しています。

## 仕組み

1 トークンあたりの時間の大半は重みの読み出しなので、(1) 読む量を減らし、(2) 1 回の読み出しで複数トークンを処理し、
(3) 行列積以外の小さな op の固定費を削る、の 3 つで速くしています。

- **bfp4 の DRAM シャーディング行列積** (`tsugumi/ops.py`): 重みを 7 バンクに分けて流し読み。
  ワーカー数 3・in0_block_w=2 で公称帯域の 93% (約 415GB/s)。全部 bfp4 で行列積だけ約 38ms/ステップ。
- **GDN 融合カーネル** (`kernels/gdn_fused_*.cpp`、`tsugumi/gdn_kernel.py`): conv 以降の
  L2 正規化 → T トークン分の DeltaNet 再帰 → gated RMSNorm を 1 本のカーネルで行う (値ヘッド 1 つに 1 コア)。
  各トークン後の状態をスロットに残すので、投機的デコードで「確定した位置」から次を再開できる。
- **T 行モード** (`forward_spec`): T 行のトークンをまとめて流す。検証は T=4 (n-gram 下書きのときは T=8)
  conv の履歴はホストが書く 0/1 の選択行列との行列積でずらすので、トレースを固定したまま採択数に追従できる。
- **大きい T の prefill** (`tsugumi/prefill.py`): プロンプトは最大 1024 行ずつ、`minimal_matmul` (同じ bfp4 重み)、
  チャンク並列の GDN (`ttnn.transformer.chunk_gated_delta_rule`)、自作の因果 conv カーネル (`kernels/conv_*.cpp`)、
  chunked SDPA で流す。終わったら GDN の状態と conv 履歴を T 行モードの形式に戻して、そのままデコードに入る。
- **MTP 投機的デコード** (`MTPHead`、`tsugumi/generator.py`): 付属の MTP 層で 3 トークン先まで下書きし、本体で一括検証。
  下書き用には lm_head の bfp4 の複製を使う。
- **GQA split-K アテンション** (`kernels/gqa_attn_*.cpp`、`gqa_comb_*.cpp`、`tsugumi/gqa_attn.py`):
  KV ヘッド 4 × 区間 27 = 108 コアが、担当区間の KV を 1 回だけ読み、同じ KV ヘッドを使う 6 q ヘッド × T 行を
  まとめて計算 (online softmax)。部分結果を別の op で合成する。区間とチャンク数は実行時に位置から決まるのでトレース可能。
  12.4 万トークンの文脈で 4 行の検証が 1 層 1.7ms (DRAM 帯域の下限 1.27ms 付近)。
  ttnn の decode SDPA に T 行を「T ユーザー」として渡す方式 (5.6ms) や chunked SDPA (32 行で 15ms) の置き換え。
- **KV 書き込みカーネル** (`kernels/kv_write.cpp`): T 行の K/V をページ KV に競合なく書く。
- 残差ストリームは 40 コアの幅方向シャードで持ち、RMSNorm もシャード版 (89µs → 6.5µs)。argmax はデバイス上で行う。
- **バッチデコード** (`tsugumi/batch.py`): 32 行を最大 8 レーン × 4 行に分け、レーンごとに別の会話を流す。行列積はそのまま、
  GDN 融合カーネルはレーンごとに自分の状態スロットから再帰を始め (`seg`)、GQA カーネルと KV 書き込みカーネルはレーンごとの位置と
  ページ表を読む (1 つのプログラムで全レーン。GQA の分割はレーンで分け合う)。conv の履歴もレーンごとの選択行列でずらす。
  レーン数 1, 2, 4, 8 ごとにトレースを取り、使っている最大のレーンを含む最小のモードで回す。

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
- 既定ではリクエストを 1 つずつ処理する (`TSUGUMI_BATCH` で同時処理)。動画入力、`response_format` (構造化出力)、`n` > 1 は未対応。
- 会話の続きは再利用します (前回のプロンプト + 返答、またはアシスタントのターン開始位置の状態スナップショットから差分だけ prefill)。
  複数の会話 (本体とサブエージェントなど) は KV のブロックを分けて保持し、リクエストごとに切り替えます (約 0.1 秒)。
  KV の合計は最大文脈ぶん (262,144 トークン) で、足りなくなると最後に使ったのが最も古い会話から捨てます。
- 短い文脈では、GQA カーネルの固定費 (約 0.1ms/層) のぶん、ttnn の SDPA より 1 ステップ 1ms ほど遅くなります。

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
| `tsugumi/batch.py` | バッチデコード (レーン、状態の受け渡し、複数リクエストのスケジューラ) |
| `tsugumi/board.py` | ボードの温度・電力・ファン (`tt-smi -s` を定期的に読む、`/health` の `board`) |
| `tsugumi/prefill.py`、`tsugumi/conv_kernel.py`、`tsugumi/page_copy.py` | 大きい T の prefill、因果 conv カーネル、ページコピー |
| `kernels/*.cpp` | 自作カーネル (generic_op で起動) |
| `server.py` | OpenAI 互換サーバー |
| `run_server.sh` | サーバーの起動とハング後の自動復帰 (デバイスのリセット → 再起動) |
| `debugging.md` | デバッグのためのヒント、および環境変数 |
| `webui/index.html` | テスト用のチャット画面 |
| `scripts/` | 開発用のテスト・計測・perplexity 評価 (使い方は [`scripts/README.md`](scripts/README.md)) |

## ライセンス

Apache-2.0
