# scripts — テスト・計測・評価

開発用のスクリプトです。サーバーを動かすだけなら使いません。

どれも `lowbit/` を import パスに足すので、`python scripts/<name>.py` で、どのディレクトリからでも実行できます。

## 実行環境

環境は親の README の「使い方」と同じです。

- ttnn の venv
- `TT_METAL_HOME`
- `TSUMI_MODEL_DIR` / `TSUMI_CACHE_DIR`

デバイスは 1 プロセスしか掴めないので、サーバーは止めてから実行してください。

## 生成と速度

| スクリプト | 内容 |
|---|---|
| `run_mtp.py --prompt "..."` | MTP 投機的デコード (貪欲法) で 1 プロンプト生成し、速度と位置ごとの採択率を表示 |
| `run_spec_test.py` | T 行モード: chunked prefill と、1 行ずつ確定する貪欲デコード |
| `run_gen.py` | T=1 の素のデコードで生成 (初期の経路) |
| `longctx.py N [max_new] [質問]` | 長文脈の干し草山テスト (needle in a haystack) と速度。<br>`TSUMI_MAX_SEQ`、`TSUMI_LARGE_PREFILL`、`TSUMI_LARGE_T` で設定を変える |

## 正しさのテスト

| スクリプト | 内容 |
|---|---|
| `cmp_hf.py` | 先頭 N 層 (+ 最終ノルム + lm_head) のロジットを HF (CPU fp32) と比較 |
| `test_vision_tt.py [WxH ...]` | デバイスの ViT を HF の CPU 版 (fp32) と比較 (PCC、速度、DRAM)。`VT_BF16=1` で重みを bf16 に |
| `test_vision_order.py [--no-warmup]` | サーバーと同じ構成 (トレースあり) で画像の符号化と生成を交互に。`--no-warmup` はハングの再現 |
| `cmp_hf_mm.py` | 画像入力: M-RoPE の位置と cos/sin を HF と比較、画像入りのプロンプトで先頭 N 層のロジットを HF と比較 |
| `test_gdn.py` | GDN のデコード (T=1) を HF の逐次再帰と比較 (乱数の重み) |
| `test_gdn_kernel.py` | GDN の再帰カーネル (自作) を torch fp32 と比較 |
| `test_gdn_fused.py T [steps]` | 融合 GDN カーネル (conv 以降) を torch と比較 |
| `test_attn_spec.py T N [chunked]` | attention 層: T=1 の逐次デコードと T 行のページ KV 経路を比較 (層 3 の実重み) |
| `test_kv_bfp8.py [max_seq]` | bfp8 の KV: 書き込みカーネルがホストの変換とビット単位で一致すること (タイル・ブロック境界、シャッフルしたページ表)、bfp8 の KV での GQA アテンション |
| `test_gqa.py [max_seq]` | GQA split-K アテンション (自作) を torch と比較 (物理ブロックをシャッフルしたページ表で) |
| `test_attn_large.py [T] [N]` | 大きい T の attention を torch fp32 と比較。<br>q/k (RoPE 後) と K キャッシュは要素ごとにも比較 |
| `test_conv.py` | 因果 conv カーネル (大きい T の prefill 用) を torch fp64 と比較 |
| `test_prefill_large.py [N] [t_max] [max_new]` | 大きい T の prefill と T=32 の経路を比較 (次トークン、MTP 下書き、続きの生成、速度) |
| `test_multiconv.py` | 複数の会話を交互に流しても単独と出力が一致すること、切り替え時間、KV が足りないときの追い出し |
| `test_reuse.py [--off]` | 会話の再利用: 3 ターン (thinking オフ/オン) の再利用の種類、応答開始までの時間、最初からとの出力比較 |
| `test_prefill_cache.py` | トレース取得後に、開始位置・長さがばらばらの prefill とデコードを繰り返しても、プログラムキャッシュが増えないこと (増えるとハングの原因になる) |
| `test_hang.py [spin]` | 終わらないカーネルを流し、ホストの待ちがどう終わるか (例外 / 保険) を見る。終わったらデバイスをリセット |
| `test_server_hang.sh [port]` | `run_server.sh` 経由のサーバーで、通常応答 → `/debug/hang` → 503 → 自動リセット・再起動 → 通常応答。<br>`BACKSTOP=20 TT_METAL_OPERATION_TIMEOUT_SECONDS=0` で保険の経路 |
| `test_sampling.py [N]` | サンプリング: seed の再現性、投機的サンプリングと 1 トークンずつの分布の比較 (TV 距離)、速度と採択率。<br>`EXACT_PROMPT=...` で比較用のプロンプトを変える |
| `test_toolcall.py` | ツール呼び出しの解析 (デバイス不要): ストリームの全分割で同じ結果、型変換、途中で切れた呼び出し、テンプレートとの往復 |
| `test_tools.py [URL]` | 起動中のサーバーに対して、架空のツールでエージェントのループを回す (並列呼び出し、stream、thinking、tool_choice) |
| `test_api2.py [URL]` | 起動中のサーバーに対して、思考の上限・logprobs・`/v1/completions`・`/tokenize` (26 項目) |
| `test_images.py [URL]` | 起動中のサーバーに対して、画像入力 (図形の色と形、文字の読み取り、2 枚の比較、会話の再利用、stream、エラー) |
| `prefill_agree.py OUT.pt [テキスト]` | prefill 中の位置ごとの argmax を保存する (prefill 経路どうしの比較用) |

## 計測

| スクリプト | 内容 |
|---|---|
| `profile_step.py [T] [P0] [--traced]` | 1 ステップ (本体の検証 + MTP) の op ごとのデバイス時間。`python -m tracy -r -p` で実行し、`TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=6000` を付ける。`--traced` はトレース実行の時間だけ |
| `summarize_profile.py [--segments N] [--seq]` | 上のプロファイルを op 名ごとに集計 (`--seq` で op の並び) |
| `bench_decode.py [max_new] [--copy\|--all] [--dump FILE]` | 決まったプロンプト集でのデコード速度 (日英の文章、コード、thinking、`--copy` で写しの多い応答)。`--dump` で出力を保存 (下書き方式のオフライン試算用) |
| `make_draft_vocab.py [N]` | MTP 下書き用の語彙表 `tsumi/draft_vocab.pt` を作る (分野ごとに重みを揃えた頻度)。評価用テキストでのカバー率も表示 |
| `mmbench.py` | デコード用行列積 (DRAM シャーディング) の実効帯域。ワーカー数・in0_block_w の掃引 |
| `mmlarge.py [M ...]` | 大きい M の行列積 (`minimal_matmul` / `ttnn.linear`) を bfp4 重みで計測 |
| `bench_parts.py` | デコード 1 ステップの部品ごとのデバイス時間 |
| `bench_spec_parts.py` | T 行モードの部品ごとのデバイス時間 |
| `bench_gqa.py [P0]` | GQA アテンションと ttnn の decode SDPA を、長い文脈で比較 |
| `bench_cgdr.py [T ...]` | `chunk_gated_delta_rule` の精度 (fp64 の逐次再帰との比較) と速度 |
| `bench_prefill_large.py [T] [P0]` | 大きい T の prefill 1 チャンクの部品ごとの時間 |
| `sdpa_start.py` | chunked SDPA の開始位置: 整数とテンソルでのプログラム数・速度・出力の比較 |
| `sdpabench.py` | chunked SDPA の速度 (k_chunk 依存) と、開始位置がずれたときの正しさ |
| `memprobe.py MAX_SEQ [--no-large]` | サーバー構成で読み込んだときの DRAM 使用量 (段階ごと、最大の連続空きも) |
| `quantstudy.py` | 重みの量子化誤差 (bfp4/bfp8 などの TT 形式、CPU) |

## 品質ベンチマーク (todo 2)

| スクリプト | 内容 |
|---|---|
| `bench_quality.py TASK OUT.jsonl [--n N] [--prec ..] [--set k=v ..] [--cache ..] [--think]` | jcqa / jsquad / jemhop / mgsm / humaneval / gpqa を貪欲法・ゼロショットで解かせて採点。<br>途中から再開できる。データは `/mnt/HDD/eval/bench` |
| `run_quality.sh [variant ..]` | 量子化の設定 (bfp4 / attn8 / mixed) ごとに全タスクを流す夜間用。<br>`GPQA=1` で GPQA Diamond (thinking オン)。HumanEval+ は EvalPlus (別の venv) で採点 |

## perplexity

| スクリプト | 内容 |
|---|---|
| `eval_ppl.py IDS.pt OUT.pt [--prec bfp4] [--set k=v ...] [--large 1024]` | エンジンの PPL。<br>llama-perplexity と同じ手順で、各チャンクを新しい状態から流し、後半だけを採点する。<br>`--large` で大きい T の prefill 経路を使う |
| `ref_ppl.py IDS.pt OUT.pt` | CPU の基準。bf16 の重みを fp32 で計算し、1 層ずつ読み込む (全 64 層で約 46 分) |
| `compare_ppl.py REF.pt RUN.pt ...` | PPL、基準との KL (上位 K 語で近似)、top-1 一致率 |

結果と経緯は `../../doc/11-b1-qwen38-27b-engine.md` にあります。

- `translate_md.py`: translate a Markdown article (en -> ja) block by block with the PLaMo 3 Translate server (port 8000); keeps code, comments, placeholders; `--bilingual` keeps the English as comments.
