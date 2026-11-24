# tsugumiのデバッグ

## デバッグ用環境変数

モデル推論、または開発段階で問題が発生した場合、これらの環境変数が原因の切り分けに役立つかもしれません。

| 変数 | 意味 |
|---|---|
| `TSUGUMI_LOG_LEVEL=DEBUG` | サーバーから詳細なログを出力する |
| `TSUGUMI_PREFILL_SDPA=hifi4` | prefill の attention を q chunk 128 + HiFi4 に戻す (既定は 256 + HiFi2) |
| `TSUGUMI_LARGE_PREFILL=0` / `TSUGUMI_LARGE_T` | 大きい T の prefill を使わない (T=32 の旧経路) / 1 チャンクの最大行数 (既定 2048。1024 なら作業用の DRAM が 124 MiB 減るが、prefill は 1〜3 割遅い) |
| `TSUGUMI_GQA_ATTN=0` | 自作の GQA アテンションを使わず、ttnn の SDPA に戻す |
| `TSUGUMI_CHUNKED_PREFILL=0` | (GQA を切ったとき) prefill で chunked SDPA を使わない |
| `TSUGUMI_FUSED_GDN=0` | GDN 融合カーネルを使わず、元の op の並びで動かす |
| `TSUGUMI_NGRAM=0` / `TSUGUMI_NGRAM_MIN` | n-gram 下書き (プロンプト参照) を使わない / 使う条件の一致長 (既定 6 トークン) |
| `TSUGUMI_VERIFY_T` | n-gram 下書きを検証する長いモードの行数 (既定 8、0 で T=4 のまま) |
| `TSUGUMI_DRAFT_VOCAB=0` | MTP の下書きを全語彙で出す (既定は頻出 63,168 語、`tsugumi/draft_vocab.pt`) |
| `TSUGUMI_CAPTURE_GUARD=1` | (デバッグ) リクエストや画像の符号化のたびに、トレース取得の後に新しくコンパイルしたプログラムや増えた DRAM がないかを調べ、あれば WARNING を出す (後のトレース再生でハングする原因になる) |


デバイスが固まった場合は `tt-smi -r 0` を 2 回、15 秒ほど待つと戻ることが多いです。
