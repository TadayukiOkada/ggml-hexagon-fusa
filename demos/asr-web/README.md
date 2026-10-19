# ローカルASR比較デモ

NeMoの途中認識とSenseVoiceの区間認識を同じ音声で比較する、日本語のローカルデモです。
`mock` は固定の試験結果を返します。音声認識は行いません。
実推論では別プロセスの常駐サーバへ接続します。ブラウザからクラウドへ音声を送りません。

## PCモック

このディレクトリで実行します（Python 3.10以上）。

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python gateway.py --port 8765
```

ブラウザで `http://localhost:8765/` を開き、マイクを許可します。
別PCからはSSHでブラウザPCのlocalhostへポート転送してください。
マイクの開始・停止・取り消し、NeMo/SenseVoice/両方、入力加工、故障シナリオを選べます。
EVK/SA8255Pは設定・到達確認ができるまで無効です。

- 最大30秒。マイク音声をAudioWorkletで収録し、状態を保持する低域フィルタ付きリサンプルで16kHz mono PCM16へ変換します。
- 20ms単位で送り、末尾は停止時にflushします。サンプル数が時間の基準です。
- WAV入力は16kHz / mono / PCM16、最大30秒です。ファイルも同じWebSocket経路で送ります。
- 入力音声はメモリ内だけに保持します。音声WAVを保存する場合は明示的に保存ボタンを押します。
- 結果JSONにはMOCK識別、セッション・世代、エンジン別結果、PCMサンプル数・SHA-256、ブラウザ内経過時間を含みます。
- 故障シナリオはモックの意味上の故障です。実ネットワークの破損・切断を注入するものではありません。
- 取り消しは旧セッションの表示を無効にします。実サーバの同期計算は安全な境界まで継続することがあります。

## 常駐ネイティブサーバ

FuSaのビルドへ任意のHTTPサーバを追加します。既定はどちらもOFFです。

```sh
cmake --preset arm64-oe-linux-v73-fusa -B <build> \
  -DGGML_HEXAGON_ASR=ON -DGGML_HEXAGON_ASR_HTTP=ON \
  -DGGML_HEXAGON_SENSEVOICE=ON -DGGML_HEXAGON_SENSEVOICE_HTTP=ON
cmake --build <build> -j 8
```

NeMoのサーバが使うcpp-httplibはNeMoのpinと同じcommitで取得します。初回はネットワークが必要です。SenseVoiceのサーバはSenseVoiceに同梱の`httplib.h`を使うため、取得は不要です。
submoduleの作業ツリーは変更しません。SenseVoiceのCLIと常駐サーバは、`sensevoice/patches/0002-runtime-library-and-server.patch`でCLIから分けた共通runtime(frontend、グラフ、scheduler、CTC、detokenize)を使い、サーバはモデルを常駐させます。25 ms未満の録音は空の最終結果になります。
SenseVoice CLIの従来の出力・`--ids`・`--keep-tags`を保持し、`--json`を追加しています。

NeMoは下記のYAMLをローカル設定として用意します（`serve`専用）。

```yaml
http:
  enabled: true
  host: 127.0.0.1
  port: 8766
  threads: 4
  max-upload-mb: 1
  playground: false
asr:
  enabled: true
  backend:
    name: cpu
    frontend: cpu
    position_backend: cpu
    threads: 4
  batching:
    enabled: false
    state_arena_slots: 1
  streaming:
    rnnt_right_context: 3
```

```sh
<NeMo-bin>/nemo-speech serve --config <serve.yaml> --asr-model <model.gguf> --no-warmup
<SenseVoice-bin>/sensevoice-server -m <model.gguf> --backend cpu --threads 4 --port 8767
python gateway.py --config profiles.example.json --port 8765
```

各実行物には**それぞれのビルドのggmlライブラリ**を使います。同名のggmlを一つの探索パスへ混ぜないでください。
HTTP有効の配布物には `demos/asr-web/` とPC用 `run-asr-web.sh` を同梱します。PC上で `pip install -r demos/asr-web/requirements.txt` の後、`./run-asr-web.sh --config <local-profiles.json>` で起動できます。
配布物の `run.sh nemo-speech ...` / `run.sh sensevoice-server ...` はライブラリを振り分けます。
NeMoのHTP利用時は `asr.backend.name: hexagon`、`device`、`library_path`を指定し、frontend/positionをCPUに維持します。
SenseVoiceは `GGML_HEXAGON_DEVICES=HTP1` を環境に設定し、`--backend hexagon --backend-lib <libggml-hexagon.so> --device HTP1` を指定します。
ホストライブラリとskelは同じビルドIDの組を使用してください。
実機での利用前に、その機器の利用規則・空き状況・検証段階を確認してください。

接続設定は `profiles.example.json` をコピーして機器情報を含まないlocalhostのトンネル宛先を指定します。
許可されるprofile名は `cpu` / `evk` / `sa8255p`、URLはloopback HTTP originのみです。
接続設定を公開リポジトリへ追加しないでください。

## API

- `GET /api/capabilities`: profileごとのready（実サーバの `/ready` を確認）、エンジン情報、音声上限。
- `WS /api/session`: `start` → `session.ready` → binary PCM → `stop` → エンジン別final/done → `session.done`。
  `start` は `generation`（整数）、`profile`、`engines`、`sample_rate:16000`を持ちます。
  stop/cancelには返却された `session_id` と同じgenerationが必須です。
- `POST /api/transcribe?profile=mock&engines=nemo,sensevoice`: 同形式のWAVを送り結果イベントをJSONで取得します。
  ブラウザのWAV入力はこのRESTではなくWSを利用します。
- 1WSにつき1収録。エンジン別送信待ち上限2秒。overflowをエラーとして表示し、音声を黙って捨てません。
- ネイティブのSenseVoiceは `/ready`、`WS /api/session`（binary PCM、stop/cancel）、`POST /api/transcribe`（WAVではなくraw標準PCM）を提供します。
- ネイティブNeMoのWSは全文 `transcript` / `segment_id` / `revision` を返すパッチが必須です。旧deltaを推測で連結しません。
- gatewayは同一OriginのWS/POSTだけを受け付け、loopbackへbindします。

## 検証

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/test_audio.mjs
```

任意のブラウザ試験（実マイクを使わず仮想マイクを使用）:

```sh
pip install playwright==1.56.0
python tests/browser_smoke.py --url http://localhost:8765 \
  --browser <Chrome-or-Chromium> --output <scratch-directory>
```

実マイクの許可・デバイス選択・音声品質は仮想マイク試験と別の確認です。
CPUでの成功はHTP/FuSaの検証結果を代用しません。新サーバは既定環境へ直接上書きせず、隔離配置で検証します。

ネイティブサーバ起動後のファイル配信確認:

```sh
python tests/native_smoke.py --profile cpu --wav <input.wav> --output <results.json>
```

これは配信数・hash・完了の確認です。CLIとの本文・ID・タグ照合、実HTPログ、EVKの待ち時間、SAのSTL確認はそれぞれ別途行います。

応答の順序・古い収録世代をブラウザだけで検証する場合:

```sh
python tests/browser_ordering.py --url http://localhost:8765 --browser <Chrome-or-Chromium>
```

全選択エンジンが起動できない場合は、マイク取得前に開始を拒否します。
片側だけの失敗はもう片側の結果と分けて表示し、native側の診断メッセージは結果JSONと画面へ残します。
