# zero_one_instruction  — vLLMを用いたSFTデータ生成ツール

## 概要

このツールは、[vLLM](https://github.com/vllm-project/vllm)を利用して、マルチステップのプロンプト進化プロセスを経て、Supervised Fine-Tuning（SFT）用の高品質なデータセット（質問と回答のペア）を自動生成します。

設定ファイル（`model_config.yml`）により、使用する言語モデルや推論パラメータを柔軟に変更できます。

## 機能

- **プロンプト進化:**
  - **Width:** 元のプロンプトを基に、同じテーマでよりユニークな新しいプロンプトを生成します。
  - **Depth:** 生成されたプロンプトに新たな制約や要件を追加し、より複雑で深みのある指示に書き換えます。
  - **Flatten:** プロンプトを理路整然とした質問形式に修正し、誤字脱字や曖昧さを解消します。
- **品質判定（Judge）:** 生成されたプロンプトが自然な日本語で、明確な意図を持ち、安全かつAIが回答可能な範囲内にあるかを判定します。
- **回答生成:** 進化・判定を経た最終的なプロンプトに対して、指定されたモデルが回答を生成します。
- **バッチ処理:** 複数のベースプロンプトをまとめて処理し、効率的にデータセットを生成します。
- **設定ファイル:** `model_config.yml`でモデル、量子化、GPU設定、バッチサイズなどを一元管理します。

## 動作環境

- Python 3.8以上（推奨）
- NVIDIA GPU（vLLMの実行に必須）
- 必要なPythonライブラリ（詳細は`requirements.txt`を参照）
  - `vllm`
  - `transformers`

## インストール方法

1. **リポジトリのクローン:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. **Python仮想環境の作成と有効化（推奨）:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOSの場合
    # venv\Scripts\activate  # Windowsの場合
    ```

3. **依存ライブラリのインストール:**
    ```bash
    pip install -r requirements.txt
    ```
    **注意:** vLLMのインストールには時間がかかることがあり、環境によっては追加の依存関係が必要です。詳細は[vLLM公式ドキュメント](https://docs.vllm.ai/en/latest/getting_started/installation.html)を参照してください。

## 設定

実行前に`model_config.yml`を編集して、環境や目的に合わせて設定を調整してください。

設定例:
```yaml
model_name: "OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym"
quantization: "awq"  # 不要ならnullを指定
gpu_memory_utilization: 0.7
tensor_parallel_size: null  # 自動設定はnull
batch_size: 32
max_tokens: 512
max_model_len: 4096
trust_remote_code: True
dtype: "auto"
temperature: 0.9
top_p: 0.9
```

各項目の詳細は元の説明を参照してください。

## 使い方

SFTデータ生成プロセスは、単一プロセスでの`main.py`実行、または複数のGPUを利用するためのラッパースクリプト`auto_run.py`を使用して開始します。

### 1. `main.py`を直接実行（単一プロセス）

```bash
python main.py --start_idx <開始インデックス> --end_idx <終了インデックス> --output_file <出力ファイル名> [オプション]
```

コマンドライン引数についての詳細は元の説明を参照してください。

### 実行例

- インデックス0〜99を処理する場合:
```bash
python main.py --start_idx 0 --end_idx 100 --output_file output.jsonl
```

- インデックス100〜199を処理し、IDを100から開始する場合:
```bash
python main.py --start_idx 100 --end_idx 200 --output_file output_part2.jsonl --id_start 100
```

### 2. `auto_run.py`を使用（マルチGPU並列実行）

`auto_run.py`は、複数GPU環境で効率的に`main.py`を並列実行するためのラッパースクリプトです。

```bash
python auto_run.py --num_prompts <総プロンプト数> --output_file <最終出力ファイル名> [オプション]
```

コマンドライン引数および動作の詳細は元の説明を参照してください。

### 実行例

- 利用可能な全GPUで10万プロンプトを処理する場合:
```bash
python auto_run.py --num_prompts 100000 --output_file my_output.jsonl
```

- GPU数を7個指定する場合（4, 2, 1のGPUグループに分割されます）:
```bash
python auto_run.py --num_gpus 7 --num_prompts 100000 --output_file my_output.jsonl
```

## 出力形式

指定した`--output_file`にJSONL形式で出力されます。

出力例:
```json
{
  "id": "record_0",
  "input": "最終的に生成された質問プロンプト",
  "output": "モデルが生成した回答",
  "conversation": [
    {"from": "system", "value": "あなたは優秀な日本語AIアシスタントです。ユーザーの質問に対して正確かつ簡潔に回答します。"},
    {"from": "human", "value": "最終的に生成された質問プロンプト"},
    {"from": "gpt", "value": "モデルが生成した回答"}
  ]
}
```

## 注意点

- **GPU要件:** vLLMはNVIDIA GPUが必須です。必要メモリ量はモデルサイズによります。
- **ベースプロンプト:** 実際のユースケースに合わせて、`main.py`内のベースプロンプトを編集してください。
- **プロンプトテンプレート:** 各プロンプト進化ステップ用のテンプレート（`PROMPT_EVOL_WIDTH`等）は、必要に応じてカスタマイズ可能です。
- **エラーハンドリング:** 推論エラー時の出力は`"<GEN_ERROR>"`となり、後続の処理から除外されます。また、品質基準を満たさないプロンプトも最終出力から除外されます。