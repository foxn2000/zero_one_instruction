import json
import argparse
from typing import List, Dict

import yaml
from tqdm import tqdm

# ===== vLLM関係のラッパクラスをインポート =====
from vllm_inf import VllmBatchInference

# =====================================
#  ステップごとのプロンプトテンプレート (例)
# =====================================
PROMPT_EVOL_WIDTH = """\
あなたは「高度に安全かつ有用なAIプロンプト」を作成する専門家です。
以下の #Given Prompt# を下敷きにしつつ、まったく新しいプロンプト (#Created Prompt#) を構成してください。
ただし、次の条件を厳守してください。

1. 同じジャンル・テーマを保ちつつ、よりユニークで独創的な視点を加える。
2. 長さや難易度は #Given Prompt# と同程度とし、文が破綻しないように注意する。
3. テキスト処理のみ可能なAIが回答できる範囲の内容に留める（外部APIや外部情報への依存は排除する）。
4. 公序良俗に反する指示や差別、誹謗中傷を含む指示にならないようにする。

#Given Prompt#:
{original_prompt}

#Created Prompt#:(作成した新しいプロンプトのみを回答してください)
"""

PROMPT_EVOL_JUDGE = """\
以下のプロンプトは「質の高いプロンプト」と言えるでしょうか？
質の高いプロンプトとは、主に次の点を満たすものと定義します。

- 日本語として自然で文意に破綻がない。
- ユーザーが何を求めているかが明確（質問として成立）。
- 公序良俗および利用規約に反しない。
- テキストのみのAIが回答可能な範囲で完結している。

「はい」または「いいえ」のみで答えてください。

プロンプト：{prompt}
あなたの判断:
"""

PROMPT_EVOL_DEPTH = """\
あなたはプロンプトを書き換える専門家で、与えられたプロンプトをより高度なバージョンに昇華する役割を担います。
次の要領で、下記の #Given Prompt# を複雑かつ深みのある指示に書き換えてください。

1. 新たな要件や制約を1つ追加し、それによって回答難易度が上がるようにする。
2. 加筆は10〜20語以内に抑え、冗長性を最小限にする。
3. 表やコードなど非テキスト要素を省略せずに残す。
4. 違法・不当な要求にならないよう注意し、安全性を損なう要素があれば適度に言い換える。

#Given Prompt#:
{prompt}

#Rewritten Prompt#:(書き換え後のプロンプトのみを回答してください)
"""

PROMPT_EVOL_FLATTEN = """\
以下のプロンプトを、理路整然とした質問形式に修正してください。
誤字脱字や文意不明な箇所があれば推測して補い、理性のないテキスト処理AIでも正しく回答できるように配慮してください。
さらに、公序良俗や利用規約に反する内容があれば、適宜削除・変更して安全な質問へ変えてください。

#Given Prompt#:
{prompt}

#Rewritten Prompt#:(改善後の質問のみを回答してください)
"""

PROMPT_RESPONSE = """\
あなたは知的で公平性を重視するAIアシスタントです。
以下の質問に対して、原則として正確かつ簡潔な回答を行ってください。
ただし、次の条件を徹底してください。

1. 質問が曖昧、または不可能な要求の場合は、「不可能」とだけ返答する。
2. 有害・違法な要求を助長する恐れがある場合も「不可能」と返答する。
3. 事実確認が取れない内容については推測で断言せず、「不確か」と明示する。
4. 日本語以外の表記（外来語、専門用語など）は、必要に応じて簡潔な注釈を添える。

質問: {prompt}

回答（または「不可能」）:
"""

def load_config(config_path: str = "model_config.yml") -> Dict:
    """
    model_config.yml を読み込んで辞書型にして返す簡単なヘルパー関数。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_sft_flow_batch(
    vllm_infer: VllmBatchInference,
    base_prompts: List[str],
    evol_depth_steps: int = 1,
    record_offset: int = 0
) -> List[Dict]:
    """
    マルチステップのSFTフローをまとめて実行し、
    有効なデータ(質問と回答のペア)のみ抽出して返す。

    Args:
        vllm_infer (VllmBatchInference) : vLLM推論オブジェクト
        base_prompts (List[str])        : 元となるプロンプトのリスト
        evol_depth_steps (int)          : evol_depth の適用回数
        record_offset (int)             : 生成されるレコードIDの開始オフセット

    Returns:
        List[Dict]: 成功した (質問,回答) のレコードリスト
                    各要素に "id", "input", "output", "conversation" が含まれる
    """
    # 同じ長さのブールリストを用意し、途中で失敗したものはFalseにする
    keep_mask = [True] * len(base_prompts)

    # =====================================================
    # 1. evol_width
    # =====================================================
    width_input = [
        PROMPT_EVOL_WIDTH.format(original_prompt=bp)
        for bp in base_prompts
    ]
    width_output = vllm_infer.generate_batch_custom(width_input)
    evol_prompts = ["" for _ in base_prompts]

    for i, ans in enumerate(width_output):
        ans_s = ans.strip()
        if not ans_s or "<GEN_ERROR>" in ans_s:
            keep_mask[i] = False
        else:
            evol_prompts[i] = ans_s

    # =====================================================
    # 2. evol_judge ("はい"/"いいえ")
    # =====================================================
    judge_input = [
        PROMPT_EVOL_JUDGE.format(prompt=p) if alive else ""
        for p, alive in zip(evol_prompts, keep_mask)
    ]
    judge_output = vllm_infer.generate_batch_custom(judge_input)
    for i, jans in enumerate(judge_output):
        if keep_mask[i]:
            # 「はい」で始まる場合だけ存続
            if not jans.strip().startswith("はい"):
                keep_mask[i] = False

    # =====================================================
    # 3. evol_depth (指定回数)
    # =====================================================
    for step_idx in range(evol_depth_steps):
        depth_input = [
            PROMPT_EVOL_DEPTH.format(prompt=p) if alive else ""
            for p, alive in zip(evol_prompts, keep_mask)
        ]
        depth_output = vllm_infer.generate_batch_custom(depth_input)
        for i, ans in enumerate(depth_output):
            if keep_mask[i]:
                ans_s = ans.strip()
                if not ans_s or "<GEN_ERROR>" in ans_s:
                    keep_mask[i] = False
                else:
                    evol_prompts[i] = ans_s

    # =====================================================
    # 4. evol_flatten
    # =====================================================
    flat_input = [
        PROMPT_EVOL_FLATTEN.format(prompt=p) if alive else ""
        for p, alive in zip(evol_prompts, keep_mask)
    ]
    flat_output = vllm_infer.generate_batch_custom(flat_input)
    for i, ans in enumerate(flat_output):
        if keep_mask[i]:
            ans_s = ans.strip()
            if not ans_s or "<GEN_ERROR>" in ans_s:
                keep_mask[i] = False
            else:
                evol_prompts[i] = ans_s

    # =====================================================
    # 5. evol_judge (2回目)
    # =====================================================
    judge_input2 = [
        PROMPT_EVOL_JUDGE.format(prompt=p) if alive else ""
        for p, alive in zip(evol_prompts, keep_mask)
    ]
    judge_output2 = vllm_infer.generate_batch_custom(judge_input2)
    for i, jans in enumerate(judge_output2):
        if keep_mask[i]:
            if not jans.strip().startswith("はい"):
                keep_mask[i] = False

    # ===== 質問が確定 =====
    final_questions = [
        evol_prompts[i] if keep_mask[i] else ""
        for i in range(len(evol_prompts))
    ]

    # =====================================================
    # 6. 回答生成 (PROMPT_RESPONSE)
    # =====================================================
    resp_input = [
        PROMPT_RESPONSE.format(prompt=q) if alive else ""
        for q, alive in zip(final_questions, keep_mask)
    ]
    resp_output = vllm_infer.generate_batch_custom(resp_input)

    # 回答が「不可能」や極端に短い場合は除外
    final_answers = ["" for _ in range(len(final_questions))]
    for i, ans in enumerate(resp_output):
        if keep_mask[i]:
            ans_s = ans.strip()
            if ("不可能" in ans_s) or (len(ans_s) < 10):
                keep_mask[i] = False
            else:
                final_answers[i] = ans_s

    # =====================================================
    # 失敗を除外しつつレコード化
    # =====================================================
    output_records = []
    for i, alive in enumerate(keep_mask):
        if not alive:
            continue
        q = final_questions[i]
        a = final_answers[i]

        conversation = [
            {"from": "system", "value": "あなたは優秀な日本語AIアシスタントです。ユーザーの質問に対して、正確かつ簡潔な回答を行います。"},
            {"from": "human",  "value": q},
            {"from": "gpt",    "value": a}
        ]
        # ★IDを record_offset + i でユニークに付与★
        rec_id = record_offset + i

        rec = {
            "id": f"record_{rec_id}",
            "input": q,
            "output": a,
            "conversation": conversation
        }
        output_records.append(rec)

    return output_records


def main():
    """
    メインエントリーポイント。
    コマンドライン引数で受け取った start_idx, end_idx に応じて
    一連のマルチステップSFTフローをバッチ単位で実行し、結果をJSONLに出力します。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, required=True, help='ベースプロンプト開始インデックス (inclusive)')
    parser.add_argument('--end_idx',   type=int, required=True, help='ベースプロンプト終了インデックス (exclusive)')
    parser.add_argument('--output_file', type=str, required=True, help='出力先のJSONLファイルパス')
    parser.add_argument('--id_start', type=int, default=0, help='出力データのIDを開始する整数')
    parser.add_argument('--config_path', type=str, default="model_config.yml", help='model_config.ymlのパス')
    args = parser.parse_args()

    # 1. 設定ファイルをロード
    config = load_config(args.config_path)

    # 2. vLLMの推論クラスを初期化
    vllm_infer = VllmBatchInference(
        model_name=config["model_name"],
        quantization=config["quantization"],
        gpu_mem_util=config["gpu_memory_utilization"],
        max_tokens=config["max_tokens"],
        max_model_len=config["max_model_len"],
        trust_remote_code=config["trust_remote_code"],
        dtype=config["dtype"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        tensor_parallel_size=config["tensor_parallel_size"]
    )

    # 3. start_idx ~ end_idx-1 の範囲で、サンプルのベースプロンプト群を用意
    base_prompts_all = [
        f"これはサンプルの下地プロンプト {i} です。"
        for i in range(args.end_idx)
    ]

    # バッチサイズも設定ファイルから読み込む
    batch_size = config["batch_size"]

    # 4. バッチ単位でSFTフローを回し、結果をJSONLに追記
    with open(args.output_file, "w", encoding="utf-8") as f_out:
        for idx in range(args.start_idx, args.end_idx, batch_size):
            batch_end = min(idx + batch_size, args.end_idx)
            batch_prompts = base_prompts_all[idx:batch_end]

            print(f"\n[INFO] バッチ {idx} ~ {batch_end - 1} を処理中...")
            final_data = run_sft_flow_batch(
                vllm_infer,
                batch_prompts,
                evol_depth_steps=1,  # depthを増やしたければ変更可
                record_offset=args.id_start + idx  # IDの開始位置を指定
            )

            for record in final_data:
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"[INFO] => 生成完了: {len(final_data)} 件を追記しました。")

    print(f"[DONE] 全バッチ処理が終了しました。 出力先: {args.output_file}")


if __name__ == "__main__":
    main()

## 実行コマンド
# python main.py --start_idx 0 --end_idx 100 --output_file output.jsonl