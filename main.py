import json
import argparse
from typing import List, Dict

import yaml
from tqdm import tqdm

# ===== vLLM関係のラッパクラスをインポート =====
from vllm_inf import VllmBatchInference

# ===== seed_prompts.py からプロンプトをインポート =====
from gen_seed import gen_seed

# =====================================
#  ステップごとのプロンプトテンプレート
# =====================================
PROMPT_EVOL_WIDTH = """\
あなたは「高度に安全かつ有用なAIプロンプト」を生み出す専門家です。
次の #Given Prompt# をベースに、新たな視点を取り入れた全く別の質問プロンプト(#Created Prompt#)を一つ作成してください。
作成する質問プロンプトは優秀な人間が書くようなプロンプトにして下さい。テーマとタスクに言及する必要はありません。
テーマとジャンルを元にして「優秀な人間が質問するならばAIにこう質問するだろう」と思うようなものを質問プロンプトとして作成してください。

【注意事項】
1. テーマやジャンルは同じままにしつつ、独特な視点やアイデアを加えること
2. 長さと難易度は #Given Prompt# と同等に保つこと
3. 回答は外部APIや外部情報を参照せず、テキスト処理だけで完結できる範囲に限定する
4. 公序良俗や利用規約に反しない内容にする
5. 無用な先入観・偏見を避け、公平な表現を心がける

#Given Prompt#:
{original_prompt}

#Created Prompt#:
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

# ---------------------------
# 新たに追加する修正用プロンプト
# ---------------------------
PROMPT_FIXER = """\
あなたはプロンプト修正専門AIです。
以下のプロンプトは質の低い、不適切と判断されたため、修正する必要があります。

修正にあたっては次の点を考慮してください。
1. 公序良俗および利用規約違反の表現を除去し、安全な内容に書き換える。
2. 質問としての体裁が破綻しないよう再構成し、可能な限り元のテーマを維持する。
3. テキストのみで回答可能な範囲に収め、外部APIや不確定情報への依存を排除する。

元の不適切プロンプト:
{failed_prompt}

#Fixed Prompt#:
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
    output_filepath: str,  # <<< 変更点: 出力ファイルパスを引数に追加
    evol_depth_steps: int = 1,
    record_offset: int = 0
) -> int: # <<< 変更点: 戻り値を生成レコード数(int)に変更
    """
    改良版のマルチステップSFTフローをバッチ処理で実行し、
    有効なデータ(質問と回答のペア)を **逐次JSONLファイルに書き出す**。

    Args:
        vllm_infer (VllmBatchInference) : vLLM推論オブジェクト
        base_prompts (List[str])        : 元となるプロンプトのリスト
        output_filepath (str)           : 結果を書き出すJSONLファイルパス
        evol_depth_steps (int)          : evol_depth の適用回数
        record_offset (int)             : 生成されるレコードIDの開始オフセット

    Returns:
        int: 生成されたレコードの総数
    """

    batch_size = vllm_infer.batch_size
    if batch_size is None:
        batch_size = 32

    # バッファ
    discard_buf_1 = []
    pass_buf = []
    discard_buf_2 = []

    # ID用カウンタ と 生成レコード数カウンタ
    current_id = record_offset
    generated_count = 0 # <<< 変更点: 生成レコード数をカウント

    # ----------------------------------------------------------------
    # 1) Step1(evol_width) → Step2(evol_judge(1)) を
    #    base_prompts からバッチ単位で繰り返す
    # ----------------------------------------------------------------
    total_prompts = len(base_prompts)
    num_full_batches = total_prompts // batch_size

    # tqdmを使って進捗を表示
    pbar = tqdm(total=num_full_batches, desc="Processing Base Prompt Batches")

    for batch_idx in range(num_full_batches):
        start_i = batch_idx * batch_size
        end_i = start_i + batch_size
        sub_prompts = base_prompts[start_i:end_i]

        # ---------- Step1: evol_width ----------
        width_inputs = [
            PROMPT_EVOL_WIDTH.format(original_prompt=p)
            for p in sub_prompts
        ]
        width_outputs = vllm_infer.generate_batch_custom(width_inputs)

        # ---------- Step2: evol_judge(1) ----------
        candidates = []
        judge_inputs = []
        for wout in width_outputs:
            if (not wout.strip()) or ("<GEN_ERROR>" in wout):
                discard_buf_1.append("")
            else:
                judge_inputs.append(wout.strip())

        if len(judge_inputs) > 0:
            judge_outputs = vllm_infer.generate_batch_custom(
                [PROMPT_EVOL_JUDGE.format(prompt=x) for x in judge_inputs]
            )

            for w_prompt, j_out in zip(judge_inputs, judge_outputs):
                if j_out.strip().startswith("はい"):
                    candidates.append(w_prompt)
                else:
                    discard_buf_1.append(w_prompt)

        pass_buf.extend(candidates)

        # ------------------------------------------------------
        # Discard #1 がバッチサイズ分溜まったら修正フロー実行
        # ------------------------------------------------------
        if len(discard_buf_1) >= batch_size:
            fix_targets = discard_buf_1[:batch_size]
            discard_buf_1 = discard_buf_1[batch_size:]

            fixer_inputs = [
                PROMPT_FIXER.format(failed_prompt=ft)
                for ft in fix_targets
            ]
            fix_outputs = vllm_infer.generate_batch_custom(fixer_inputs)

            rejudge_inputs = []
            fix_mapping = []
            for i, f_out in enumerate(fix_outputs):
                if (not f_out.strip()) or ("<GEN_ERROR>" in f_out):
                    continue
                rejudge_inputs.append(f_out.strip())
                fix_mapping.append((i, f_out.strip()))

            if len(rejudge_inputs) > 0:
                rejudge_outputs = vllm_infer.generate_batch_custom(
                    [PROMPT_EVOL_JUDGE.format(prompt=x) for x in rejudge_inputs]
                )

                for (map_idx, prompt_val), j_out in zip(fix_mapping, rejudge_outputs):
                    if j_out.strip().startswith("はい"):
                        pass_buf.append(prompt_val)
                    else:
                        pass

        # ----------------------------------------------------------------
        # pass_buf がバッチサイズに達したら Step3~6 を実施
        # ----------------------------------------------------------------
        while len(pass_buf) >= batch_size:
            target_prompts = pass_buf[:batch_size]
            pass_buf = pass_buf[batch_size:]

            # Step3: evol_depth (N回)
            current_list = target_prompts
            for _ in range(evol_depth_steps):
                depth_inputs = [PROMPT_EVOL_DEPTH.format(prompt=p) for p in current_list]
                depth_outputs = vllm_infer.generate_batch_custom(depth_inputs)
                next_list = []
                for d_out in depth_outputs:
                    d_str = d_out.strip()
                    if d_str and "<GEN_ERROR>" not in d_str:
                        next_list.append(d_str)
                current_list = next_list
                if len(current_list) == 0:
                    break

            # Step4: evol_flatten
            if len(current_list) > 0:
                flat_inputs = [PROMPT_EVOL_FLATTEN.format(prompt=p) for p in current_list]
                flat_outputs = vllm_infer.generate_batch_custom(flat_inputs)
                next_list = []
                for f_out in flat_outputs:
                    f_str = f_out.strip()
                    if f_str and "<GEN_ERROR>" not in f_str:
                        next_list.append(f_str)
                current_list = next_list

            # Step5: evol_judge(2)
            if len(current_list) > 0:
                judge2_inputs = [PROMPT_EVOL_JUDGE.format(prompt=p) for p in current_list]
                judge2_outputs = vllm_infer.generate_batch_custom(judge2_inputs)
                new_candidates = []
                for p_val, j2_out in zip(current_list, judge2_outputs):
                    if j2_out.strip().startswith("はい"):
                        new_candidates.append(p_val)
                    else:
                        discard_buf_2.append(p_val)
                current_list = new_candidates

            # Step6: response gen (PROMPT_RESPONSE)
            if len(current_list) > 0:
                resp_inputs = [PROMPT_RESPONSE.format(prompt=x) for x in current_list]
                resp_outputs = vllm_infer.generate_batch_custom(resp_inputs)

                # <<< 変更点: ここでファイルに追記する >>>
                with open(output_filepath, "a", encoding="utf-8") as f_out:
                    for q_val, ans_val in zip(current_list, resp_outputs):
                        ans_str = ans_val.strip()
                        if ("不可能" in ans_str) or (len(ans_str) < 10):
                            discard_buf_2.append(q_val)
                        else:
                            # OK => レコード化してファイルに書き出す
                            conversation = [
                                {"from": "system", "value": "あなたは優秀な日本語AIアシスタントです。ユーザーの質問に対して、正確かつ簡潔な回答を行います。"},
                                {"from": "human", "value": q_val},
                                {"from": "gpt", "value": ans_str}
                            ]
                            rec = {
                                "id": f"record_{current_id}",
                                "input": q_val,
                                "output": ans_str,
                                "conversation": conversation
                            }
                            # ファイルにJSONL形式で追記
                            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            current_id += 1
                            generated_count += 1 # 生成数をカウントアップ

        # tqdmの進捗を更新
        pbar.update(1)
        pbar.set_postfix({"Generated": generated_count}) # 生成数を表示

    pbar.close()
    # 余りの base_prompts は処理せず終了
    # discard_buf_1 や pass_buf, discard_buf_2 の中身もバッチサイズ未満なら処理せず終了

    return generated_count # <<< 変更点: 生成レコード数を返す


def main():
    """
    メインエントリーポイント。
    コマンドライン引数で受け取った start_idx, end_idx に応じて
    一連のマルチステップSFTフローをバッチ単位で実行し、結果を **逐次JSONLに出力** します。
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
        batch_size=config["batch_size"],
        tensor_parallel_size=config["tensor_parallel_size"]
    )
    vllm_infer.batch_size = config["batch_size"]

    # 3. ベースプロンプトを生成
    #   大量に生成される可能性があるため、必要な範囲だけ読み込むように変更
    #   (gen_seed の実装が不明なため、ここでは一旦全量読み込む想定のままにする)
    print(f"Generating base prompts up to index {args.end_idx}...")
    base_prompts_all = gen_seed(int(args.end_idx))
    print(f"Total base prompts generated: {len(base_prompts_all)}")

    # 対象範囲のプロンプトを取得
    prompts_to_process = base_prompts_all[args.start_idx:args.end_idx]
    if not prompts_to_process:
        print("No prompts to process in the specified range.")
        return

    print(f"Processing prompts from index {args.start_idx} to {args.end_idx-1}")

    # 4. バッチ単位でSFTフローを回し、結果をJSONLに **逐次** 出力
    generated_count = run_sft_flow_batch( # <<< 変更点: 戻り値を受け取る
        vllm_infer,
        prompts_to_process,
        output_filepath=args.output_file, # <<< 変更点: ファイルパスを渡す
        evol_depth_steps=1,
        record_offset=args.id_start
    )

    # 5. 成果をファイルに書き出し <<< 変更点: このブロックは不要になったため削除 >>>
    # with open(args.output_file, "w", encoding="utf-8") as f_out:
    #     for record in final_data:
    #         f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[DONE] 全バッチ処理が終了しました。出力先: {args.output_file}")
    print(f" - 生成レコード数: {generated_count}") # <<< 変更点: 戻り値を使って表示


if __name__ == "__main__":
    main()