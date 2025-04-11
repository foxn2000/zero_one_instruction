import math
import subprocess
import argparse
import os
import torch
import sys # エラー終了用にインポート

def powers_of_two_split(n: int):
    """GPU枚数nを 2の冪 の和に分割する (例:7→[4,2,1], 6→[4,2], 5→[4,1])"""
    splits = []
    remain = n
    while remain > 0:
        p = 2 ** int(math.floor(math.log2(remain)))
        splits.append(p)
        remain -= p
    return splits

def main():
    parser = argparse.ArgumentParser(description="vLLM推論を複数のGPUで並列実行するスクリプト")
    parser.add_argument("--num_prompts", type=int, default=10000,
                        help="ベースプロンプトの総数")
    parser.add_argument("--output_file", type=str, default="sft_output.jsonl",
                        help="最終的に結合したファイルのパス")
    parser.add_argument("--main_script", type=str, default="main.py",
                        help="実際にvLLM推論を行うスクリプトのパス")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="使用するGPUの数。指定しない場合は利用可能な全てのGPUを使用。")
    args = parser.parse_args()

    # 利用可能なGPU数を取得
    num_gpus_available = torch.cuda.device_count()

    if args.num_gpus is not None:
        # --num_gpus が指定された場合
        if args.num_gpus <= 0:
            print(f"[ERROR] --num_gpus には正の整数を指定してください: {args.num_gpus}", file=sys.stderr)
            sys.exit(1)
        if args.num_gpus > num_gpus_available:
            print(f"[ERROR] 指定されたGPU数 ({args.num_gpus}) が利用可能なGPU数 ({num_gpus_available}) を超えています。", file=sys.stderr)
            sys.exit(1)
        num_gpus = args.num_gpus
        print(f"[INFO] コマンドライン引数により、使用するGPU数を {num_gpus} に設定しました。")
    else:
        # --num_gpus が指定されなかった場合
        if num_gpus_available == 0:
            print("[ERROR] 利用可能なGPUが1枚も認識されていません。", file=sys.stderr)
            sys.exit(1)
        num_gpus = num_gpus_available
        print(f"[INFO] 検出した利用可能なGPU数: {num_gpus}")

    # 使用するGPUのインデックスリスト (0から始まる)
    # 例: num_gpus=4 なら visible_gpu_indices = [0, 1, 2, 3]
    # このリストは CUDA_VISIBLE_DEVICES を設定する際に使用する物理GPUのインデックスを示す
    visible_gpu_indices = list(range(num_gpus))

    # 2の冪かチェック
    is_power_of_two = (num_gpus & (num_gpus - 1) == 0)

    if is_power_of_two:
        print(f"[INFO] 使用するGPU数 ({num_gpus}) は2の冪です。単一プロセスで指定されたGPUを利用します。")

        # 使用するGPUをCUDA_VISIBLE_DEVICESで指定
        cuda_visible = ",".join(str(gid) for gid in visible_gpu_indices)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible

        cmd = [
            "python",
            args.main_script,
            "--start_idx", "0",
            "--end_idx", str(args.num_prompts),
            "--output_file", args.output_file
            # 必要に応じて main.py 側にも使用するGPU数を渡す引数を追加してもよい
            # "--tensor_parallel_size", str(num_gpus) # 例: vLLMの場合
        ]
        print(f"[CMD] CUDA_VISIBLE_DEVICES={cuda_visible} {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] サブプロセス実行中にエラーが発生しました: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        print(f"[INFO] 使用するGPU数 ({num_gpus}) は2の冪ではありません。2の冪の和に分割して並列実行します。")

        # たとえば num_gpus=7 → [4,2,1]
        group_sizes = powers_of_two_split(num_gpus)
        print(f"  => GPUグループ構成: {group_sizes}")

        # ベースプロンプトも group_sizes に応じて分割
        total_prompts = args.num_prompts
        start_idx = 0

        # 一時ファイルを複数作って後で結合
        temp_files = []
        current_gpu_offset = 0 # visible_gpu_indices 内でのオフセット

        processes = []
        for i, grp_size in enumerate(group_sizes):
            # このグループが処理するプロンプト数
            if i < (len(group_sizes) - 1):
                allocated = round(total_prompts * (grp_size / num_gpus))
            else:
                allocated = total_prompts - start_idx

            end_idx = start_idx + allocated
            if end_idx > args.num_prompts:
                end_idx = args.num_prompts

            # このグループが使用する物理GPUのIDリスト
            # visible_gpu_indices からスライスして取得
            gpu_ids_for_group = visible_gpu_indices[current_gpu_offset : current_gpu_offset + grp_size]
            current_gpu_offset += grp_size

            # 環境変数 CUDA_VISIBLE_DEVICES で使用GPUを制限
            cuda_visible = ",".join(str(gid) for gid in gpu_ids_for_group)

            out_file = f"gen_temp_output_{i}.jsonl"
            temp_files.append(out_file)

            cmd = [
                "python",
                args.main_script,
                "--start_idx", str(start_idx),
                "--end_idx",   str(end_idx),
                "--output_file", out_file
                # 必要に応じて main.py 側にも使用するGPU数を渡す引数を追加してもよい
                # "--tensor_parallel_size", str(grp_size) # 例: vLLMの場合
            ]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = cuda_visible

            print(f"\n[LAUNCH] Group {i}: GPUs={gpu_ids_for_group}, Prompts=[{start_idx},{end_idx}), Output={out_file}")
            print(f"  Command: CUDA_VISIBLE_DEVICES={cuda_visible} {' '.join(cmd)}")

            # 非同期でサブプロセスを起動
            try:
                p = subprocess.Popen(cmd, env=env)
                processes.append(p)
            except Exception as e:
                 print(f"[ERROR] サブプロセス '{' '.join(cmd)}' の起動に失敗しました: {e}", file=sys.stderr)
                 # 必要であれば他の起動済みプロセスを停止する処理を追加
                 sys.exit(1)

            start_idx = end_idx
            if start_idx >= args.num_prompts:
                break # 全てのプロンプトを割り当てたら終了

        # すべてのサブプロセス終了を待機
        exit_codes = []
        for p in processes:
            exit_codes.append(p.wait())

        # エラーが発生したサブプロセスがないか確認
        if any(code != 0 for code in exit_codes):
            print("[ERROR] 一部のサブプロセスがエラーで終了しました。", file=sys.stderr)
            # エラーの詳細を表示することも可能
            for i, code in enumerate(exit_codes):
                if code != 0:
                    print(f"  Group {i} process exited with code {code}", file=sys.stderr)
            # 一時ファイルは結合せずに終了するなどの判断が可能
            sys.exit(1)

        # temp_files を結合して最終的な output_file を作成
        print(f"\n[INFO] すべてのジョブが正常に終了しました。一時ファイルを結合します...")
        try:
            with open(args.output_file, "w", encoding="utf-8") as f_out:
                for tempf in temp_files:
                    if not os.path.exists(tempf):
                        print(f"[WARN] 一時ファイルが見つかりません: {tempf}", file=sys.stderr)
                        continue
                    try:
                        with open(tempf, "r", encoding="utf-8") as f_in:
                            for line in f_in:
                                f_out.write(line)
                    except Exception as e:
                        print(f"[ERROR] 一時ファイル '{tempf}' の読み込み中にエラー: {e}", file=sys.stderr)
                        # 処理を続行するか、ここで中断するかを決定
            print(f"[INFO] 結果を {args.output_file} に統合しました。")
        except Exception as e:
             print(f"[ERROR] 最終ファイル '{args.output_file}' の書き込み中にエラー: {e}", file=sys.stderr)
             sys.exit(1)

        # 必要なら一時ファイルを削除
        print("[INFO] 一時ファイルを削除します...")
        for tf in temp_files:
           try:
               if os.path.exists(tf):
                   os.remove(tf)
           except Exception as e:
               print(f"[WARN] 一時ファイル '{tf}' の削除に失敗しました: {e}", file=sys.stderr)
        print("[INFO] 完了しました。")


if __name__ == "__main__":
    main()

"""
python auto_run.py --num_gpus 7 --main_script main.py --num_prompts 100 --output_file my_output.jsonl
"""