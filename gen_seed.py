# gen_seed.py

from datasets import load_dataset

def gen_seed(num: int):
    """
    Hugging Faceの「TeamDelta/tasks」「TeamDelta/domain」から
    タスクと話題を掛け合わせたシードプロンプトを生成する。
    常に引数 num で指定された数のプロンプトを返す。

    Args:
        num (int): 生成するプロンプト数

    Returns:
        List[str]: 指定数のシードプロンプト
    """

    # 1. データセット読み込み (train splitを仮定)
    tasks_ds = load_dataset("TeamDelta/tasks")   # キー "task_name"
    domain_ds = load_dataset("TeamDelta/domain") # キー "domain_name"
    tasks_list = tasks_ds["train"]["task_name"]
    domain_list = domain_ds["train"]["domain_name"]

    # 2. タスク×話題の組み合わせを準備
    combined_prompts = []
    for t in tasks_list:
        for d in domain_list:
            # ここでシードプロンプトの形を少し工夫してみる
            prompt = (
                f"【テーマ】{d}\n"
                f"【タスク】{t}\n"
                "あなたはこのテーマとタスクの専門家です。これに関して何を調査・検討すべきか、"
                "あるいはどのように知識を整理すればよいのかを教えてください。"
            )
            combined_prompts.append(prompt)

    # 3. 必要数が足りなければ繰り返し追加
    #    (たとえば掛け合わせの総数が20しかないのに、num=50 が要求された場合など)
    final_prompts = []
    idx = 0
    while len(final_prompts) < num:
        final_prompts.append(combined_prompts[idx % len(combined_prompts)])
        idx += 1

    # 4. ちょうど要求数だけ切り出して返す
    return final_prompts[:num]
