import math
from typing import List, Optional
import torch
from vllm import LLM, SamplingParams


class VllmBatchInference:
    """
    vLLM によるバッチ推論を行うクラス。
    モデルロードや generate_batch_custom などの推論に必要な機能をまとめています。
    """
    def __init__(self,
                 model_name: str,
                 quantization: Optional[str],
                 gpu_mem_util: float,
                 max_tokens: int,
                 temperature: float,
                 top_p: float,
                 max_model_len: int,
                 trust_remote_code: bool,
                 dtype: str,
                 batch_size: int,
                 tensor_parallel_size: Optional[int]):
        """
        Args:
            model_name (str)           : vLLMで使用するモデルのパスや名称
            quantization (str | None)  : 量子化方式 (例: "awq" など)。不要なら None
            gpu_mem_util (float)       : GPUメモリ使用率
            max_tokens (int)           : 1回の生成ステップで出力する最大トークン数
            temperature (float)        : サンプリング時の温度
            top_p (float)              : nucleus sampling用 top_p
            tensor_parallel_size (int) : テンソル並列のサイズ(GPU枚数などに応じて設定)。Noneなら自動計算
            max_model_len (int)        : モデルの最大長
            trust_remote_code (bool)   : リモートコードを信頼するかどうか
            dtype (str)               : モデルのデータ型 (例: "bfloat16" など)
        """
        self.model_name = model_name
        self.quantization = quantization
        self.gpu_mem_util = gpu_mem_util
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self.batch_size = batch_size

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise RuntimeError("GPUが見つかりません。vLLMの実行にはGPUが必要です。")

        if tensor_parallel_size is None:
            # GPU数から最適(2の冪)のtensor_parallel_sizeを推定
            exponent = int(math.floor(math.log2(num_gpus)))
            self.tensor_parallel_size = 2 ** exponent
        else:
            if tensor_parallel_size > num_gpus:
                print(f"警告: 指定されたtensor_parallel_size ({tensor_parallel_size}) はGPU数({num_gpus})より大きいです。"
                      f" -> {num_gpus} に修正します。")
                self.tensor_parallel_size = num_gpus
            else:
                self.tensor_parallel_size = tensor_parallel_size

        print(f"[vLLM Init] Model={model_name}, TP={self.tensor_parallel_size}, quant={quantization}")

        # vLLMの LLM オブジェクトを初期化し、モデルロードを行う
        self.llm = LLM(
            model=self.model_name,
            quantization=self.quantization,
            gpu_memory_utilization=self.gpu_mem_util,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            max_num_seqs=self.batch_size,
        )

        # 便利のためにトークナイザも保持しておく
        self.tokenizer = self.llm.get_tokenizer()

    def generate_batch_custom(self, prompts: List[str]) -> List[str]:
        """
        与えられた複数のプロンプトをまとめて推論し、出力テキストリストを返す。

        Args:
            prompts (List[str]): 推論したいプロンプト文字列のリスト

        Returns:
            List[str]: モデル出力のリスト（順番は入力順に対応）
        """
        if not prompts:
            return []

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        try:
            outputs = self.llm.generate(prompts, sampling_params)
            return [o.outputs[0].text for o in outputs]
        except Exception as e:
            print(f"[ERROR] generate_batch_custom fail: {e}")
            # エラーが起きた場合は、失敗した分だけ "<GEN_ERROR>" を返す
            return ["<GEN_ERROR>"] * len(prompts)
