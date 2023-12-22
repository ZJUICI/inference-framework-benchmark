# Inference Framework Benchmark

[![en](https://img.shields.io/badge/lang-en-green.svg)](https://github.com/ZJUICI/inference-framework-benchmark/blob/main/README.md)
[![zh-cn](https://img.shields.io/badge/lang-zh--cn-green.svg)](https://github.com/ZJUICI/inference-framework-benchmark/blob/main/README.zh-cn.md)

## 简介

随着大型语言模型（LLM）的流行，越来越多的企业需要私有化部署大语言模型。选择正确的推理框架对于优化性能和资源利用至关重要。本仓库旨在各种 LLM 推理框架进行性能测试，为用户在不同的推理框架间进行选择时提供性能指标。

## 测试条件

### 硬件

由于条件的限制，所有的测试都是在少量的 A100-40G 上完成的。测试结果仅在 A100-40G 上具有参考意义

### 数据集

由于这是一个关于推理时间的性能测试，您可以选择任何数据集。我们选择了 `ShareGPT_Vicuna_unfiltered` 数据集。要获取该数据集，只需执行以下命令：

```shell
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### 推理框架

现有的测试是使用 vllm（0.2.5）和 text-generation-inference（1.3.3）进行的。要启动推理服务，请按照以下步骤进行（请记得根据您的具体模型自定义参数）：

- VLLM

    ```shell
    python -m vllm.entrypoints.api_server --model=$MODEL_ID_OR_PATH
    ```

- TGI

    ```shell
    text-generation-launcher --model-id $MODEL_ID_OR_PATH \
    --max-batch-prefill-tokens 3580 \
    --max-input-length 3580 \
    --max-total-tokens 4096 \
    --max-batch-total-tokens 8192 \
    --port 8000
    ```
