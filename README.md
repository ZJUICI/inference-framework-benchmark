# Inference Framework Benchmark

[![en](https://img.shields.io/badge/lang-en-green.svg)](https://github.com/ZJUICI/inference-framework-benchmark/blob/main/README.md)
[![zh-cn](https://img.shields.io/badge/lang-zh--cn-green.svg)](https://github.com/ZJUICI/inference-framework-benchmark/blob/main/README.zh-cn.md)

## Introduction

With the popularity of large language models (LLMs), an increasing number of enterprises are seeking private deployments
of these models. Choosing the right inference framework is crucial for optimizing performance and resource utilization.
This repository is designed to conduct performance benchmark on various LLM inference frameworks, providing users with
performance metrics to aid in the selection of the most suitable framework for their needs.

## Setup

### Hardware

Due to constraints, all tests were conducted on a limited number of A100-40G units. The test results are only indicative for the A100-40G and may not be representative across other configurations.

### Dataset

You are free to choose any dataset since this is a time performance test. We have opted for the `ShareGPT_Vicuna_unfiltered` dataset. To obtain the dataset, simply execute the following command:

```shell
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### Frameworks

The existing benchmark is conducted using vllm (0.2.5) and text-generation-inference (1.3.3). To initiate the inference services, follow the steps below (please remember to customize the parameters for your specific model):

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
