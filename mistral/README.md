# Mistral 7B Benchmark

## vllm server

```sh
python3 -m vllm.entrypoints.api_server --model <model dir> --port 8080 --disable-log-requests  # if quantized model --quantization <quantize method>
```

## Text-generation-inference

```sh
text-generation-launcher --model-id <model name or path> --port 8080  --max-batch-prefill-tokens 4096 --max-input-length 4096 --max-total-tokens 8192 --max-concurrent-requests 1024 # if quantized model --quantize <quantize method>

```

## benchmark

```sh
# `obo-num` represents the number of one-by-one requests sent
# `num-prompts` without `requests rate`, this will send all requests simultaneously.
python3 -m benchmark.benchmark_serving --backend {tgi|vllm} --port 8080 --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json  --tokenizer <model name or path> --obo-num 10 --input-lengths 128,512,1024,2048 --num-prompts 399 --output ./output/output.md
```
