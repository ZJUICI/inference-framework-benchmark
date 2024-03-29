# Mistral-7B-OpenOrca-GPTQ

## A100 40G

## vllm v0.2.5

### Launching Command
```sh
python3 -m vllm.entrypoints.api_server --model <model dir> --port 8080 --disable-log-requests --quantization gptq
```

| Input Length | Output Length | Throughput (requests/s) | Throughput (output token/s) | Average latency per token (ms) | Average latency per output token (ms) |
| :----------: | :-----------: | :---------------------: | :-------------------------: | :----------------------------: | :-----------------------------------: |
|     128      |      128      |          16.01          |           2049.78           |              5.70              |                 11.40                 |
|     128      |      512      |          4.00           |           2050.01           |              8.95              |                 11.19                 |
|     128      |     1024      |          1.70           |           1740.40           |             10.35              |                 11.65                 |
|     128      |     2048      |          0.59           |           1210.26           |             11.22              |                 11.92                 |
|     512      |      128      |          9.10           |           1164.83           |              2.57              |                 12.87                 |
|     512      |      512      |          2.93           |           1497.70           |              6.08              |                 12.16                 |
|     512      |     1024      |          1.31           |           1344.21           |              8.22              |                 12.33                 |
|     512      |     2048      |          0.48           |           978.78            |              9.71              |                 12.14                 |
|     1024     |      128      |          5.40           |           691.56            |              1.42              |                 12.80                 |
|     1024     |      512      |          1.92           |           984.48            |              4.13              |                 12.38                 |
|     1024     |     1024      |          0.99           |           1010.71           |              6.17              |                 12.34                 |
|     1024     |     2048      |          0.40           |           816.18            |              8.38              |                 12.57                 |
|     2048     |      128      |          3.02           |           386.52            |              0.85              |                 14.43                 |
|     2048     |      512      |          1.21           |           617.41            |              2.58              |                 12.92                 |
|     2048     |     1024      |          0.64           |           657.37            |              4.24              |                 12.72                 |
|     2048     |     2048      |          0.28           |           576.70            |              6.42              |                 12.83                 |

## tgi v1.3.4

### Launching Command
```sh
text-generation-launcher --model-id <model name or path> --port 8080  --max-batch-prefill-tokens 4096 --max-input-length 4096 --max-total-tokens 8192 --max-concurrent-requests 1024 --quantize gptq
```

| Input Length | Output Length | Throughput (requests/s) | Throughput (output token/s) | Average latency per token (ms) | Average latency per output token (ms) |
| :----------: | :-----------: | :---------------------: | :-------------------------: | :----------------------------: | :-----------------------------------: |
|     128      |      128      |          3.51           |           356.18            |             40.29              |                 94.79                 |
|     128      |      512      |          2.26           |           599.83            |             50.16              |                 88.07                 |
|     128      |     1024      |          1.54           |           559.23            |             56.93              |                 87.78                 |
|     128      |     2048      |          1.02           |           451.51            |             50.44              |                 88.00                 |
|     512      |      128      |          1.69           |           165.63            |             14.63              |                 90.86                 |
|     512      |      512      |          1.23           |           265.99            |             23.05              |                 93.59                 |
|     512      |     1024      |          1.05           |           279.02            |             22.41              |                 91.86                 |
|     512      |     2048      |          0.92           |           237.95            |             28.35              |                 90.24                 |
|     1024     |      128      |          0.90           |           100.04            |             10.23              |                 92.07                 |
|     1024     |      512      |          0.66           |           211.11            |             14.20              |                 96.27                 |
|     1024     |     1024      |          0.62           |           286.12            |             20.10              |                 93.63                 |
|     1024     |     2048      |          0.41           |           254.69            |             24.84              |                 91.68                 |
|     2048     |      128      |          0.48           |            56.58            |              4.98              |                105.10                 |
|     2048     |      512      |          0.41           |           154.23            |             11.92              |                 94.03                 |
|     2048     |     1024      |          0.37           |           204.42            |             17.48              |                 94.71                 |
|     2048     |     2048      |          0.27           |           208.15            |             13.32              |                 94.26                 |

### Benchmark Command
```sh
# `obo-num` represents the number of one-by-one requests sent
# `num-prompts` without `requests rate`, this will send all requests simultaneously.
python3 -m benchmark.benchmark_serving --backend {tgi|vllm} --port 8080 --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json  --tokenizer <model name or path> --obo-num 10 --input-lengths 128,512,1024,2048 --num-prompts 399 --output ./output/output.md
```


## Throughput (Requests/s)
![image](./throughput-req.png)
## Throughput (Output Token/s)
![image](./throughput-token-out.png)
## Average Latency Per token/(ms)
![image](./pertoken.png)
## Average Latency Per Output Token(ms)
![image](./per-out-token.png)