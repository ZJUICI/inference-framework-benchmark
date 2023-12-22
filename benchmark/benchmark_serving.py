"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    (vLLM backend)
    python -m vllm.entrypoints.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_hf_server.sh <your_model>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --tokenizer <your_model> --dataset <target_dataset> \
        --request-rate <request_rate>
"""
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
import requests
from transformers import PreTrainedTokenizerBase
from benchmark.transformers_utils.tokenizer import get_tokenizer


def tokenized_datasets(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids

    return [
        (prompts[i], len(prompt_token_ids[i]), len(completion_token_ids[i]))
        for i in range(len(prompts))
    ]


def sample_requests(
    input_length: int,
    output_length: int,
    num_requests: int,
    dataset: List[Tuple[str, int, int]],
) -> List[Tuple[str, int, int]]:
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_len, completion_len in dataset:
        if prompt_len < 4 or completion_len < 4:
            continue
        if (
            prompt_len > 2048
            or prompt_len not in range(input_length - 50, input_length + 50)
            or prompt_len + output_length > 4096
        ):
            continue
        filtered_dataset.append((prompt, prompt_len, output_length))
    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    backend: str,
    api_url: str,
    prompt: str,
    output_len: int,
    best_of: int,
    use_beam_search: bool,
) -> None:
    headers = {"User-Agent": "Benchmark Client"}
    if backend == "vllm":
        pload = {
            "prompt": prompt,
            "n": 1,
            "best_of": best_of,
            "use_beam_search": use_beam_search,
            "temperature": 0.0 if use_beam_search else 1.0,
            "top_p": 1.0,
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
        }
    elif backend == "tgi":
        assert not use_beam_search
        params = {
            "best_of": best_of,
            "max_new_tokens": output_len,
            "do_sample": True,
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break


def send_request_one_by_one(
    url: str,
    backend: str,
    best_of: int,
    use_beam_search: bool,
    input_prompt_len: int,
    max_new_tokens: int,
    prompt_requests: List[Tuple[str, int, int]],
):
    request_latency: List[float] = []
    for prompt, _, _ in prompt_requests:
        request_start_time = time.perf_counter()
        headers = {"User-Agent": "Benchmark Client"}
        if backend == "vllm":
            pload = {
                "prompt": prompt,
                "n": 1,
                "best_of": best_of,
                "use_beam_search": use_beam_search,
                "temperature": 0.0 if use_beam_search else 1.0,
                "top_p": 1.0,
                "max_tokens": max_new_tokens,
                "ignore_eos": True,
                "stream": False,
            }
        elif backend == "tgi":
            assert not use_beam_search
            params = {
                "best_of": best_of,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
            }
            pload = {
                "inputs": prompt,
                "parameters": params,
            }
        else:
            raise ValueError(f"Unknown backend: {backend}")
        while True:
            response = requests.post(url, json=pload, headers=headers)
            if response.status_code == 200 and "error" not in response.json():
                break
        request_end_time = time.perf_counter()
        elapsed = request_end_time - request_start_time
        request_latency.append(elapsed)

    # Compute the latency statistics.
    _avg_per_token_latency = np.mean(
        [latency / (input_prompt_len + max_new_tokens) for latency in request_latency]
    )
    _avg_per_output_token_latency = np.mean(
        [latency / max_new_tokens for latency in request_latency]
    )

    return _avg_per_token_latency, _avg_per_output_token_latency


async def benchmark(
    backend: str,
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(
            send_request(
                backend,
                api_url,
                prompt,
                output_len,
                best_of,
                use_beam_search,
            )
        )
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    global_datasets = tokenized_datasets(args.datasets, tokenizer)
    print("Tokenized datasets finished")

    metrics = [128, 512, 1024, 2048]
    benchmark_metrics = [
        (input_len, output_len) for input_len in metrics for output_len in metrics
    ]

    print(
        "| Input Length | Output Length | Throughput (requests/s) | Throughput (output token/s) "
        "| Average latency per token (ms) | Average latency per output token (ms) |"
    )
    print(
        "|:------------:|:-------------:|:-----------------------:|:---------------------------:"
        "|:------------------------------:|:-------------------------------------:|"
    )

    for input_len, output_len in benchmark_metrics:
        # 20% of num_prompts are used to test latency,
        # while 80% are used to test throughput.
        avg_latency_input_requests = sample_requests(
            input_length=input_len,
            output_length=output_len,
            num_requests=args.num_prompts * 0.2,
            dataset=global_datasets,
        )

        # Process requests one by one to prevent queue congestion.
        #   avg_per_token_latency: Average latency per token (ms)
        #   avg_per_output_token_latency: Average latency per output token (ms)
        avg_per_token_latency, avg_per_output_token_latency = send_request_one_by_one(
            url=api_url,
            backend=args.backend,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            input_prompt_len=input_len,
            max_new_tokens=output_len,
            prompt_requests=avg_latency_input_requests,
        )

        # Filter out 80% of the requests for testing throughput.
        throughput_input_requests = sample_requests(
            input_length=input_len,
            output_length=output_len,
            num_requests=args.num_prompts * 0.8,
            dataset=global_datasets,
        )

        benchmark_start_time = time.perf_counter()
        asyncio.run(
            benchmark(
                args.backend,
                api_url,
                throughput_input_requests,
                args.best_of,
                args.use_beam_search,
                args.request_rate,
            )
        )
        benchmark_end_time = time.perf_counter()
        benchmark_time = benchmark_end_time - benchmark_start_time

        # throughput_requests: Throughput (requests/s)
        # throughput_output_tokens: Throughput (output token/s)
        throughput_requests = f"{(args.num_prompts * 0.8) / benchmark_time:.2f}"
        throughput_output_tokens = (
            f"{(args.num_prompts * 0.8 * output_len) / benchmark_time:.2f}"
        )

        print(
            f"|{input_len}|{output_len}|{throughput_requests}|{throughput_output_tokens}"
            f"|{avg_per_token_latency*1000:.2f}|{avg_per_output_token_latency*1000:.2f}|"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "tgi"])
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset."
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Name or path of the tokenizer."
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and " "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts", type=int, default=1000, help="Number of prompts to process."
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="trust remote code from huggingface",
    )
    args = parser.parse_args()
    main(args)
