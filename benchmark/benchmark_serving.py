import argparse
import asyncio
import json
import random
import time
import warnings
import pathlib
from typing import AsyncGenerator, List, Tuple

import aiohttp
import numpy as np
import requests
from transformers import PreTrainedTokenizerBase
from benchmark.transformers_utils.tokenizer import get_tokenizer

from loguru import logger



GENERATED_TOKENS:List[int] = []

class TableLog:
    """markdown table log"""

    def __init__(self, print=False) -> None:
        self._text = ""
        self._prt = print

    @property
    def text(self) -> str:
        return self._text

    def write(self, s: str):
        self._text += s + "\n"
        if self._prt:
            logger.info(s)

    def save(self, p: str, mkdir=True):
        pt = pathlib.Path(p).resolve().absolute()
        if mkdir:
            pt.parent.mkdir(parents=True, exist_ok=True)

        pt.write_text(self._text)

        logger.info(f"Markdown result saved. path:{pt}")


def tokenized_datasets(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> List[List[int]]:
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
    logger.info(f"Begin encoding the prompt string.")
    encoding_s = time.perf_counter()
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids

    # Unnecessary encode.
    # completions = [completion for _, completion in dataset]
    # completion_token_ids = tokenizer(completions).input_ids
    logger.info(
        f"Encoding prompts string to token IDs completed. Time spent: {time.perf_counter() - encoding_s:.4f}s"
    )
    return prompt_token_ids


def sample_requests(
    input_length: int,
    output_length: int,
    num_requests: int,
    # dataset: List[Tuple[str, int, int]],
    dataset: List[List[int]],
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    filtered_dataset: List[Tuple[str, int, int]] = []

    logger.info(f"filter requests by input length:{input_length}")
    s = time.perf_counter()
    for prompt_token_id in dataset:
        if len(prompt_token_id) < input_length:
            continue
        elif len(prompt_token_id) in range(
            input_length, input_length + input_length // 2
        ):
            # NOTE: split prompt ids by input length
            filtered_dataset.append(
                (
                    tokenizer.decode(
                        prompt_token_id[:input_length], skip_special_tokens=True
                    ),
                    input_length,
                    output_length,
                )
            )
    logger.info(f"Filtered requests completed, time spent: {time.perf_counter() - s:.4f}s")
    # Sample the requests.
    if len(filtered_dataset) < num_requests:
        warnings.warn(
            f"The number of requests {num_requests} is larger than the dataset length: {len(filtered_dataset)}, so it has been automatically set to {len(filtered_dataset)}"
        )
        return filtered_dataset
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
            "details":True
        }
        pload = {
            "inputs": prompt,
            "parameters": params,
        }
    elif backend == "trt":
        pload = {
            "text_input":prompt,
            "max_tokens":output_len,
            "bad_words": "", 
            # NOTE: Empty `stop_words` and `end_id` can make server generated tokens extend to the `max_tokens` limit.
            "stop_words": "",
            "temperature": 0.01
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
                global GENERATED_TOKENS 
                generated_tokens = output_len
                if backend == "tgi":
                    generated_tokens = output["details"]["generated_tokens"]
                
                GENERATED_TOKENS.append(generated_tokens)
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

    request_latency: List[Tuple(float,int)] = []

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
            # NOTE(zt): For TGI, the HTTP router doesn't support the `ignore_eos`` parameter. 
            # However, TGI can use `details:True` to get the token count in the response. 
            # So, we might think about using this count to calculate token latency.
            params = {
                "best_of": best_of,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "details":True
            }
            pload = {
                "inputs": prompt,
                "parameters": params,
            }
        elif backend == "trt":
            pload = {
                "text_input":prompt,
                "max_tokens":max_new_tokens,
                "bad_words": "", 
                # NOTE: Empty `stop_words` and `end_id` can make server generated tokens extend to the `max_tokens` limit.
                "stop_words": "",
                "temperature": 0.01
            }
        else:
            raise ValueError(f"Unknown backend: {backend}")
        while True:
            response = requests.post(url, json=pload, headers=headers)
            if response.status_code == 200 and "error" not in response.json():
                generated_tokens = max_new_tokens
                if backend == "tgi":
                    generated_tokens = response.json()["details"]["generated_tokens"]
                
                break

        request_end_time = time.perf_counter()
        elapsed = request_end_time - request_start_time
        # request_latency.append(elapsed)
        request_latency.append((elapsed,generated_tokens))

    # Compute the latency statistics.
    _avg_per_token_latency = np.mean(
        [latency / (input_prompt_len + generated_token) for latency, generated_token in request_latency]
    )
    _avg_per_output_token_latency = np.mean(
        [latency / generated_token for latency,generated_token in request_latency]
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
        prompt, _, output_len = request
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
    tl = TableLog()
    logger.info(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"

    if args.backend == "trt":
        # TODO: get model name by args named trt-model?
        api_url = f"http://{args.host}:{args.port}/v2/models/ensemble/generate"
    
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    global_datasets = tokenized_datasets(args.dataset, tokenizer)
    logger.info("Tokenized datasets finished")

    metrics_extract = lambda x: [int(i) for i in x.split(",")]

    metrics_in = metrics_extract(args.input_lengths)
    logger.info(f"input lengths {metrics_in} ")

    if not args.output_lengths:
        logger.info(f"Output lengths are not passed, set to be the same as input lengths.")
        metrics_out = metrics_in
    else:
        metrics_out = metrics_extract(args.output_lengths)
    logger.info(f"output lengths {metrics_out} ")
    benchmark_metrics = [
        (input_len, output_len)
        for input_len in metrics_in
        for output_len in metrics_out
    ]

    tl.write(
        "| Input Length | Output Length | Throughput (requests/s) | Throughput (output token/s) "
        "| Average latency per token (ms) | Average latency per output token (ms) |"
    )
    tl.write(
        "|:------------:|:-------------:|:-----------------------:|:---------------------------:"
        "|:------------------------------:|:-------------------------------------:|"
    )

    for input_len, output_len in benchmark_metrics:
        # 20% of num_prompts are used to test latency,
        # while 80% are used to test throughput.
        logger.info(
            f"Start making requests one by one, with input length: {input_len} and output length: {output_len}."
        )
        avg_latency_input_requests = sample_requests(
            input_length=input_len,
            output_length=output_len,
            # num_requests=int(args.num_prompts * 0.2),
            num_requests=int(args.obo_num),
            dataset=global_datasets,
            tokenizer=tokenizer,
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
        logger.info(
            f"Begin benchmarks for input length:{input_len} and output length:{output_len}."
        )
        # Filter out 80% of the requests for testing throughput.
        throughput_input_requests = sample_requests(
            input_length=input_len,
            output_length=output_len,
            num_requests=int(args.num_prompts),
            dataset=global_datasets,
            tokenizer=tokenizer,
        )
        global GENERATED_TOKENS
        GENERATED_TOKENS = []
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
        throughput_requests = f"{(args.num_prompts) / benchmark_time:.2f}"

        #
        # throughput_output_tokens = (
        #     f"{(args.num_prompts * output_len) / benchmark_time:.2f}"
        # )   
        
        # all output length / all time
        throughput_output_tokens = (
            f"{sum(GENERATED_TOKENS) / benchmark_time:.2f}"
        )

        # 
        tl.write(
            f"|{input_len}|{output_len}|{throughput_requests}|{throughput_output_tokens}"
            f"|{avg_per_token_latency*1000:.2f}|{avg_per_output_token_latency*1000:.2f}|"
        )

        logger.info(f"\n{tl.text}")

    if args.output:
        tl.save(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput."
    )
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "tgi","trt"])
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

    parser.add_argument(
        "--obo-num",
        type=int,
        help="Number of send one by one request",
    )

    parser.add_argument(
        "--input-lengths",
        type=str,
        default="128,512,1024,2048",
        help="The input length of prompts, separated by ','",
    )

    parser.add_argument(
        "--output-lengths",
        type=str,
        help="The output length of max new tokens, separated by ','",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output path for saving the metrics table as markdown.",
    )
    args = parser.parse_args()
    main(args)
