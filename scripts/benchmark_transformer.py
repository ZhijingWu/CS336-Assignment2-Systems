import argparse
import timeit
import statistics
import torch.cuda.nvtx as nvtx

import torch

from cs336_basics.model import BasicsTransformerLM

MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
    "10B": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    },
}

VOCAB_SIZE = 10_000


def build_model(
    model_size: str,
    context_length: int,
    device: str,
):
    config = MODEL_CONFIGS[model_size]

    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )

    return model.to(device)

def make_batch(
    batch_size: int,
    context_length: int,
    device: str,
):
    x = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(batch_size, context_length),
        device=device,
    )

    y = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(batch_size, context_length),
        device=device,
    )

    return x,y

def run_step(
    model,
    x,
    y,
    mode,
    optimizer=None,
):
    if mode == "forward":
        with torch.no_grad():
            with nvtx.range("forward"):
                model(x)

    elif mode == "forward_backward":
        model.zero_grad(set_to_none=True)

        with nvtx.range("forward"):
            logits = model(x)

        with nvtx.range("loss"):
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1,VOCAB_SIZE),
                y.reshape(-1)
            )

        with nvtx.range("backward"):
            loss.backward()

    elif mode == "train":
        optimizer.zero_grad(set_to_none=True)

        with nvtx.range("forward"):
            logits = model(x)

        with nvtx.range("loss"):
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1,VOCAB_SIZE),
                y.reshape(-1)
            )

        with nvtx.range("backward"):
            loss.backward()

        with nvtx.range("optimizer"):
            optimizer.step()

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

def synchronize(device: str):
    if device.startswith("cuda"):
        torch.cuda.synchronize()

def benchmark(
    model,
    x,
    y,
    mode,
    optimizer,
    warmup_steps,
    measurement_steps,
    device,
):
    with nvtx.range("warmup"):
        for _ in range(warmup_steps):
            run_step(
                model,
                x,
                y,
                mode,
                optimizer,
            )

            synchronize(device)

    times = []

    with nvtx.range("measurement_steps"):
        for _ in range(measurement_steps):
            synchronize(device)

            start = timeit.default_timer()

            run_step(
                model,
                x,
                y,
                mode,
                optimizer,
            )

            synchronize(device)

            end = timeit.default_timer()

            time = end - start

            times.append(time)

    mean_time = statistics.mean(times)
    std_time = (
        statistics.stdev(times)
        if len(times) > 1
        else 0.0
    )

    return mean_time, std_time
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS.keys(),
        default="small",
    )

    parser.add_argument(
        "--mode",
        choices=[
            "forward",
            "forward_backward",
            "train",
        ],
        default="forward",
    )

    parser.add_argument(
        "--context-length",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--measurement-steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.context_length <= 0:
        raise ValueError("context_length must be positive")

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")

    if args.measurement_steps <= 0:
        raise ValueError("measurement_steps must be positive")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available.")


    model = build_model(
        args.model_size,
        args.context_length,
        args.device,
    )

    if args.mode == "forward":
        model.eval()
    else:
        model.train()

    x, y = make_batch(
        args.batch_size,
        args.context_length,
        args.device,
    )

    optimizer = None

    if args.mode == "train":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
        )

    mean_time, std_time = benchmark(
        model=model,
        x=x,
        y=y,
        mode=args.mode,
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        measurement_steps=args.measurement_steps,
        device=args.device,
    )

    print(f"mean = {mean_time * 1000:.3f} ms")
    print(f"std  = {std_time * 1000:.3f} ms")



if __name__ == "__main__":
    main()
