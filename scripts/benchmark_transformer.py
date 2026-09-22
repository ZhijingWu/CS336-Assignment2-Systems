import timeit
import statistics

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


# From Section 2.1.2
VOCAB_SIZE = 10_000
BATCH_SIZE = 4
CONTEXT_LENGTH = 512

# From benchmarking problem (b)
WARMUP_STEPS = 5
MEASUREMENT_STEPS = 10

# GPU benchmarking
DEVICE = "cuda"

def build_model(model_size: str):
    config = MODEL_CONFIGS[model_size]

    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=CONTEXT_LENGTH,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )

    return model.to(DEVICE)

def make_batch():
    x = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(BATCH_SIZE, CONTEXT_LENGTH),
        device=DEVICE,
    )

    y = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(BATCH_SIZE, CONTEXT_LENGTH),
        device=DEVICE,
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
            model(x)

    elif mode == "forward_backward":
        model.zero_grad(set_to_none=True)

        logits = model(x)

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1,VOCAB_SIZE),
            y.reshape(-1)
        )

        loss.backward()

    elif mode == "train":
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(x)
        
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1,VOCAB_SIZE),
            y.reshape(-1)
        )
        
        loss.backward()

        optimizer.step()

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )


model = build_model("small")
x, y = make_batch()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
)

mode = "forward"

for _ in range(WARMUP_STEPS):
    run_step(
        model,
        x,
        y,
        mode,
        optimizer,
    )

times = []

for _ in range(MEASUREMENT_STEPS):
    start = timeit.default_timer()

    run_step(
        model,
        x,
        y,
        mode,
        optimizer,
    )

    torch.cuda.synchronize()

    end = timeit.default_timer()

    time = end - start

    times.append(time)

mean_time = statistics.mean(times)
std_time = statistics.stdev(times)

print(f"mean = {mean_time:.6f} s")
print(f"std  = {std_time:.6f} s")


