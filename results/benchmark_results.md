# Assignment 2 Benchmark Results

Environment:
- GPU: NVIDIA GeForce RTX 5090 D, 32 GB
- PyTorch: 2.11.0+cu130
- CUDA: 13.0
- batch_size: 4
- context_length: 512
- warmup_steps: 5
- measurement_steps: 10

## Results

| Model | Mode | Mean (ms) | Std (ms) |
|---|---|---:|---:|
| small | forward | 15.979 | 0.011 |
| small | forward_backward | 54.184 | 3.017 |
| small | train | 59.601 | 2.788 |
| medium | forward | 47.000 | 0.188 |
| medium | forward_backward | 149.026 | 0.169 |
| medium | train | 168.591 | 0.139 |
| large | forward | 109.811 | 0.064 |
| large | forward_backward | 326.986 | 0.499 |
| large | train | 375.863 | 0.164 |
| xl | forward | 326.959 | 0.320 |
| xl | forward_backward | OOM | - |
| xl | train | not run (expected OOM) | - |