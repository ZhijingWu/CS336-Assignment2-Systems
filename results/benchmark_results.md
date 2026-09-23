## Final Benchmark Results on RTX PRO 6000

Environment:
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- VRAM: ~95 GB
- PyTorch: 2.11.0+cu130
- CUDA: 13.0
- batch_size: 4
- context_length: 512
- warmup_steps: 5
- measurement_steps: 10

| Model | Mode | Mean (ms) | Std (ms) |
|---|---|---:|---:|
| small | forward | 18.074 | 0.094 |
| small | forward_backward | 54.664 | 0.230 |
| small | train | 61.718 | 0.397 |
| medium | forward | 48.046 | 0.654 |
| medium | forward_backward | 145.744 | 0.921 |
| medium | train | 169.378 | 0.242 |
| large | forward | 110.397 | 0.179 |
| large | forward_backward | 332.668 | 0.179 |
| large | train | 385.523 | 0.255 |
| xl | forward | 324.926 | 0.045 |
| xl | forward_backward | 899.741 | 0.300 |
| xl | train | 1086.688 | 0.456 |
| 10B | forward | 1038.298 | 0.046 |
| 10B | forward_backward | OOM | - |
| 10B | train | not run | - |

## Memory Notes

- `10B forward` succeeded on the RTX PRO 6000.
- `10B forward_backward` caused CUDA OOM with batch size 4 and context length 512.
- At the OOM point, PyTorch had allocated approximately 93.71 GiB on a GPU with approximately 94.97 GiB total capacity.
- `10B train` was not attempted because it requires at least as much memory as `forward_backward`, plus optimizer-related memory.