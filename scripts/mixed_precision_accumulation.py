import torch


# 1. FP32 value + FP32 accumulation
s = torch.tensor(0, dtype=torch.float32)

for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)

print("FP32 + FP32 accumulation:", s)


# 2. FP16 value + FP16 accumulation
s = torch.tensor(0, dtype=torch.float16)

for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)

print("FP16 + FP16 accumulation:", s)


# 3. FP16 value, but accumulate in FP32
s = torch.tensor(0, dtype=torch.float32)

for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)

print("FP16 value + FP32 accumulation:", s)