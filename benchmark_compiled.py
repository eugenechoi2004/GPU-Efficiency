import torch
import torch.nn as nn
import time
import numpy as np
import torch.cuda.profiler as profiler


B, V = 32768, 128256
device = 'cuda'
dtype = torch.float32xqx
NUM_TRIALS = 100

def benchmark_loss_fn(loss_fn, label="Compiled"):
    times = []
    targets = torch.randint(0, V, (B,), device=device)
    for _ in range(5):
        logits = torch.randn(B, V, dtype=dtype, device=device, requires_grad=True)
        loss = loss_fn(logits, targets)
        loss.sum().backward()
    profiler.start() 
    for _ in range(NUM_TRIALS):
        logits = torch.randn(B, V, dtype=dtype, device=device, requires_grad=True)
        targets = torch.randint(0, V, (B,), device=device)
        torch.cuda.synchronize()
        start = time.time()

        loss = loss_fn(logits, targets)
        loss.backward()

        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"{label} run: {elapsed:.4f} seconds")
    profiler.stop()  

    times = np.array(times)
    print(times)
    print(f"\n{label} mean: {times.mean():.4f} s ± {times.std():.4f} s\n")
    return times.mean(), times.std()

if __name__ == "__main__":
    base_loss_fn = nn.CrossEntropyLoss(reduction='sum')
    compiled_loss_fn = torch.compile(base_loss_fn)
    benchmark_loss_fn(compiled_loss_fn, label="Compiled")
