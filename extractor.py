import os
import time
import numpy as np

os.environ["TORCH_COMPILE_DEBUG"] = "1"

import torch
import torch.nn as nn
from torch._dynamo import export
import torch._inductor.config as inductor_conf

inductor_conf.debug = True
inductor_conf.verbose_progress = True

B, V = 32768, 128256
device = "cuda"
dtype = torch.float32
NUM_TRIALS = 1

loss_module = nn.CrossEntropyLoss(reduction="none")
def loss_fn(logits, targets):
    return loss_module(logits, targets)

logits_fx = torch.randn(B, V, device=device, dtype=dtype, requires_grad=True)
targets_fx = torch.randint(0, V, (B,), device=device)

fx_export = export(loss_fn)
fx_mod, guards = fx_export(logits_fx, targets_fx)
print("\n=== FX GRAPH ===")
fx_mod.graph.print_tabular()

def benchmark(fn, label):
    times = []
    for _ in range(NUM_TRIALS):
        x = torch.randn(B, V, device=device, dtype=dtype, requires_grad=True)
        y = torch.randint(0, V, (B,), device=device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start = time.time()
        out = fn(x, y)
        out.sum().backward()
        torch.cuda.synchronize()

        elapsed = time.time() - start
        times.append(elapsed)
        print(f"{label}: {elapsed:.4f}s")

    mean, std = np.mean(times), np.std(times)
    print(f"{label} mean: {mean:.4f}s ± {std:.4f}s\n")
    return mean, std

benchmark(loss_fn, "Eager")

compiled_fn = torch.compile(loss_fn)
benchmark(compiled_fn, "Compiled")
