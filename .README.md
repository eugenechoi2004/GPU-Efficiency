# CrossEntropyLoss GPU Benchmarking

This repository benchmarks the forward and backward pass performance of `torch.nn.CrossEntropyLoss` using two modes of execution: PyTorch eager mode and `torch.compile`.

## File Overview

- `benchmark_compiled.py`: Runs the compiled version of the loss benchmark using `torch.compile`. It sets relevant environment variables for debugging and profiling.
- `benchmark_eager.py`: Runs the same benchmark using PyTorch’s eager execution for comparison.
- `extractor.py`: Pulls out and saves internal IRs at different stages of the compilation pipeline.
- `internals/`: Stores the intermediate representations dumped from the compiler. This includes FX graphs, TorchDynamo graphs, and Inductor IR.

