
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config

torch._inductor.config.debug = True
torch._inductor.config.verbose_progress = True
torch._functorch.config.debug_partitioner = True


isolate_fails_code_str = None



# torch version: 2.1.1+cu121
# torch cuda version: 12.1
# torch git version: 4c55dc50355d5e923642c59ad2a23d6ad54711e7


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Thu_Sep_12_02:18:05_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.77 
# Build cuda_12.6.r12.6/compiler.34841621_0 

# GPU Hardware Info: 
# NVIDIA A100 80GB PCIe : 2 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, primals_1, primals_2):
        convert_element_type = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        amax = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp = torch.ops.aten.exp.default(sub)
        sum_1 = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        ne = torch.ops.aten.ne.Scalar(primals_2, -100)
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(ne, primals_2, full_default);  full_default = None
        unsqueeze = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather = torch.ops.aten.gather.default(convert_element_type_1, 1, unsqueeze);  unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(ne, neg, full_default_1);  ne = neg = full_default_1 = None
        return [where_1, primals_2, convert_element_type_1]
        
def load_args(reader):
    buf0 = reader.storage(None, 8405385216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (32768, 128256), dtype=torch.bfloat16, requires_grad=True, is_leaf=True)  # primals_1
    buf1 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf1, (32768,), dtype=torch.int64, is_leaf=True)  # primals_2
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
