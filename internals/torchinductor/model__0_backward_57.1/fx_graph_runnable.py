
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

    
    
    def forward(self, primals_2, convert_element_type_1, tangents_1):
        full_default = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(primals_2, 1);  primals_2 = None
        ne_2 = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        where_2 = torch.ops.aten.where.self(ne_2, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None
        full_default_3 = torch.ops.aten.full.default([32768, 128256], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter = torch.ops.aten.scatter.value(full_default_3, 1, where_2, -1.0);  full_default_3 = where_2 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(tangents_1, 1);  tangents_1 = None
        where_3 = torch.ops.aten.where.self(ne_2, unsqueeze_2, full_default_1);  ne_2 = unsqueeze_2 = full_default_1 = None
        mul = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        exp_1 = torch.ops.aten.exp.default(convert_element_type_3);  convert_element_type_3 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [1], True)
        mul_1 = torch.ops.aten.mul.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        sub_2 = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_1);  convert_element_type_2 = mul_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        return [convert_element_type_4, None]
        
def load_args(reader):
    buf0 = reader.storage(None, 262144, device=device(type='cuda', index=0), dtype_hint=torch.int64)
    reader.tensor(buf0, (32768,), dtype=torch.int64, is_leaf=True)  # primals_2
    buf1 = reader.storage(None, 8405385216, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (32768, 128256), dtype=torch.bfloat16, is_leaf=True)  # convert_element_type_1
    buf2 = reader.storage(None, 65536, device=device(type='cuda', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf2, (32768,), dtype=torch.bfloat16, is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
