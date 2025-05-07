
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/tmp.czIkyP4WhG/torchinductor_ec0342/rt/crtqxargralxa3433wjzyv7b6r7s3o3nt3agbhjawpwacoj6yavq.py
# Source Nodes: [loss_module], Original ATen: [aten._log_softmax]
# loss_module => amax, convert_element_type, convert_element_type_1, exp, log, sub, sub_1, sum_1
triton_red_fused__log_softmax_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32768, 131072],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i64', 3: 'i64'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128256
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :].to(tl.int64)
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128256*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (128256*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = tl.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (128256*x0)), rmask, other=0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13 - tmp3
        tmp15 = tl.log(tmp10)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp16.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (128256*x0)), tmp17, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/tmp.czIkyP4WhG/torchinductor_ec0342/gq/cgqf6hbcwogbgt6dsmkgxewofwt6geqwm3qxedyajdhilotanrvw.py
# Source Nodes: [loss_module], Original ATen: [aten.nll_loss_forward]
# loss_module => full_default_1, ne, neg, where_1
triton_poi_fused_nll_loss_forward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*i64', 1: '*bf16', 2: '*bf16', 3: 'i64'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_forward_1', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:].to(tl.int64)
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tl.where(tmp4 < 0, tmp4 + 128256, tmp4)
    tl.device_assert((0 <= tmp5) & (tmp5 < 128256), "index out of bounds: 0 <= tmp5 < 128256")
    tmp6 = tl.load(in_ptr1 + (tmp5 + (128256*x0)), None).to(tl.float32)
    tmp7 = -tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tl.store(out_ptr0 + (x0), tmp9, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2 = args
    args.clear()
    assert_size_stride(primals_1, (32768, 128256), (128256, 1))
    assert_size_stride(primals_2, (32768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf2 = empty_strided((32768, 128256), (128256, 1), device='cuda', dtype=torch.bfloat16)
        # Source Nodes: [loss_module], Original ATen: [aten._log_softmax]
        stream0 = get_cuda_stream(0)
        triton_red_fused__log_softmax_0.run(primals_1, buf2, 32768, 128256, grid=grid(32768), stream=stream0)
        del primals_1
        buf3 = empty_strided((32768, ), (1, ), device='cuda', dtype=torch.bfloat16)
        # Source Nodes: [loss_module], Original ATen: [aten.nll_loss_forward]
        triton_poi_fused_nll_loss_forward_1.run(primals_2, buf2, buf3, 32768, grid=grid(32768), stream=stream0)
        return (buf3, primals_2, buf2, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((32768, 128256), (128256, 1), device='cuda:0', dtype=torch.bfloat16)
    primals_2 = rand_strided((32768, ), (1, ), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
