
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


# kernel path: /tmp/tmp.czIkyP4WhG/torchinductor_ec0342/s5/cs5chhiomwon7r3gktzliapvn7bmhjx566jqn5ecufqolahbnigb.py
# Source Nodes: [loss_module], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss_module => full_default
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[4294967296], filename=__file__, meta={'signature': {0: '*bf16', 1: 'i64'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4202692608
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:].to(tl.int64)
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/tmp.czIkyP4WhG/torchinductor_ec0342/5q/c5q4cle4iwbrdku5mixlfo4agrvaxxsy5owmdxhwk35xduntbefa.py
# Source Nodes: [loss_module], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# loss_module => full_default
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*i64', 1: '*bf16', 2: 'i64'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = -1.0
    tl.store(out_ptr0 + (tmp5 + (128256*x0)), tmp6, None)
''')


# kernel path: /tmp/tmp.czIkyP4WhG/torchinductor_ec0342/zr/czrhk6iikt5quduzhnptiihrkinr24sat6pnkldfwh5fpgxnpcek.py
# Source Nodes: [loss_module], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# loss_module => full_default_1
triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*bf16', 1: '*i64', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i64', 6: 'i64'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32768
    rnumel = 128256
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :].to(tl.int64)
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (128256*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tl.full([1, 1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp5 = 0.0
        tmp6 = tl.where(tmp3, tmp4, tmp5)
        tmp7 = tmp0 * tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (128256*x0)), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_ptr3 + (r1 + (128256*x0)), rmask, other=0).to(tl.float32)
        tmp13 = tl.full([1, 1], -100, tl.int64)
        tmp14 = tmp1 != tmp13
        tmp15 = 0.0
        tmp16 = tl.where(tmp14, tmp4, tmp15)
        tmp17 = tmp12 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tl.exp(tmp20)
        tmp22 = tmp21 * tmp10
        tmp23 = tmp18 - tmp22
        tmp24 = tmp23.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (128256*x0)), tmp24, rmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_2, convert_element_type_1, tangents_1 = args
    args.clear()
    assert_size_stride(primals_2, (32768, ), (1, ))
    assert_size_stride(convert_element_type_1, (32768, 128256), (128256, 1))
    assert_size_stride(tangents_1, (32768, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((32768, 128256), (128256, 1), device='cuda', dtype=torch.bfloat16)
        # Source Nodes: [loss_module], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 4202692608, grid=grid(4202692608), stream=stream0)
        # Source Nodes: [loss_module], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_2, buf0, 32768, grid=grid(32768), stream=stream0)
        buf3 = empty_strided((32768, 128256), (128256, 1), device='cuda', dtype=torch.bfloat16)
        # Source Nodes: [loss_module], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_2, tangents_1, convert_element_type_1, buf3, 32768, 128256, grid=grid(32768), stream=stream0)
        del buf0
        del convert_element_type_1
        del primals_2
        del tangents_1
        return (buf3, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_2 = rand_strided((32768, ), (1, ), device='cuda:0', dtype=torch.int64)
    convert_element_type_1 = rand_strided((32768, 128256), (128256, 1), device='cuda:0', dtype=torch.bfloat16)
    tangents_1 = rand_strided((32768, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    return print_performance(lambda: call([primals_2, convert_element_type_1, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
