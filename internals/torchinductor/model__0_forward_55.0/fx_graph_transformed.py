class GraphModule(torch.nn.Module):
    def forward(self, primals_1: bf16[32768, 128256], primals_2: i64[32768]):
        # File: /scratch/network/OPEN-CATALYST/bench_v0.py:28, code: return loss_module(logits, targets)
        convert_element_type: f32[32768, 128256] = torch.ops.prims.convert_element_type.default(primals_1, torch.float32);  primals_1 = None
        amax: f32[32768, 1] = torch.ops.aten.amax.default(convert_element_type, [1], True)
        sub: f32[32768, 128256] = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: f32[32768, 128256] = torch.ops.aten.exp.default(sub)
        sum_1: f32[32768, 1] = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: f32[32768, 1] = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: f32[32768, 128256] = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_1: bf16[32768, 128256] = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        ne: b8[32768] = torch.ops.aten.ne.Scalar(primals_2, -100)
        full_default: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: i64[32768] = torch.ops.aten.where.self(ne, primals_2, full_default);  full_default = None
        unsqueeze: i64[32768, 1] = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: bf16[32768, 1] = torch.ops.aten.gather.default(convert_element_type_1, 1, unsqueeze);  unsqueeze = None
        squeeze: bf16[32768] = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: bf16[32768] = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: bf16[] = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: bf16[32768] = torch.ops.aten.where.self(ne, neg, full_default_1);  ne = neg = full_default_1 = None
        return [where_1, primals_2, convert_element_type_1]
        