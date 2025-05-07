class GraphModule(torch.nn.Module):
    def forward(self, primals_2: i64[32768], convert_element_type_1: bf16[32768, 128256], tangents_1: bf16[32768]):
        # File: /scratch/network/OPEN-CATALYST/bench_v0.py:28, code: return loss_module(logits, targets)
        full_default: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: bf16[] = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        unsqueeze_1: i64[32768, 1] = torch.ops.aten.unsqueeze.default(primals_2, 1);  primals_2 = None
        ne_2: b8[32768, 1] = torch.ops.aten.ne.Scalar(unsqueeze_1, -100)
        where_2: i64[32768, 1] = torch.ops.aten.where.self(ne_2, unsqueeze_1, full_default);  unsqueeze_1 = full_default = None
        full_default_3: bf16[32768, 128256] = torch.ops.aten.full.default([32768, 128256], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        scatter: bf16[32768, 128256] = torch.ops.aten.scatter.value(full_default_3, 1, where_2, -1.0);  full_default_3 = where_2 = None
        unsqueeze_2: bf16[32768, 1] = torch.ops.aten.unsqueeze.default(tangents_1, 1);  tangents_1 = None
        where_3: bf16[32768, 1] = torch.ops.aten.where.self(ne_2, unsqueeze_2, full_default_1);  ne_2 = unsqueeze_2 = full_default_1 = None
        mul: bf16[32768, 128256] = torch.ops.aten.mul.Tensor(scatter, where_3);  scatter = where_3 = None
        convert_element_type_2: f32[32768, 128256] = torch.ops.prims.convert_element_type.default(mul, torch.float32);  mul = None
        convert_element_type_3: f32[32768, 128256] = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        exp_1: f32[32768, 128256] = torch.ops.aten.exp.default(convert_element_type_3);  convert_element_type_3 = None
        sum_2: f32[32768, 1] = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [1], True)
        mul_1: f32[32768, 128256] = torch.ops.aten.mul.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        sub_2: f32[32768, 128256] = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_1);  convert_element_type_2 = mul_1 = None
        convert_element_type_4: bf16[32768, 128256] = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        return [convert_element_type_4, None]
        