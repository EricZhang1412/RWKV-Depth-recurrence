########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import gc
import importlib
import math
import os
import pdb, types, time, re, random
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.strategies import DeepSpeedStrategy
from rwkvfla.modules.token_shift import token_shift
from rwkvfla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from rwkvfla.ops.rwkv7.fused_k_update import fused_k_rwkv7
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from torch.utils.checkpoint import checkpoint

if importlib.util.find_spec("deepspeed"):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

try:
    print("RWKV_MY_TESTING", os.environ["RWKV_MY_TESTING"])
except BaseException:
    os.environ["RWKV_MY_TESTING"] = ""


def __nop(ob):
    return ob


ROCm_flag = torch.version.hip is not None
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyFunction = torch.compile


########################################################################################################
# CUDA Kernel
########################################################################################################


HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])

RWKV_TEST_DEMO=True
USE_CUDA_KERNEL=False
USE_CUDA_FAST_KERNEL=True
# DTYPE = torch.bfloat16
DTYPE = torch.half # better

if "x070" in os.environ["RWKV_MY_TESTING"] and RWKV_TEST_DEMO==False:
    CHUNK_LEN = 16

    if ROCm_flag is True:
        flags = [
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "-xhip",
            "-fopenmp",
            "-ffast-math",
            "-O3",
            "-munsafe-fp-atomics",
        ]
        load(
            name="wind_backstepping_hip",
            sources=["cuda/wkv7_hip.hip", "cuda/wkv7_op.hip"],
            is_python_module=False,
            verbose=True,
            extra_cuda_cflags=flags,
        )
    else:
        flags = [
            "-res-usage",
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
        ]
        load(
            name="wind_backstepping",
            sources=["cuda/wkv7_cuda.cu", "cuda/wkv7_op.cpp"],
            is_python_module=False,
            verbose=True,
            extra_cuda_cflags=flags,
        )

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, z, b):
            B, T, H, C = w.shape
            assert T % CHUNK_LEN == 0
            assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
            assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
            torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
            return y

        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype == torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w, q, k, v, z, b, s, sa = ctx.saved_tensors
            dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
            torch.ops.wind_backstepping.backward(
                w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db
            )
            return dw, dq, dk, dv, dz, db

    def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
        B, T, HC = q.shape
        q, w, k, v, a, b = [i.view(B, T, HC // 64, 64) for i in [q, w, k, v, a, b]]
        return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)


if RWKV_TEST_DEMO:
    ########################################################################################################
    # CUDA Kernel
    ########################################################################################################
    if USE_CUDA_FAST_KERNEL:
        load(name="wkv7s", sources=["cuda/wkv7s_op_infer.cpp", f"cuda/wkv7s_infer.cu"], is_python_module=False,
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
    
    if not USE_CUDA_FAST_KERNEL and USE_CUDA_KERNEL:

        from torch.utils.cpp_extension import load

        load(name="wkv7", sources=["cuda/wkv7_op_infer.cpp", f"cuda/wkv7_infer.cu"], is_python_module=False,
                            verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
        class WKV_7(torch.autograd.Function):
            @staticmethod
            def forward(ctx, r, w, k, v, a, b):
                with torch.no_grad():
                    B, T, C = r.size()
                    H = C // HEAD_SIZE
                    N = HEAD_SIZE
                    assert HEAD_SIZE == C // H
                    assert r.dtype == DTYPE
                    assert w.dtype == DTYPE
                    assert k.dtype == DTYPE
                    assert v.dtype == DTYPE
                    assert a.dtype == DTYPE
                    assert b.dtype == DTYPE
                    assert r.is_contiguous()
                    assert w.is_contiguous()
                    assert k.is_contiguous()
                    assert v.is_contiguous()
                    assert a.is_contiguous()
                    assert b.is_contiguous()
                    y = torch.empty((B, T, C), device=k.device, dtype=DTYPE,    memory_format=torch.contiguous_format)
                    torch.ops.wkv7.forward(B, T, C, H, r, w, k, v, a, b, y)
                    return y

        def RWKV7_OP(r, w, k, v, a, b):
            return WKV_7.apply(r, w, k, v, a, b)

    else:

        def RWKV7_OP(r, w, k, v, a, b):
            B, T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            r = r.view(B, T, H, N).float()
            k = k.view(B, T, H, N).float()
            v = v.view(B, T, H, N).float()
            a = a.view(B, T, H, N).float()
            b = b.view(B, T, H, N).float()
            w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
            out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
            state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)

            for t in range(T):
                kk = k[:, t, :].view(B, H, 1, N)
                rr = r[:, t, :].view(B, H, N, 1)
                vv = v[:, t, :].view(B, H, N, 1)
                aa = a[:, t, :].view(B, H, N, 1)
                bb = b[:, t, :].view(B, H, 1, N)
                state = state * w[: , t, :, None, :] + state @ aa @ bb + vv @ kk
                out[:, t, :] = (state @ rr).view(B, H, N)

                # another method using einsum
                #
                # kk = k[:, t, :]
                # rr = r[:, t, :]
                # vv = v[:, t, :]
                # aa = a[:, t, :]
                # bb = b[:, t, :]
                # sab = torch.einsum('bhik,bhk,bhj->bhij', state, aa, bb)
                # state = state * w[: , t, :, None, :] + sab + torch.einsum('bhj,bhi->bhij',    kk, vv)
                # out[:, t, :] = torch.einsum('bhj,bhij->bhi', rr, state)

            return out.view(B, T, C).to(dtype=DTYPE)
########################################################################################################


class RWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = (
                            math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        )
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = (
                            math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        )
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1**0.3)

            D_DECAY_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))  # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            # !!! 0.5 comes from F.softplus !!!
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

            D_AAA_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))  # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(
                torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4
            )

            D_MV_LORA = max(32, int(round((1.3 * (C**0.5)) / 32) * 32))  # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round((0.6 * (C**0.8)) / 32) * 32))  # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            # !!! notice eps value !!!
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

            self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.data.zero_()
            del www, zigzag, linear, ddd

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        # xx = self.time_shift(x) - x
        # equals to:
        xx = token_shift(x)

        # xr = torch.addcmul(x, xx, self.x_r)
        # xw = torch.addcmul(x, xx, self.x_w)
        # xk = torch.addcmul(x, xx, self.x_k)
        # xv = torch.addcmul(x, xx, self.x_v)
        # xa = torch.addcmul(x, xx, self.x_a)
        # xg = torch.addcmul(x, xx, self.x_g)
        # equivalent to:
        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(x, xx, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g)

        r = self.receptance(xr)
        # soft-clamp to (-inf, -0.5)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = torch.lerp(
                v, v_first, torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            )  # add value residual
        # a is "in-context learning rate"
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        # k = k * (1 + (a-1) * self.k_a) equivalent to:
        k = fused_k_rwkv7(k, a, self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


########################################################################################################

############RWKV_Tmix_x070_v2: shared layers############
class RWKV_Tmix_x070_v2(nn.Module):
    def __init__(self, args, group_id, loops_per_group):
        super().__init__()
        self.args = args
        self.grad_cp = getattr(args, 'grad_cp', 0)
        self.group_id = group_id
        self.loops_per_group = loops_per_group
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        num_all_layers = args.num_hidden_groups * args.inner_group_num
        with torch.no_grad():
            ratio_0_to_1 = group_id * loops_per_group / (num_all_layers - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (group_id * loops_per_group / num_all_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            
            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = (
                            math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        )
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = (
                            math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        )
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)
            D_DECAY_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) #!!! 0.5 comes from F.softplus!!!
            D_AAA_LORA = max(32, int(round(  (1.8*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)
            D_MV_LORA = max(32, int(round(  (1.3*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)
            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (0.6*(C**0.8))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))
            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) #!!! notice eps value!!!
            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()
            del www, zigzag, linear, ddd
    
    @MyFunction
    def _forward_impl(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        # xx = self.time_shift(x) - x
        # equals to:
        xx = token_shift(x)

        # xr = torch.addcmul(x, xx, self.x_r)
        # xw = torch.addcmul(x, xx, self.x_w)
        # xk = torch.addcmul(x, xx, self.x_k)
        # xv = torch.addcmul(x, xx, self.x_v)
        # xa = torch.addcmul(x, xx, self.x_a)
        # xg = torch.addcmul(x, xx, self.x_g)
        # equivalent to:
        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(x, xx, self.x_r, self.x_w, self.x_k, self.x_v, self.x_a, self.x_g)
        r = self.receptance(xr)
        # soft-clamp to (-inf, -0.5)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)

        if self.group_id * self.loops_per_group == 0:
            v_first = v  # store the v of the first layer
        else:
            v = torch.lerp(
                v, v_first, torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            )  # add value residual
        
        # a is "in-context learning rate"
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        # k = k * (1 + (a-1) * self.k_a) equivalent to:
        k = fused_k_rwkv7(k, a, self.k_a)

        if RWKV_TEST_DEMO==False:
            x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a)
        else:
            x = RWKV7_OP(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first
    def forward(self, x, v_first):
        if self.grad_cp > 0:
            return checkpoint(self._forward_impl, x, v_first, use_reentrant=False)
        else:
            return self._forward_impl(x, v_first)

class RWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(
            -0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5)
        )
        self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        xx = token_shift(x)
        k = torch.addcmul(x, xx, self.x_k)
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

############RWKV_CMix_x070_v2: shared layers############
class RWKV_CMix_x070_v2(nn.Module):
    def __init__(self, args, group_id, loops_per_group):
        super().__init__()
        self.args = args
        self.grad_cp = getattr(args, 'grad_cp', 0)
        self.group_id = group_id
        self.loops_per_group = loops_per_group

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        num_all_layers = args.num_hidden_groups * args.inner_group_num
        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (group_id * loops_per_group / num_all_layers)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))
        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(
            -0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5)
        )
        self.value.weight.data.zero_()

    @MyFunction
    def _forward_impl(self, x):
        xx = token_shift(x)
        k = torch.addcmul(x, xx, self.x_k)
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)
    def forward(self, x):
        if self.grad_cp > 0:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first

class Block_v2(nn.Module):
    def __init__(self, args, group_id, loops_per_group):
        super().__init__()
        self.args = args
        self.grad_cp = getattr(args, 'grad_cp', 0)
        self.group_id = group_id
        self.loops_per_group = loops_per_group

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.group_id * loops_per_group == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070_v2(args, group_id, loops_per_group)
        self.ffn = RWKV_CMix_x070_v2(args, group_id, loops_per_group)
    def forward(self, x, v_first):
        if self.group_id * self.loops_per_group == 0:
            x = self.ln0(x)
        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn
        x = x + self.ffn(self.ln2(x))
        return x, v_first

class BlockGroup(nn.Module):
    def __init__(self, args, group_id):
        super().__init__()
        self.args = args
        self.rwkv_layers = nn.ModuleList(
                [
                    Block_v2(args, group_id, i) 
                        for i in range(args.inner_group_num)
                ]
            ) # inner_group_num layers per group

    def forward(
            self, 
            x, 
            v_first,
            output_x,
            output_v_first,
        ):
        layer_x_states = ()
        layer_v_first_states = ()
        for rwkv_layer in self.rwkv_layers:
            x_states, v_first_states = rwkv_layer(x, v_first) # layer_output[0] is x, layer_output[1] is v_first

            if output_x:
                layer_x_states = layer_x_states + (x_states,)
            if output_v_first:
                layer_v_first_states = layer_v_first_states + (v_first_states,)
        outputs = (x_states, v_first_states)
        # if output_x:
        #     outputs = outputs + (layer_x_states,)
        # if output_v_first:
        #     outputs = outputs + (layer_v_first_states,)
        return outputs

class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)

def sample_repeat_layers(
        num_layers: int,
        min_repeat: int = 1,
        max_repeat: int = 4,
        repeat_prob: float = 0.4,
    ) -> dict:
        """
        随机采样每一层是否要 repeat，以及重复多少次
        :param num_layers: 总共的 group 层数
        :param min_repeat: 最少重复次数（默认1）
        :param max_repeat: 最大重复次数
        :param repeat_prob: 每层有概率开启 repeat（否则使用默认1次）
        :return: dict[layer_idx] = repeat_times
        """
        repeat_layers = {}
        for i in range(num_layers):
            if random.random() < repeat_prob:
                repeat_times = random.randint(min_repeat, max_repeat)
                repeat_layers[i] = repeat_times
            # else 默认不加，表示只执行1次
        return repeat_layers

class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, "dim_att"):
            args.dim_att = args.n_embd
        if not hasattr(args, "dim_ffn"):
            args.dim_ffn = int(
                (args.n_embd * 3.5) // 32 * 32
            )  # default = 3.5x emb size
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def configure_optimizers(self):
        args = self.args

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if "att.w0" in n:
                lr_2x.add(n)
            elif (
                (len(p.squeeze().shape) >= 2)
                and (args.weight_decay > 0)
                and (".weight" in n)
            ):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if self.trainer.is_global_zero:
            print("decay", lr_decay, "\n")
            print("1x", lr_1x, "\n")
            print("2x", lr_2x, "\n")

        param_dict = {n: p for n, p in self.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[n] for n in lr_1x],
                "weight_decay": 0.0,
                "my_lr_scale": 1.0,
            },
            {
                "params": [param_dict[n] for n in lr_2x],
                "weight_decay": 0.0,
                "my_lr_scale": 2.0,
            },
        ]

        if args.weight_decay > 0:
            optim_groups += [
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "weight_decay": args.weight_decay,
                    "my_lr_scale": 1.0,
                }
            ]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adamw_mode=True,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adam_w_mode=True,
                amsgrad=False,
            )
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adamw_mode=False,
                    weight_decay=0,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adam_w_mode=False,
                weight_decay=0,
                amsgrad=False,
            )

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            """
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(
                f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end=""
            )

            scale = 1.0
            if (
                "ln_" in n
                or ".ln" in n
                or "time_" in n
                or "_mask" in n
                or "pos_emb" in n
                or ".mask." in n
                or n.endswith("_w")
                or n.endswith("_w1")
                or n.endswith("_w2")
                or n.endswith("_bias")
                or (".weight" not in n)
            ):
                if "ln_x.weight" in n:
                    layer_scale = (1 + int(n.split(".")[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale**0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.args.vocab_size > self.args.n_embd:
                    scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith(".weight")  # should always be true

                zero = [
                    ".att.output.",
                    ".ffn.value.",
                    ".ffn.receptance.",
                    ".ffnPre.value.",
                    ".ffnPre.receptance.",
                    "head_q.",
                    ".oo.",
                    ".rr.",
                ]

                for kk in zero:
                    if kk in n:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()

        print("model params", n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m

class RWKV_shared(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        #####################################################
        self.rand_step = getattr(args, 'rand_step', 0)
        self.mean_recurrence = getattr(args, 'mean_recurrence', 1)
        self.mean_backprop_depth = getattr(args,'mean_backprop_depth', 1)
        self.sampling_scheme = getattr(args, 'sampling_scheme', 'none')
        self.lockstep_n = getattr(args,'lockstep_n', False)
        self.lockstep_k = getattr(args,'lockstep_k', False)

        self.injection_type = getattr(args, 'injection_type', 'none')

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.rwkv_layer_groups = nn.ModuleList([BlockGroup(args, i) for i in range(args.num_hidden_groups)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        if self.injection_type == "linear":
            self.input_injection_adapter = nn.Linear(
                args.n_embd * 2,
                args.n_embd,
                bias=True,
            )
    
    def configure_optimizers(self):
        args = self.args

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if "att.w0" in n:
                lr_2x.add(n)
            elif (
                (len(p.squeeze().shape) >= 2)
                and (args.weight_decay > 0)
                and (".weight" in n)
            ):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if self.trainer.is_global_zero:
            print("decay", lr_decay, "\n")
            print("1x", lr_1x, "\n")
            print("2x", lr_2x, "\n")

        param_dict = {n: p for n, p in self.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[n] for n in lr_1x],
                "weight_decay": 0.0,
                "my_lr_scale": 1.0,
            },
            {
                "params": [param_dict[n] for n in lr_2x],
                "weight_decay": 0.0,
                "my_lr_scale": 2.0,
            },
        ]

        if args.weight_decay > 0:
            optim_groups += [
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "weight_decay": args.weight_decay,
                    "my_lr_scale": 1.0,
                }
            ]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adamw_mode=True,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adam_w_mode=True,
                amsgrad=False,
            )
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adamw_mode=False,
                    weight_decay=0,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adam_w_mode=False,
                weight_decay=0,
                amsgrad=False,
            )

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(
        self, 
        idx,
        output_x=False,
        output_v_first=False,
    ):
        args = self.args
        B, T = idx.size()

        if torch.any(idx < 0) or torch.any(idx >= self.args.vocab_size):
            print(f"[Error] idx out of bounds in forward(): min={idx.min().item()}, max={idx.max().item()}, vocab_size={self.args.vocab_size}")
            raise ValueError("Input token index out of bounds.")

        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        all_x_states = ()
        all_v_first_states = ()
        all_logits = []

        x = self.emb(idx)
        v_first = torch.empty_like(x)
        # import pdb
        # pdb.set_trace()
        ########## Get the number of the rwkv_shared groups ##########
        num_hidden_groups = len(self.rwkv_layer_groups)
        num_inner_layers = len(self.rwkv_layer_groups[0].rwkv_layers)

        settings_num_hidden_groups = args.num_hidden_groups
        settings_num_inner_layers = args.inner_group_num
        assert num_hidden_groups == settings_num_hidden_groups, "The number of hidden groups does not match the settings."
        assert num_inner_layers == settings_num_inner_layers, "The number of inner layers does not match the settings."

        # repeat_layers = {
            
        # }
        repeat_layers = sample_repeat_layers(num_layers=len(self.rwkv_layer_groups))
        total_steps = len(self.rwkv_layer_groups)  # 假设你用了所有 group

        for i in range(total_steps):
            repeat_count = repeat_layers.get(i, 1)  # 默认为1次

            for _ in range(repeat_count):
                outputs = self.rwkv_layer_groups[i](
                    x, v_first,
                    output_x=output_x,
                    output_v_first=output_v_first
                )
                x_states, v_first_states = outputs[0], outputs[1]

                if output_x:
                    all_x_states += (x_states,)
                if output_v_first:
                    all_v_first_states += (v_first_states,)

                if self.injection_type == "add":
                    x = x + x_states
                    v_first = v_first + v_first_states
                elif self.injection_type in ["linear", "ffn"]:
                    # x = self.input_injection_adapter(torch.cat([x, x_states], dim=-1))
                    x = self.input_injection_adapter(torch.cat([
                            F.layer_norm(x, x.shape[-1:]),
                            F.layer_norm(x_states, x_states.shape[-1:])
                        ], dim=-1))
                    # v_first = self.input_injection_adapter(torch.cat([v_first, v_first_states], dim=-1))
                    v_first = self.input_injection_adapter(torch.cat([
                            F.layer_norm(v_first, v_first.shape[-1:]),
                            F.layer_norm(v_first_states, v_first_states.shape[-1:])
                        ], dim=-1))
                else:
                    raise NotImplementedError(f"Unknown injection type: {self.injection_type}")

            logits = self.head(self.ln_out(x))
            all_logits.append(logits)

        x = self.ln_out(x)
        x = self.head(x)

        all_logits.append(x)

        ######### get all loop times from "repeat_layers"
        total_repeat_times = sum(repeat_layers.values())
        # mean_repeat_times = total_repeat_times / len(self.rwkv_layer_groups)
        return all_logits, total_repeat_times
    
    def training_step(self, batch, batch_idx):
        idx, targets = batch
        # logits, total_repeat_times = self(idx)
        logits_list, total_repeat_times = self(idx)
        # print(f"total_repeat_times: {total_repeat_times}")
        # print(f"logits_list length: {len(logits_list)}")
        elbayad_exponent = getattr(self.args, "elbayad_exponent", 1.6)
        weights = torch.arange(1, len(logits_list)+1, device=logits_list[0].device, dtype=torch.float32)
        weights = weights ** elbayad_exponent
        weights = weights / weights.sum()

        total_loss = 0
        for i, logits in enumerate(logits_list):
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # loss = loss / total_repeat_times
            total_loss += weights[i] * loss

        self.log('train_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return L2Wrap.apply(total_loss, logits_list[-1])

        # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        # return L2Wrap.apply(loss, logits)

    # def training_step_end(self, batch_parts):
    #     all = self.all_gather(batch_parts)
    #     if self.trainer.is_global_zero:
    #         self.trainer.my_loss_all = all
    def training_step_end(self, batch_parts):
        try:
            # 确保batch_parts是tensor
            if not isinstance(batch_parts, torch.Tensor):
                print(f"Warning: batch_parts is not a tensor: {type(batch_parts)}")
                return batch_parts

            # 检查tensor是否有效
            if torch.isnan(batch_parts).any() or torch.isinf(batch_parts).any():
                print("Warning: batch_parts contains NaN or Inf values")
                return batch_parts

            # 进行all_gather
            all = self.all_gather(batch_parts)

            if self.trainer.is_global_zero:
                self.trainer.my_loss_all = all

            return batch_parts

        except Exception as e:
            print(f"Error in training_step_end: {e}")
            # 返回原始数据，继续训练
            return batch_parts

    def generate_init_weight(self):
        print(
            """
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(
                f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end=""
            )

            scale = 1.0
            if (
                "ln_" in n
                or ".ln" in n
                or "time_" in n
                or "_mask" in n
                or "pos_emb" in n
                or ".mask." in n
                or n.endswith("_w")
                or n.endswith("_w1")
                or n.endswith("_w2")
                or n.endswith("_bias")
                or (".weight" not in n)
            ):
                if "ln_x.weight" in n:
                    layer_scale = (1 + int(n.split(".")[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale**0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.args.vocab_size > self.args.n_embd:
                    scale = 0.5 * math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith(".weight")  # should always be true

                zero = [
                    ".att.output.",
                    ".ffn.value.",
                    ".ffn.receptance.",
                    ".ffnPre.value.",
                    ".ffnPre.receptance.",
                    "head_q.",
                    ".oo.",
                    ".rr.",
                ]

                for kk in zero:
                    if kk in n:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()

        print("model params", n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m


# load(name="wkv7s", sources=["cuda/wkv7s_op_infer.cpp", f"cuda/wkv7s_infer.cu"], is_python_module=False,
#                     verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
class WKV_7_fast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, r, w, k, v, a, b):
        with torch.no_grad():
            T, C = r.size()
            H = C // HEAD_SIZE
            N = HEAD_SIZE
            assert HEAD_SIZE == C // H
            assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
            assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
            y = torch.empty((T, C), device=k.device, dtype=DTYPE, requires_grad=False, memory_format=torch.contiguous_format)
            torch.ops.wkv7s.forward(1, T, C, H, state, r, w, k, v, a, b, y)
            return y
def RWKV7_OP(state, r, w, k, v, a, b):
    return WKV_7_fast.apply(state, r, w, k, v, a, b)

class RWKV_x070_infer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_embd = args.n_embd
        self.num_hidden_groups = args.num_hidden_groups
        self.inner_group_num = args.inner_group_num
        self.eval()
        
        MODEL_NAME = "/data/projects/RWKV-LM-V7-Depth-recur/out/L32-D2048-x070/rwkv-52.pth"
        self.z = torch.load(MODEL_NAME, map_location='cuda')
        z = self.z
        self.n_head, self.head_size = z['rwkv_layer_groups.0.rwkv_layers.0.att.r_k'].shape

        keys = list(z.keys())
        for k in keys:
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                z[k] = z[k].t()
            z[k] = z[k].squeeze().to(dtype=DTYPE)
            if k.endswith('att.r_k'): z[k] = z[k].flatten()
        assert self.head_size == args.head_size

        z['emb.weight'] = F.layer_norm(z['emb.weight'], (args.n_embd,), weight=z['rwkv_layer_groups.0.rwkv_layers.0.ln0.weight'], bias=z['rwkv_layer_groups.0.rwkv_layers.0.ln0.bias'])
        z['rwkv_layer_groups.0.rwkv_layers.0.att.v0'] = z['rwkv_layer_groups.0.rwkv_layers.0.att.a0'] # actually ignored
        z['rwkv_layer_groups.0.rwkv_layers.0.att.v1'] = z['rwkv_layer_groups.0.rwkv_layers.0.att.a1'] # actually ignored
        z['rwkv_layer_groups.0.rwkv_layers.0.att.v2'] = z['rwkv_layer_groups.0.rwkv_layers.0.att.a2'] # actually ignored

    def forward(self, idx, state, full_output=False):
        if state == None:
            tot_layers = self.num_hidden_groups * self.inner_group_num
            state = [None for _ in range(tot_layers * 3)]
            for i in range(tot_layers): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
                state[i*3+0] = torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")
                state[i*3+1] = torch.zeros((args.n_embd // args.head_size, args.head_size, args.head_size), dtype=torch.float, requires_grad=False, device="cuda")
                state[i*3+2] = torch.zeros(args.n_embd, dtype=DTYPE, requires_grad=False, device="cuda")

        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)
        else:
            return self.forward_one(idx, state)

    @MyFunction
    def forward_one(self, idx:int, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x070_CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state
        
    @MyFunction
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            for i in range(self.n_layer):
                bbb = f'blocks.{i}.'
                att = f'blocks.{i}.att.'
                ffn = f'blocks.{i}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[i*3+2] = RWKV_x070_CMix_seq(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state

########################################################################################################

@torch.jit.script
def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    xx = x_prev - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
    w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

    vk = v.view(H,N,1) @ k.view(H,1,N)
    ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
    state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
    xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

    xx = torch.nn.functional.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
    xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
    return (xx * g) @ O_, x, state, v_first

@torch.jit.script
def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
    T = x.shape[0]
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

    r = xr @ R_
    w = torch.tanh(xw @ w1) @ w2
    k = xk @ K_
    v = xv @ V_
    a = torch.sigmoid(a0 + (xa @ a1) @ a2)
    g = torch.sigmoid(xg @ g1) @ g2

    kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
    k = k * (1 + (a-1) * k_a)
    if layer_id == 0: v_first = v
    else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

    ######## cuda-free method 
    # w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
    # for t in range(T):
    #     r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
    #     vk = v_.view(H,N,1) @ k_.view(H,1,N)
    #     ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
    #     state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
    #     xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

    w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5
    xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

    xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
    xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
    return (xx * g) @ O_, x[-1,:], state, v_first

########################################################################################################

@torch.jit.script
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x

@torch.jit.script
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x[-1,:]