########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib, pdb, types, time, re
from typing import List
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

from torch.utils.checkpoint import checkpoint

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop

if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method

MyStatic = torch.jit.script

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])

if 'x070' in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16

    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(
        name="wind_backstepping", 
        sources=[
            f'/data/projects/RWKV-Depth-recurrence/RWKV-v7/train_temp/cuda/wkv7_cuda.cu', 
            '/data/projects/RWKV-Depth-recurrence/RWKV-v7/train_temp/cuda/wkv7_op.cpp'
        ], 
        is_python_module=False, 
        verbose=True, 
        extra_cuda_cflags=flags
    )

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w,q,k,v,z,b):
            B,T,H,C = w.shape 
            assert T%CHUNK_LEN == 0
            assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
            assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
            y = torch.empty_like(v)
            s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
            sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
            torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
            ctx.save_for_backward(w,q,k,v,z,b,s,sa)
            return y
        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype==torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w,q,k,v,z,b,s,sa = ctx.saved_tensors
            dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
            torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
            return dw,dq,dk,dv,dz,db

    def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
        B,T,HC = q.shape
        q,w,k,v,a,b = [i.view(B,T,HC//64,64) for i in [q,w,k,v,a,b]]
        return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)

########################################################################################################

############RWKV_Tmix_x070_v2: shared layers############
class RWKV_Tmix_x070_v2(MyModule):
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
                        gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
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
    
    @MyFunction
    def _forward_impl(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        if self.group_id * self.loops_per_group == 0:
            v_first = v # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual
        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x, v_first
    def forward(self, x, v_first):
        if self.grad_cp > 0:
            return checkpoint(self._forward_impl, x, v_first, use_reentrant=False)
        else:
            return self._forward_impl(x, v_first)

    # @MyFunction
    # def forward(self, x, v_first):
    #     B, T, C = x.size()
    #     H = self.n_head
    #     xx = self.time_shift(x) - x

    #     xr = x + xx * self.x_r
    #     xw = x + xx * self.x_w
    #     xk = x + xx * self.x_k
    #     xv = x + xx * self.x_v
    #     xa = x + xx * self.x_a
    #     xg = x + xx * self.x_g
        
    #     r = self.receptance(xr)
    #     w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
    #     k = self.key(xk)
    #     v = self.value(xv)
        

    #     if self.group_id * self.loops_per_group == 0:
    #         v_first = v # store the v of the first layer
    #     else:
    #         v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2) # add value residual

    #     a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
    #     g = torch.sigmoid(xg @ self.g1) @ self.g2
    #     kk = k * self.k_k
    #     kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
    #     k = k * (1 + (a-1) * self.k_a)
        
    #     x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
    #     x = self.ln_x(x.view(B * T, C)).view(B, T, C)
    #     x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
    #     x = self.output(x * g)
    #     return x, v_first

    


############RWKV_CMix_x070_v2: shared layers############
class RWKV_CMix_x070_v2(MyModule):
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
        self.key.weight.data.uniform_(-0.5/(args.n_embd**0.5), 0.5/(args.n_embd**0.5))
        self.value.weight.data.zero_()
    # @MyFunction
    # def forward(self, x):
    #     xx = self.time_shift(x) - x

    #     k = x + xx * self.x_k
    #     k = torch.relu(self.key(k)) ** 2
    #     return self.value(k)
    @MyFunction
    def _forward_impl(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)
    def forward(self, x):
        if self.grad_cp > 0:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

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

    # def _forward_impl(self, x, v_first):
    #     """实际的forward实现，用于gradient checkpointing"""
    #     if self.group_id * self.loops_per_group == 0:
    #         x = self.ln0(x)
    #     x_attn, v_first = self.att(self.ln1(x), v_first)
    #     x = x + x_attn
    #     x = x + self.ffn(self.ln2(x))
    #     return x, v_first
    # def forward(self, x, v_first):
    #     if self.grad_cp > 0 and self.training:
    #         # 使用gradient checkpointing
    #         return checkpoint(self._forward_impl, x, v_first, use_reentrant=False)
    #     else:
    #         # 正常forward
    #         return self._forward_impl(x, v_first)
    
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
            if ("att.w0" in n):
                lr_2x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and (".weight" in n):
                lr_decay.add(n)
            else:
                lr_1x.add(n)
        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        if self.trainer.is_global_zero:
            print('decay', lr_decay, '\n')
            print('1x', lr_1x, '\n')
            print('2x', lr_2x, '\n')
        param_dict = {n: p for n, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
            {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
        ]
        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
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
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        all_x_states = ()
        all_v_first_states = ()

        x = self.emb(idx)
        v_first = torch.empty_like(x)
        ########## Get the number of the rwkv_shared groups ##########
        num_hidden_groups = len(self.rwkv_layer_groups)
        num_inner_layers = len(self.rwkv_layer_groups[0].rwkv_layers)

        settings_num_hidden_groups = args.num_hidden_groups
        settings_num_inner_layers = args.inner_group_num
        assert num_hidden_groups == settings_num_hidden_groups, "The number of hidden groups does not match the settings."
        assert num_inner_layers == settings_num_inner_layers, "The number of inner layers does not match the settings."

        # print(f"num_hidden_groups: {num_hidden_groups}")
        # print(f"num_inner_layers: {num_inner_layers}")
        ##############################################################

        # for i in range(args.num_hidden_layers):
        #     layers_per_group = int(args.num_hidden_layers / args.num_hidden_groups)
        #     group_idx = int(i / layers_per_group)

        #     res_x = x
        #     res_v_first = v_first

        #     x_states, v_first_states = self.rwkv_layer_groups[group_idx](x, v_first, output_x=output_x, output_v_first=output_v_first)

        #     if output_x:
        #         all_x_states = all_x_states + (x_states,)
        #     if output_v_first:
        #         all_v_first_states = all_v_first_states + (v_first_states,)
        #     x = res_x + x_states
        #     v_first = res_v_first + v_first_states
        # import pdb; pdb.set_trace()
        for i in range(num_hidden_groups):
            res_x = x
            res_v_first = v_first
            outputs = self.rwkv_layer_groups[i](x, v_first, output_x=output_x, output_v_first=output_v_first)
            x_states, v_first_states = outputs[0], outputs[1]
            if output_x:
                all_x_states = all_x_states + (x_states,)
            if output_v_first:
                all_v_first_states = all_v_first_states + (v_first_states,)

            if self.injection_type == "add":
                x = res_x + x_states
                v_first = res_v_first + v_first_states
            # elif self.injection_type == "gate":
            #     x = x * input_embeds
            elif self.injection_type in ["linear", "ffn"]:
                x = self.input_injection_adapter(torch.cat([res_x, x_states], dim=-1))
                v_first = self.input_injection_adapter(torch.cat([res_v_first, v_first_states], dim=-1))
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
            f"""
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
            print(f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end="")

            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n or n.endswith('_w') or n.endswith('_w1') or n.endswith('_w2') or n.endswith('_bias') or (".weight" not in n):
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / (self.args.num_hidden_groups * self.args.inner_group_num)
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
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
                assert n.endswith('.weight') # should always be true

                zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']

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

        print('model params', n_params)
        print('params:', torch.sum(torch.tensor([p.numel() for p in self.parameters()])))
        gc.collect()
        torch.cuda.empty_cache()
        return m



# @MyFunction
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

if os.environ.get('RWKV_CUDA_ON') == '1':
    from torch.utils.cpp_extension import load
    current_path = "/data/projects/RWKV-Depth-recurrence/RWKV-v7/train_temp"
    load(
        name="wkv7s", 
        sources=[
            f"{current_path}/cuda/rwkv7_op.cpp", 
            f"{current_path}/cuda/rwkv7.cu"], 
        is_python_module=False,
        verbose=True, 
        extra_cuda_cflags=[
            "-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f'-D_N_={HEAD_SIZE}',
        ]
    )
    class WKV_7(torch.autograd.Function):
        @staticmethod
        def forward(ctx, state, r, w, k, v, a, b):
            with torch.no_grad():
                T, C = r.size()
                H = C // HEAD_SIZE
                N = HEAD_SIZE
                assert HEAD_SIZE == C // H
                assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
                assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
                y = torch.empty((T, C), device=DEVICE, dtype=r.dtype, requires_grad=False, memory_format=torch.contiguous_format)

                if DTYPE == torch.float16:
                    torch.ops.wkv7s.forward_fp16(1, T, C, H, state, r, w, k, v, a, b, y)
                elif DTYPE == torch.bfloat16:
                    torch.ops.wkv7s.forward_bf16(1, T, C, H, state, r, w, k, v, a, b, y)
                elif DTYPE == torch.float32:
                    torch.ops.wkv7s.forward_fp32(1, T, C, H, state, r, w, k, v, a, b, y)

            return y

    def RWKV7_OP(state, r, w, k, v, a, b):
        return WKV_7.apply(state, r, w, k, v, a, b)
    # @MyFunctionvvv
    def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1,a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
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

        w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5
        xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

        xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
        xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
        return (xx * g) @ O_, x[-1,:], state, v_first
else:
    # @MyFunction
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

        w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
        for t in range(T):
            r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
            vk = v_.view(H,N,1) @ k_.view(H,1,N)
            ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
            state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
            xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

        xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
        xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
        return (xx * g) @ O_, x[-1,:], state, v_first

########################################################################################################

# @MyFunction
def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
    xx = x_prev - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x

# @MyFunction
def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
    xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
    k = x + xx * x_k
    k = torch.relu(k @ K_) ** 2
    return k @ V_, x[-1,:]

########################################################################################################


MyStatic = torch.jit.script

class RWKV_shared_inference(MyModule):
    def __init__(self, model, strategy):
        global DTYPE, DEVICE
        super().__init__()
        self.eval()
        args = types.SimpleNamespace()
        self.args = args
        args.MODEL_NAME = model

        print(f'Loading {model} ({strategy})\n')

        ss = strategy.split(' ')
        DEVICE = ss[0]
        if ss[1] == 'fp16':
            DTYPE = torch.half
        elif ss[1] == 'fp32':
            DTYPE = torch.float32
        elif ss[1] == 'bf16':
            DTYPE = torch.bfloat16
        else:
            assert False, "currently rwkv7 strategy must be: cuda/cpu fp16/fp32/bf16"
            
        temp_z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu', mmap=True)

        # # print(temp_z.keys())
        for key in temp_z.keys():
            print(key)
        

        self.n_head, self.head_size = temp_z['rwkv_layer_groups.0.rwkv_layers.0.att.r_k'].shape
        args.head_size = self.head_size
        args.vocab_size, args.n_embd = temp_z['emb.weight'].shape

        print(f'vocab_size = {args.vocab_size}, n_embd = {args.n_embd}')
        print(f'n_head = {self.n_head}, head_size = {self.head_size}')

        args.num_hidden_layers = 400
        args.num_hidden_groups = 8
        args.inner_group_num = 1

        self.num_hidden_layers = args.num_hidden_layers
        self.num_hidden_groups = args.num_hidden_groups
        self.inner_group_num = args.inner_group_num
        

        args.n_layer = 0
        keys = list(temp_z.keys())
        self.z = {}

        for k in keys:
            layer_id = int(k.split('.')[1]) if ('rwkv_layers.' in k) else 0
            args.n_layer = max(args.n_layer, layer_id+1)
            # print(f'loading {args.n_layer} layers...')
            tensor = temp_z[k]
            if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
                tensor = tensor.t().contiguous()
            tensor = tensor.squeeze()
            if k.endswith('att.r_k'): 
                tensor = tensor.flatten()
            self.z[k] = tensor.to(DEVICE).to(DTYPE)
            del temp_z[k]
            if keys.index(k) % 5 == 0:
                torch.cuda.empty_cache()

        self.n_embd = args.n_embd
        # self.n_layer = args.num_hidden_groups * args.inner_group_num
        self.args.n_layer = args.n_layer

        self.z['emb.weight'] = F.layer_norm(
            self.z['emb.weight'], (args.n_embd,), weight=self.z['rwkv_layer_groups.0.rwkv_layers.0.ln0.weight'], bias=self.z['rwkv_layer_groups.0.rwkv_layers.0.ln0.bias']
        )
        self.z['blocks.0.att.v0'] = torch.empty(0, device=DEVICE, dtype=DTYPE) # actually ignored
        self.z['blocks.0.att.v1'] = torch.empty(0, device=DEVICE, dtype=DTYPE) # actually ignored
        self.z['blocks.0.att.v2'] = torch.empty(0, device=DEVICE, dtype=DTYPE) # actually ignored
        torch.cuda.empty_cache()

    def forward(self, idx, state, full_output=False):
        if state == None:
            state = [None for _ in range(self.args.n_layer * 3)]
            for i in range(self.args.n_layer):
                state[i*3+0] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)
                state[i*3+1] = torch.zeros((self.args.n_embd // self.args.head_size, self.args.head_size, self.args.head_size), dtype=torch.float, requires_grad=False, device=DEVICE)
                state[i*3+2] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)
        
        if type(idx) is list:
            if len(idx) > 1:
                return self.forward_seq(idx, state, full_output)
            else:
                return self.forward_one(idx[0], state)
        else:
            return self.forward_one(idx, state)

    # @MyStatic
    def forward_one(self, idx:int, state:List[torch.Tensor]):
        with torch.no_grad(): 
            z = self.z
            x = z['emb.weight'][idx]

            v_first = torch.empty_like(x)
            # layer-wise processing
            for i in range(self.num_hidden_layers):
                layers_per_group = int(self.num_hidden_layers / self.num_hidden_groups)
                loops_per_inner_block = int(layers_per_group / self.inner_group_num)
                group_idx = int(i / layers_per_group)
                inner_block_idx = int(i % layers_per_group / loops_per_inner_block)

                bbb = f'rwkv_layer_groups.{group_idx}.rwkv_layers.{inner_block_idx}.'
                att = f'rwkv_layer_groups.{group_idx}.rwkv_layers.{inner_block_idx}.att.'
                ffn = f'rwkv_layer_groups.{group_idx}.rwkv_layers.{inner_block_idx}.ffn.'

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                #### mapping from loop constant 'i' to actual looped layer index 'j'
                j = i % (self.num_hidden_groups * self.inner_group_num)
                
                xx, state[j*3+0], state[j*3+1], v_first = RWKV_x070_TMix_one(
                    i, # or maybe 'i' is better? i or j?
                    self.n_head, 
                    self.head_size, 
                    xx, 
                    state[j*3+0], 
                    v_first, 
                    state[j*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'], 
                    z[att+'a0'], z[att+'a1'], z[att+'a2'], 
                    z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'], 
                    z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias'])
                x = x + xx
                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[j*3+2] = RWKV_x070_CMix_one(xx, state[j*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx

                # if math.isnan(torch.min(x).item()): print(idx, i)

            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state

    # @MyStatic
    def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output=False):
        with torch.no_grad():
            z = self.z
            x = z['emb.weight'][idx]
            v_first = torch.empty_like(x)
            for i in range(self.num_hidden_layers):
                layers_per_group = int(self.num_hidden_layers / self.num_hidden_groups)
                loops_per_inner_block = int(layers_per_group / self.inner_group_num)
                group_idx = int(i / layers_per_group)
                inner_block_idx = int(i % layers_per_group / loops_per_inner_block)
                # print(f"i: {i}, loops_per_group: {layers_per_group}, \
                #         loops_per_inner_block: {loops_per_inner_block}, \
                #         group_idx: {group_idx}, \
                #         inner_block_idx: {inner_block_idx}")

                bbb = f'rwkv_layer_groups.{group_idx}.rwkv_layers.{inner_block_idx}.'
                att = f'rwkv_layer_groups.{group_idx}.rwkv_layers.{inner_block_idx}.att.'
                ffn = f'rwkv_layer_groups.{group_idx}.rwkv_layers.{inner_block_idx}.ffn.'
                
                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

                #### mapping from loop constant 'i' to actual looped layer index 'j'
                j = i % (self.num_hidden_groups * self.inner_group_num)

                xx, state[j*3+0], state[j*3+1], v_first = RWKV_x070_TMix_seq(
                    i, # or maybe 'i' is better? i or j?
                    self.n_head,
                    self.head_size,
                    xx,
                    state[j*3+0],
                    v_first,
                    state[j*3+1],
                    z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
                    z[att+'w0'], z[att+'w1'], z[att+'w2'],
                    z[att+'a0'], z[att+'a1'], z[att+'a2'],
                    z[att+'v0'], z[att+'v1'], z[att+'v2'],
                    z[att+'g1'], z[att+'g2'],
                    z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
                    z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
                    z[att+'ln_x.weight'], z[att+'ln_x.bias']
                )
                x = x + xx

                xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

                xx, state[j*3+2] = RWKV_x070_CMix_seq(xx, state[j*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
                x = x + xx
            
            if not full_output: x = x[-1,:]
            x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
            x = x @ z['head.weight']
            return x, state

        

# ###################### RWKV implemetation in pypi package ######################
# from typing import Optional
# import types, gc, os, time, re, math
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.matmul.allow_tf32 = True
# current_path = os.path.dirname(os.path.abspath(__file__))

# ########################################################################################################

# if os.environ.get('RWKV_JIT_ON') != '0':
#     os.environ["RWKV_JIT_ON"] = '1'
#     MyModule = torch.jit.ScriptModule
#     MyFunction = torch.jit.script_method
#     MyStatic = torch.jit.script
# else:
#     MyModule = torch.nn.Module
#     def __nop(ob):
#         return ob
#     MyFunction = __nop
#     MyStatic = __nop

# if os.environ.get('RWKV_CUDA_ON') == '1':
#     from torch.utils.cpp_extension import load
#     try:
#         load(
#             name=f"wkv_cuda",
#             sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu", f"{current_path}/cuda/gemm_fp16_cublas.cpp"],
#             verbose=True,
#             extra_ldflags=["cublas.lib" if os.name == "nt" else ""],
#             extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
#             is_python_module=False)
#         DISABLE_CUBLAS_GEMM = False
#     except:
#         print("Failed to build cuBLAS matmul, falling back to torch.matmul. Small model with fp16 will overflow.")
#         load(
#             name=f"wkv_cuda",
#             sources=[f"{current_path}/cuda/wrapper.cpp", f"{current_path}/cuda/operators.cu"],
#             verbose=True,
#             extra_cuda_cflags=["--use_fast_math", "-O3", "--extra-device-vectorization"],
#             extra_cflags=["-DDISABLE_CUBLAS_GEMM"],
#             is_python_module=False)
#         DISABLE_CUBLAS_GEMM = True

#     @MyStatic
#     def cuda_wkv(T: int, C: int, w, u, k, v, aa, bb, pp): # WKV (Weighted Key-Value) Attention for RWKV
#         assert 1 * C % min(C, 32) == 0
#         assert k.dtype == v.dtype == torch.float16 or k.dtype == v.dtype == torch.float32
#         assert w.dtype == u.dtype == aa.dtype == bb.dtype == pp.dtype == torch.float32
#         w = w.contiguous()
#         u = u.contiguous()
#         k = k.contiguous()
#         v = v.contiguous()
#         y = torch.empty((T, C), device=w.device, memory_format=torch.contiguous_format, dtype=k.dtype)
#         torch.ops.rwkv.wkv_forward(1, T, C, w, u, k, v, y, aa, bb, pp)
#         return y, aa, bb, pp
#     @MyStatic
#     def cuda_mm8_seq(B: int, N: int, M: int, x, w, mx, rx, my, ry): # 8-bit (weight) Matrix Multiplication for parallel batch
#         assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
#         assert x.dtype == torch.float32 or x.dtype == torch.float16
#         assert w.dtype == torch.uint8
#         assert x.shape == (B, N)
#         assert w.shape == (N, M)
#         assert rx.shape == mx.shape == (M,)
#         assert ry.shape == my.shape == (N, 1)
#         y = torch.empty((B, M), device=w.device, dtype=x.dtype)
#         torch.ops.rwkv.mm8_seq(B, N, M, x, w, mx, rx, my, ry, y)
#         return y
#     @MyStatic
#     def cuda_mm8_one(N: int, M: int, x, w, mx, rx, my, ry): # 8-bit (weight) Matrix Multiplication for single batch
#         assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
#         assert x.dtype == torch.float32 or x.dtype == torch.float16
#         assert w.dtype == torch.uint8
#         assert x.shape == (N,)
#         assert w.shape == (N, M)
#         assert rx.shape == mx.shape == (M,)
#         assert ry.shape == my.shape == (N, 1)
#         y = torch.zeros((M,), device=w.device, dtype=torch.float32)
#         torch.ops.rwkv.mm8_one(N, M, x, w, mx, rx, my, ry, y)
#         return y.to(dtype=x.dtype)
# else:
#     os.environ["RWKV_CUDA_ON"] = '0'


# @MyStatic
# def torch_mm8_seq(x, w, mx, rx, my, ry):
#     return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

# @MyStatic
# def torch_mm8_one(x, w, mx, rx, my, ry):
#     return x @ ((w.to(dtype=x.dtype) + 0.5) * ry * rx + my + mx)

# if os.environ.get('RWKV_CUDA_ON') == '1':
#     @MyStatic
#     def mm8_seq(x, w, mx, rx, my, ry):
#         if w.device.type == 'cuda' and x.dtype == torch.float16:
#             B, N, M = x.shape[0], w.shape[0], w.shape[1]
#             return cuda_mm8_seq(B, N, M, x, w, mx, rx, my, ry)
#         else:
#             return torch_mm8_seq(x, w, mx, rx, my, ry)
#     @MyStatic
#     def mm8_one(x, w, mx, rx, my, ry):
#         if w.device.type == 'cuda':
#             N, M = w.shape[0], w.shape[1]
#             return cuda_mm8_one(N, M, x, w, mx, rx, my, ry)
#         else:
#             return torch_mm8_one(x, w, mx, rx, my, ry)
# else:
#     @MyStatic
#     def mm8_seq(x, w, mx, rx, my, ry):
#         return torch_mm8_seq(x, w, mx, rx, my, ry)
#     @MyStatic
#     def mm8_one(x, w, mx, rx, my, ry):
#         return torch_mm8_one(x, w, mx, rx, my, ry)

# def mm8(x: torch.Tensor, w: torch.Tensor, mx: torch.Tensor, rx: torch.Tensor, my: torch.Tensor, ry: torch.Tensor):
#     if len(x.shape) == 1:
#         return mm8_one(x, w, mx, rx, my, ry)
#     return mm8_seq(x, w, mx, rx, my, ry)

# def matmul(a, b, mx: Optional[torch.Tensor]=None, rx: Optional[torch.Tensor]=None, my: Optional[torch.Tensor]=None, ry: Optional[torch.Tensor]=None, output_dtype: Optional[torch.dtype]=None) -> torch.Tensor:
#     if output_dtype is None:
#         output_dtype = a.dtype
#     if b.dtype in [torch.float16, torch.bfloat16, torch.float32]:
#         assert a.dtype == b.dtype
#         return matmul_float(a, b, output_dtype=output_dtype)
#     elif b.dtype == torch.uint8:
#         assert mx is not None
#         assert rx is not None
#         assert my is not None
#         assert ry is not None
#         return mm8(a, b, mx, rx, my, ry).to(output_dtype)
#     else:
#         raise ValueError("Unsupported dtype")


# if os.environ.get('RWKV_CUDA_ON') == '1' and not DISABLE_CUBLAS_GEMM:
#     def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
#         if output_dtype is None:
#             output_dtype = a.dtype
#         if a.dtype == b.dtype == torch.float16 and a.device.type == 'cuda':
#             if len(a.shape) == 1:
#                 assert len(b.shape) == 2
#                 c = torch.empty((b.shape[-1],), dtype=output_dtype, device=a.device)
#                 a = a.unsqueeze(0)
#             else:
#                 assert len(a.shape) == len(b.shape)
#                 assert len(a.shape) == 2 or len(a.shape) == 3
#                 # torch.empty((*a.shape[:-1], b.shape[-1])) doesn't work with jit
#                 if len(a.shape) == 2:
#                     c = torch.empty((a.shape[0], b.shape[-1]), dtype=output_dtype, device=a.device)
#                 else:
#                     c = torch.empty((a.shape[0], a.shape[1], b.shape[-1]), dtype=output_dtype, device=a.device)
#             torch.ops.rwkv.gemm_fp16_cublas(a, b, c)
#             return c
#         else:
#             return (a @ b).to(output_dtype)

# else:
#     def matmul_float(a, b, output_dtype: Optional[torch.dtype]=None):
#         return (a @ b).to(output_dtype)
# ######################################## Above are all impl for MatMul in several precisions ############################

# if os.environ.get('RWKV_DML_ON') == '1':
#     import torch_directml
#     print("PyTorch with DirectML Enabled")

# if os.environ.get('RWKV_V7_ON') == '1':

#     print(f'\n### RWKV-7 "Goose" enabled ###\n')

#     torch.backends.cudnn.benchmark = True
#     torch.backends.cudnn.allow_tf32 = True
#     torch.backends.cuda.matmul.allow_tf32 = True
#     # torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
#     # torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
#     torch._C._jit_set_autocast_mode(False)

#     MyModule = torch.jit.ScriptModule
#     MyFunction = torch.jit.script_method
#     MyStatic = torch.jit.script
#     from typing import List

#     DTYPE = None
#     DEVICE = None
#     HEAD_SIZE = 64

#     if os.environ.get('RWKV_CUDA_ON') == '1':
#         from torch.utils.cpp_extension import load
#         load(name="wkv7s", sources=[f"{current_path}/cuda/rwkv7_op.cpp", f"{current_path}/cuda/rwkv7.cu"], is_python_module=False,
#                             verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
#         class WKV_7(torch.autograd.Function):
#             @staticmethod
#             def forward(ctx, state, r, w, k, v, a, b):
#                 with torch.no_grad():
#                     T, C = r.size()
#                     H = C // HEAD_SIZE
#                     N = HEAD_SIZE
#                     assert HEAD_SIZE == C // H
#                     assert all(x.dtype == DTYPE for x in [r,w,k,v,a,b])
#                     assert all(x.is_contiguous() for x in [r,w,k,v,a,b])
#                     y = torch.empty((T, C), device=DEVICE, dtype=r.dtype, requires_grad=False, memory_format=torch.contiguous_format)

#                     if DTYPE == torch.float16:
#                         torch.ops.wkv7s.forward_fp16(1, T, C, H, state, r, w, k, v, a, b, y)
#                     elif DTYPE == torch.bfloat16:
#                         torch.ops.wkv7s.forward_bf16(1, T, C, H, state, r, w, k, v, a, b, y)
#                     elif DTYPE == torch.float32:
#                         torch.ops.wkv7s.forward_fp32(1, T, C, H, state, r, w, k, v, a, b, y)

#                     return y
#         def RWKV7_OP(state, r, w, k, v, a, b):
#             return WKV_7.apply(state, r, w, k, v, a, b)
    
#     ########################################################################################################

#     class RWKV_x070(MyModule):
#         def __init__(self, model, strategy):
#             global DTYPE, DEVICE
#             super().__init__()
#             self.eval()
#             args = types.SimpleNamespace()
#             self.args = args
#             args.MODEL_NAME = model

#             print(f'Loading {model} ({strategy})\n')

#             ss = strategy.split(' ')
#             DEVICE = ss[0]
#             if ss[1] == 'fp16':
#                 DTYPE = torch.half
#             elif ss[1] == 'fp32':
#                 DTYPE = torch.float32
#             elif ss[1] == 'bf16':
#                 DTYPE = torch.bfloat16
#             else:
#                 assert False, "currently rwkv7 strategy must be: cuda/cpu fp16/fp32/bf16"
            
#             temp_z = torch.load(args.MODEL_NAME + '.pth', map_location='cpu', mmap=True)

#             self.n_head, self.head_size = temp_z['blocks.0.att.r_k'].shape
#             args.head_size = self.head_size
#             args.vocab_size, args.n_embd = temp_z['emb.weight'].shape

#             args.n_layer = 0
#             keys = list(temp_z.keys())
#             self.z = {}

#             for k in keys:
#                 layer_id = int(k.split('.')[1]) if ('blocks.' in k) else 0
#                 args.n_layer = max(args.n_layer, layer_id+1)
#                 tensor = temp_z[k]
#                 if 'key.weight' in k or 'value.weight' in k or 'receptance.weight' in k or 'output.weight' in k or 'head.weight' in k:
#                     tensor = tensor.t().contiguous()
#                 tensor = tensor.squeeze()
#                 if k.endswith('att.r_k'): 
#                     tensor = tensor.flatten()
#                 self.z[k] = tensor.to(DEVICE).to(DTYPE)
#                 del temp_z[k]
#                 if keys.index(k) % 5 == 0:
#                     torch.cuda.empty_cache()

#             self.n_embd = args.n_embd
#             self.n_layer = args.n_layer

#             self.z['emb.weight'] = F.layer_norm(self.z['emb.weight'], (args.n_embd,), weight=self.z['blocks.0.ln0.weight'], bias=self.z['blocks.0.ln0.bias'])
#             self.z['blocks.0.att.v0'] = torch.empty(0, device=DEVICE, dtype=DTYPE) # actually ignored
#             self.z['blocks.0.att.v1'] = torch.empty(0, device=DEVICE, dtype=DTYPE) # actually ignored
#             self.z['blocks.0.att.v2'] = torch.empty(0, device=DEVICE, dtype=DTYPE) # actually ignored
#             torch.cuda.empty_cache()

#         def forward(self, idx, state, full_output=False):
#             if state == None:
#                 state = [None for _ in range(self.args.n_layer * 3)]
#                 for i in range(self.args.n_layer): # state: 0=att_x_prev 1=att_kv 2=ffn_x_prev
#                     state[i*3+0] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)
#                     state[i*3+1] = torch.zeros((self.args.n_embd // self.args.head_size, self.args.head_size, self.args.head_size), dtype=torch.float, requires_grad=False, device=DEVICE)
#                     state[i*3+2] = torch.zeros(self.args.n_embd, dtype=DTYPE, requires_grad=False, device=DEVICE)

#             if type(idx) is list:
#                 if len(idx) > 1:
#                     return self.forward_seq(idx, state, full_output)
#                 else:
#                     return self.forward_one(idx[0], state)
#             else:
#                 return self.forward_one(idx, state)

#         @MyFunction
#         def forward_one(self, idx:int, state:List[torch.Tensor]):
#             with torch.no_grad(): 
#                 z = self.z
#                 x = z['emb.weight'][idx]

#                 v_first = torch.empty_like(x)
#                 for i in range(self.n_layer):
#                     bbb = f'blocks.{i}.'
#                     att = f'blocks.{i}.att.'
#                     ffn = f'blocks.{i}.ffn.'

#                     xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

#                     xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_one(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
#                         z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
#                         z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
#                         z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
#                         z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
#                         z[att+'ln_x.weight'], z[att+'ln_x.bias'])
#                     x = x + xx

#                     xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

#                     xx, state[i*3+2] = RWKV_x070_CMix_one(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
#                     x = x + xx
                
#                     # if math.isnan(torch.min(x).item()): print(idx, i)

#                 x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
#                 x = x @ z['head.weight']
#                 return x, state
            
#         @MyFunction
#         def forward_seq(self, idx:List[int], state:List[torch.Tensor], full_output:bool=False):
#             with torch.no_grad(): 
#                 z = self.z
#                 x = z['emb.weight'][idx]

#                 v_first = torch.empty_like(x)
#                 for i in range(self.n_layer):
#                     bbb = f'blocks.{i}.'
#                     att = f'blocks.{i}.att.'
#                     ffn = f'blocks.{i}.ffn.'

#                     xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln1.weight'], bias=z[bbb+'ln1.bias'])

#                     xx, state[i*3+0], state[i*3+1], v_first = RWKV_x070_TMix_seq(i, self.n_head, self.head_size, xx, state[i*3+0], v_first, state[i*3+1],
#                         z[att+'x_r'], z[att+'x_w'], z[att+'x_k'], z[att+'x_v'], z[att+'x_a'], z[att+'x_g'],
#                         z[att+'w0'], z[att+'w1'], z[att+'w2'], z[att+'a0'], z[att+'a1'], z[att+'a2'], z[att+'v0'], z[att+'v1'], z[att+'v2'],
#                         z[att+'g1'], z[att+'g2'], z[att+'k_k'], z[att+'k_a'], z[att+'r_k'],
#                         z[att+'receptance.weight'], z[att+'key.weight'], z[att+'value.weight'], z[att+'output.weight'],
#                         z[att+'ln_x.weight'], z[att+'ln_x.bias'])
#                     x = x + xx

#                     xx = F.layer_norm(x, (self.n_embd,), weight=z[bbb+'ln2.weight'], bias=z[bbb+'ln2.bias'])

#                     xx, state[i*3+2] = RWKV_x070_CMix_seq(xx, state[i*3+2], z[ffn+'x_k'], z[ffn+'key.weight'], z[ffn+'value.weight'])
#                     x = x + xx
                
#                 if not full_output: x = x[-1,:]
#                 x = F.layer_norm(x, (self.n_embd,), weight=z['ln_out.weight'], bias=z['ln_out.bias'])
#                 x = x @ z['head.weight']
#                 return x, state

#     ########################################################################################################

#     @MyStatic
#     def RWKV_x070_TMix_one(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
#         xx = x_prev - x
#         xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

#         r = xr @ R_
#         w = torch.tanh(xw @ w1) @ w2
#         k = xk @ K_
#         v = xv @ V_
#         a = torch.sigmoid(a0 + (xa @ a1) @ a2)
#         g = torch.sigmoid(xg @ g1) @ g2

#         kk = torch.nn.functional.normalize((k * k_k).view(H,N), dim=-1, p=2.0).view(H*N)
#         k = k * (1 + (a-1) * k_a)
#         if layer_id == 0: v_first = v
#         else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)
#         w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)

#         vk = v.view(H,N,1) @ k.view(H,1,N)
#         ab = (-kk).view(H,N,1) @ (kk*a).view(H,1,N)
#         state = state * w.view(H,1,N) + state @ ab.float() + vk.float()
#         xx = (state.to(dtype=x.dtype) @ r.view(H,N,1))

#         xx = torch.nn.functional.group_norm(xx.view(1,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(H*N)    
#         xx = xx + ((r * k * r_k).view(H,N).sum(dim=-1, keepdim=True) * v.view(H,N)).view(H*N)
#         return (xx * g) @ O_, x, state, v_first

#     if os.environ.get('RWKV_CUDA_ON') == '1':
#         @MyStatic
#         def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
#             T = x.shape[0]
#             xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
#             xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

#             r = xr @ R_
#             w = torch.tanh(xw @ w1) @ w2
#             k = xk @ K_
#             v = xv @ V_
#             a = torch.sigmoid(a0 + (xa @ a1) @ a2)
#             g = torch.sigmoid(xg @ g1) @ g2

#             kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
#             k = k * (1 + (a-1) * k_a)
#             if layer_id == 0: v_first = v
#             else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

#             w = -torch.nn.functional.softplus(-(w0 + w)) - 0.5
#             xx = RWKV7_OP(state, r, w, k, v, -kk, kk*a)

#             xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
#             xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
#             return (xx * g) @ O_, x[-1,:], state, v_first
#     else:
#         @MyStatic
#         def RWKV_x070_TMix_seq(layer_id: int, H:int, N:int, x, x_prev, v_first, state, x_r, x_w, x_k, x_v, x_a, x_g, w0, w1, w2, a0, a1, a2, v0, v1, v2, g1, g2, k_k, k_a, r_k, R_, K_, V_, O_, ln_w, ln_b):
#             T = x.shape[0]
#             xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
#             xr, xw, xk, xv, xa, xg = x+xx*x_r, x+xx*x_w, x+xx*x_k, x+xx*x_v, x+xx*x_a, x+xx*x_g

#             r = xr @ R_
#             w = torch.tanh(xw @ w1) @ w2
#             k = xk @ K_
#             v = xv @ V_
#             a = torch.sigmoid(a0 + (xa @ a1) @ a2)
#             g = torch.sigmoid(xg @ g1) @ g2

#             kk = torch.nn.functional.normalize((k * k_k).view(T,H,N), dim=-1, p=2.0).view(T,H*N)
#             k = k * (1 + (a-1) * k_a)
#             if layer_id == 0: v_first = v
#             else: v = v + (v_first - v) * torch.sigmoid(v0 + (xv @ v1) @ v2)

#             w = torch.exp(-0.606531 * torch.sigmoid((w0 + w).float())) # 0.606531 = exp(-0.5)
#             for t in range(T):
#                 r_, w_, k_, v_, kk_, a_ = r[t], w[t], k[t], v[t], kk[t], a[t]
#                 vk = v_.view(H,N,1) @ k_.view(H,1,N)
#                 ab = (-kk_).view(H,N,1) @ (kk_*a_).view(H,1,N)
#                 state = state * w_.view(H,1,N) + state @ ab.float() + vk.float()
#                 xx[t] = (state.to(dtype=x.dtype) @ r_.view(H,N,1)).view(H*N)

#             xx = torch.nn.functional.group_norm(xx.view(T,H*N), num_groups=H, weight=ln_w, bias=ln_b, eps = 64e-5).view(T,H*N)
#             xx = xx + ((r * k * r_k).view(T,H,N).sum(dim=-1, keepdim=True) * v.view(T,H,N)).view(T,H*N)
#             return (xx * g) @ O_, x[-1,:], state, v_first

#     ########################################################################################################

#     @MyStatic
#     def RWKV_x070_CMix_one(x, x_prev, x_k, K_, V_):
#         xx = x_prev - x
#         k = x + xx * x_k
#         k = torch.relu(k @ K_) ** 2
#         return k @ V_, x

#     @MyStatic
#     def RWKV_x070_CMix_seq(x, x_prev, x_k, K_, V_):
#         xx = torch.cat((x_prev.unsqueeze(0), x[:-1,:])) - x
#         k = x + xx * x_k
#         k = torch.relu(k @ K_) ** 2
#         return k @ V_, x[-1,:]
        
# class RWKV(MyModule):
#     def __init__(self, model, strategy, verbose = True, convert_and_save_and_exit = None):
#         super().__init__()
#         if verbose:
#             prxxx = lambda *args, **kwargs: print(*args, **kwargs)
#         else:
#             prxxx = lambda *args, **kwargs: None

#         STRATEGY_REGEX = r"^(?:(?:^|->) *(?:cuda(?::[\d]+)?|cpu|mps|dml) (?:fp(?:16|32)|bf16)(?:i8|i4|i3)?(?: \*[\d]+\+?)? *)+$"
#         if not re.match(STRATEGY_REGEX, strategy):
#             raise ValueError("Invalid strategy. Please read https://pypi.org/project/rwkv/")

#         strategy = ('->'.join([x.strip() for x in strategy.split('->')])).replace('->', ' -> ')
#         self.args = types.SimpleNamespace()
#         args = self.args
#         args.MODEL_NAME = model
#         args.strategy_string = strategy

#         # Rescale for fp16 mode: set x = x/2 every X layer (to avoid fp16 overflow)
#         try:
#             self.RESCALE_LAYER = int(os.environ["RWKV_RESCALE_LAYER"]) # !!! NOTE: SEEMS YOU SHOULD SET IT TO 999 (disable) FOR RWKV-MUSIC MODELS !!!
#         except:
#             self.RESCALE_LAYER = 6 if 'fp16' in strategy else 0
#         prxxx(f'RWKV_JIT_ON {os.environ["RWKV_JIT_ON"]} RWKV_CUDA_ON {os.environ["RWKV_CUDA_ON"]} RESCALE_LAYER {self.RESCALE_LAYER}\n')

#         args.MODEL_NAME = args.MODEL_NAME.strip()
#         if not args.MODEL_NAME.endswith('.pth'):
#             args.MODEL_NAME += '.pth'
#         prxxx(f'Loading {args.MODEL_NAME} ...')
#         with torch.no_grad():
#             self.w = torch.load(args.MODEL_NAME, map_location='cpu') # load model to CPU first
#             gc.collect()
#             w = self.w

#             ALREADY_CONVERTED = False
#             if '_strategy' in w:
#                 ALREADY_CONVERTED = True
#                 assert convert_and_save_and_exit == None # you should only convert a raw model
#                 prxxx(f"Converted model: strategy {w['_strategy']}, version {w['_version']}\n")
#                 assert w['_strategy'] == args.strategy_string # if you are using a new strategy, re-convert the model
#                 assert float(w['_version']) >= 0.7 # sometimes you should re-convert using latest convert_model.py
#                 assert w['_rescale_layer'] == self.RESCALE_LAYER # must use same RESCALE_LAYER to avoid mistakes
#                 del w['_strategy']
#                 del w['_version']
#                 del w['_rescale_layer']
            
#             args.n_embd = w['emb.weight'].shape[1]
#             args.n_att = w['blocks.0.att.key.weight'].shape[0] # note: transposed matrix
#             args.n_ffn = w['blocks.0.ffn.key.weight'].shape[0] # note: transposed matrix
#             args.n_layer = 0
#             keys = list(w.keys())
#             self.version = 4
#             for x in keys:
#                 layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
#                 args.n_layer = max(args.n_layer, layer_id+1)
#                 if 'ln_x' in x:
#                     self.version = max(5, self.version)
#                 if 'gate.weight' in x:
#                     self.version = max(5.1, self.version)
#                 if int(self.version) == 5 and 'att.time_decay' in x:
#                     args.n_head = w[x].shape[0]
#                     if len(w[x].shape) > 1:
#                         if w[x].shape[1] > 1:
#                             self.version = max(5.2, self.version)
#                 if 'time_maa' in x:
#                     self.version = max(6, self.version)
#                 if int(self.version) == 6 and 'time_faaaa' in x:
#                     args.n_head = w[x].shape[0]
#             prxxx(f'Model detected: v{self.version:.1f}')

#             ####################### Compute strategy

#             s = [x.strip().split(' ') for x in strategy.split('->')]
#             plan = [0] * len(s)
#             stream_i = -1
#             stream_count = 0
#             to_allocate = args.n_layer + 1
#             allocated = 0
#             free_slots = 0
#             for i in range(len(s)):
#                 si = s[i]
#                 si1 = si[1]
#                 if si1.startswith('fp32'): si[1] = [torch.float]
#                 elif si1.startswith('fp16'): si[1] = [torch.float16]
#                 elif si1.startswith('bf16'): si[1] = [torch.bfloat16]
#                 if si1.endswith('i8'): si[1] += [torch.uint8]
#                 else: si[1] += [si[1][0]]
#                 if len(si) > 2:
#                     ss = si[2]
#                     assert ss.startswith('*')
#                     if ss.endswith('+'):
#                         plan[i] = int(ss[1:-1])
#                         stream_i = i
#                     else:
#                         plan[i] = int(ss[1:])
#                     allocated += plan[i]
#                     if allocated >= to_allocate:
#                         plan[i] += to_allocate - allocated
#                         break
#                 else:
#                     free_slots += 1
#             if stream_i < 0:
#                 if free_slots > 0 and to_allocate > allocated:
#                     for i in range(len(s)):
#                         if plan[i] == 0:
#                             plan[i] = (to_allocate - allocated) // free_slots
#                             allocated += plan[i]
#                             free_slots -= 1
#                 if to_allocate > allocated:
#                     plan[len(s)-1] += to_allocate - allocated
#             else:
#                 if to_allocate > allocated:
#                     stream_count = to_allocate - allocated
#                     plan[stream_i] += stream_count
#             prxxx(f'Strategy: (total {args.n_layer}+1={args.n_layer+1} layers)')
#             for i in range(len(s)):
#                 ss = s[i]
#                 if i != stream_i:
#                     prxxx(f'* {ss[0]} {str(ss[1]).replace("torch.","")}, store {plan[i]} layers')
#                 else:
#                     prxxx(f'* {ss[0]} {str(ss[1]).replace("torch.","")}, store {plan[i]-stream_count} layers, stream {stream_count} layers')
#                 plan[i] += (0 if i == 0 else plan[i-1])
#             self.strategy = [None] * (args.n_layer + 1)
#             strategy = self.strategy
#             for n in range(args.n_layer + 1):
#                 for i in range(len(s)):
#                     if n < plan[i]:
#                         strategy[n] = types.SimpleNamespace()
#                         strategy[n].device = s[i][0]
#                         strategy[n].atype = s[i][1][0]
#                         strategy[n].wtype = s[i][1][1]
#                         strategy[n].stream = False
#                         if strategy[n].device == 'dml':
#                             strategy[n].device = torch_directml.device()
#                         if i == stream_i and n >= (plan[i] - stream_count):
#                             strategy[n].stream = True
#                         break
#                 prxxx(f"{n}-{strategy[n].device}-{str(strategy[n].atype).replace('torch.','')}-{str(strategy[n].wtype).replace('torch.','')}{'-stream' if strategy[n].stream else ''}",end=' ')
#             prxxx()

#             ####################### Load weights to self.w

#             if not ALREADY_CONVERTED:
#                 try: # precompute embedding
#                     w['emb.weight'] = F.layer_norm(w['emb.weight'], (args.n_embd,), weight=w['blocks.0.ln0.weight'], bias=w['blocks.0.ln0.bias'])
#                 except:
#                     w['emb.weight'] = F.layer_norm(w['emb.weight'].float(), (args.n_embd,), weight=w['blocks.0.ln0.weight'].float(), bias=w['blocks.0.ln0.bias'].float())
#                 del w['blocks.0.ln0.weight']
#                 del w['blocks.0.ln0.bias']

#             print_need_newline = False

#             REAL_TIME_FIRST = False
#             args.time_state = False
#             for x in list(w.keys()):
#                 if '.time_faaaa' in x: REAL_TIME_FIRST = True
#                 if '.time_state' in x: args.time_state = True
#             if REAL_TIME_FIRST:
#                 w = {k.replace('.time_faaaa','.time_first') if '.time_faaaa' in k else k: v for k, v in w.items()}
#                 self.w = w
            
#             keys = list(w.keys())
#             for x in keys:
#                 w[x].requires_grad = False
#                 layer_id = int(x.split('.')[1]) if ('blocks.' in x) else 0
#                 if ('ln_out.' in x) or ('head.' in x):
#                     layer_id = args.n_layer
#                 dd = strategy[layer_id]
#                 DEVICE = dd.device
#                 ATYPE = dd.atype
#                 WTYPE = dd.wtype

#                 if not ALREADY_CONVERTED:
#                     if self.RESCALE_LAYER > 0:
#                         if 'att.output.weight' in x:
#                             w[x] = w[x] / (2 ** int(layer_id // self.RESCALE_LAYER))
#                         if 'ffn.value.weight' in x:
#                             w[x] = w[x] / (2 ** int(layer_id // self.RESCALE_LAYER))

#                     if '.time_' in x:
#                         w[x] = w[x].squeeze()
#                     if 'key.weight' in x or 'value.weight' in x or 'receptance.weight' in x or 'gate.weight' in x or 'output.weight' in x or 'head.weight' in x:
#                         w[x] = w[x].t()

#                     if '.time_decay' in x and '_w' not in x: # need fp32 for this
#                         if self.version == 4:
#                             w[x] = -torch.exp(w[x].float())
#                         elif int(self.version) == 5:
#                             w[x] = torch.exp(-torch.exp(w[x].float())).reshape(-1,1,1)
#                             if self.version == 5.2:
#                                 w[x] = w[x].reshape(args.n_head, -1, 1)
#                         elif self.version == 6.0:
#                             w[x] = w[x].float().reshape(args.n_head, -1, 1)
#                     elif '.time_first' in x: # need fp32 for this
#                         if self.version == 4:
#                             w[x] = w[x].float()
#                         elif int(self.version) in [5, 6]:
#                             if REAL_TIME_FIRST:
#                                 w[x] = w[x].float().reshape(-1,1,1)
#                             else:
#                                 w[x] = torch.exp(w[x].float()).reshape(-1,1,1)
#                             if self.version in [5.2, 6.0]:
#                                 w[x] = w[x].reshape(args.n_head, -1, 1)
#                     elif '.ln_x' in x: # need fp32 for group_norm
#                         w[x] = w[x].float()
#                     else:
#                         if (len(w[x].shape) == 2) and ('emb' not in x) and ('_w1' not in x) and ('_w2' not in x):
#                             if WTYPE != torch.uint8:
#                                 w[x] = w[x].to(dtype=WTYPE)
#                             else:
#                                 w[x] = w[x].float()

#                                 if w[x].shape[0] > w[x].shape[1]:
#                                     w[x+'_my'] = torch.amin(w[x], dim=1).unsqueeze(1)
#                                     w[x] = w[x] - w[x+'_my']
#                                     w[x+'_mx'] = torch.amin(w[x], dim=0)
#                                     w[x] = w[x] - w[x+'_mx']
#                                     w[x+'_rx'] = torch.amax(w[x], dim=0)
#                                     w[x] = w[x] / w[x+'_rx']
#                                     w[x+'_ry'] = torch.amax(w[x], dim=1).unsqueeze(1)
#                                     w[x] = w[x] / w[x+'_ry']
#                                 else:
#                                     w[x+'_mx'] = torch.amin(w[x], dim=0)
#                                     w[x] = w[x] - w[x+'_mx']
#                                     w[x+'_my'] = torch.amin(w[x], dim=1).unsqueeze(1)
#                                     w[x] = w[x] - w[x+'_my']
#                                     w[x+'_rx'] = torch.amax(w[x], dim=0)
#                                     w[x] = w[x] / w[x+'_rx']
#                                     w[x+'_ry'] = torch.amax(w[x], dim=1).unsqueeze(1)
#                                     w[x] = w[x] / w[x+'_ry']

#                                 w[x] = torch.clip(torch.floor(w[x] * 256), min=0, max=255).to(dtype=torch.uint8)
#                                 w[x+'_mx'] = w[x+'_mx'].to(dtype=ATYPE).contiguous()
#                                 w[x+'_rx'] = (w[x+'_rx'] / 16).to(dtype=ATYPE).contiguous()
#                                 w[x+'_my'] = w[x+'_my'].to(dtype=ATYPE).contiguous()
#                                 w[x+'_ry'] = (w[x+'_ry'] / 16).to(dtype=ATYPE).contiguous()
#                         else:
#                             w[x] = w[x].to(dtype=ATYPE)
                
#                 if convert_and_save_and_exit == None:
#                     if 'emb.' in x:
#                         w[x] = w[x].contiguous()
#                     elif (dd.stream) and (x.endswith('key.weight') or x.endswith('value.weight') or x.endswith('receptance.weight') or x.endswith('output.weight')):
#                         try:
#                             w[x] = w[x].contiguous().pin_memory() # if you see "CUDA error: out of memory" here, that's out of CPU RAM, not VRAM. Get more RAM :)
#                         except:
#                             print('Note: You are running out of RAM. Get more CPU RAM. Now this will run much slower.')
#                     elif DEVICE != 'cpu':
#                         w[x] = w[x].to(device=DEVICE).contiguous()
                    
#                     if (dd.stream) or (DEVICE != 'cpu'):
#                         try:
#                             w[x+'_mx'] = w[x+'_mx'].to(device=DEVICE).contiguous()
#                             w[x+'_rx'] = w[x+'_rx'].to(device=DEVICE).contiguous()
#                             w[x+'_my'] = w[x+'_my'].to(device=DEVICE).contiguous()
#                             w[x+'_ry'] = w[x+'_ry'].to(device=DEVICE).contiguous()
#                         except:
#                             pass

#                 if 'ffn.value.weight' in x:
#                     gc.collect()
#                     if 'cuda' in args.strategy_string:
#                         torch.cuda.empty_cache()

#                 shape = [i for i in w[x].shape if i != 1]
#                 if len(shape) > 2:
#                     shape = f" {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)} {str(shape[2]).rjust(5)}"
#                 elif len(shape) > 1:
#                     shape = f" {str(shape[0]).rjust(5)} {str(shape[1]).rjust(5)}      "
#                 else:
#                     shape = f" {str(shape[0]).rjust(5)}            "
#                 if layer_id == 0 or layer_id >= args.n_layer-1:
#                     if print_need_newline:
#                         prxxx('\n', end = '')
#                         print_need_newline = False
#                     dt = str(w[x].dtype).replace('torch.', '')
#                     dt = dt.replace('float32', 'f32').replace('bfloat16', 'bf16').replace('float16', 'f16').replace('uint8', 'i8')
#                     prxxx(x.ljust(32), dt.rjust(4), str(w[x].device).rjust(8), shape, ' (pinned)' if w[x].is_pinned() else '')
#                 else:
#                     print_need_newline = True
#                     prxxx('.', end = '', flush = True)
            
#             if convert_and_save_and_exit:
#                 w['_strategy'] = args.strategy_string
#                 w['_rescale_layer'] = self.RESCALE_LAYER
#                 w['_version'] = '0.7'
#                 if not convert_and_save_and_exit.endswith('.pth'):
#                     convert_and_save_and_exit += '.pth'
#                 prxxx(f'Saving to {convert_and_save_and_exit}...')
#                 torch.save(w, convert_and_save_and_exit)
#                 prxxx(f'Converted and saved. Now this will exit.')
#                 exit(0)
            
#             if self.version == 5.2 and os.environ["RWKV_CUDA_ON"] == '1':
#                 HEAD_SIZE = args.n_att // args.n_head
#                 rwkv5 = load(name="rwkv5", sources=[f"{current_path}/cuda/rwkv5_op.cpp", f"{current_path}/cuda/rwkv5.cu"],
#                                 verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3" if os.name != "nt" else "", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])

#                 class RWKV_5(torch.autograd.Function):
#                     @staticmethod
#                     def forward(ctx, B, T, C, H, state, r, k, v, w, u):
#                         with torch.no_grad():
#                             assert HEAD_SIZE == C // H
#                             ctx.B = B
#                             ctx.T = T
#                             ctx.C = C
#                             ctx.H = H
#                             assert state.dtype == torch.float32
#                             assert w.dtype == torch.float32
#                             assert r.is_contiguous()
#                             assert k.is_contiguous()
#                             assert v.is_contiguous()
#                             assert w.is_contiguous()                            
#                             assert u.is_contiguous()                            
#                             assert state.is_contiguous()

#                             y = torch.empty((B, T, C), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format)
#                             if r.dtype == torch.bfloat16:
#                                 rwkv5.forward_bf16(B, T, C, H, state, r, k, v, w, u, y)
#                             elif r.dtype == torch.float16:
#                                 rwkv5.forward_fp16(B, T, C, H, state, r, k, v, w, u, y)
#                             elif r.dtype == torch.float32:
#                                 rwkv5.forward_fp32(B, T, C, H, state, r, k, v, w, u, y)
#                             return y, state
#                 self.RWKV_5 = RWKV_5

#             if self.version == 6.0 and os.environ["RWKV_CUDA_ON"] == '1':
#                 HEAD_SIZE = args.n_att // args.n_head
#                 rwkv6 = load(name="rwkv6", sources=[f"{current_path}/cuda/rwkv6_op.cpp", f"{current_path}/cuda/rwkv6.cu"],
#                                 verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3" if os.name != "nt" else "", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={4096}"])
                    
#                 class RWKV_6(torch.autograd.Function):
#                     @staticmethod
#                     def forward(ctx, B, T, C, H, state, r, k, v, w, u):
#                         with torch.no_grad():
#                             assert HEAD_SIZE == C // H
#                             ctx.B = B
#                             ctx.T = T
#                             ctx.C = C
#                             ctx.H = H
#                             assert state.dtype == torch.float32
#                             assert w.dtype == torch.float32
#                             assert r.is_contiguous()
#                             assert k.is_contiguous()
#                             assert v.is_contiguous()
#                             assert w.is_contiguous()
#                             assert u.is_contiguous()
#                             eew = torch.exp(-torch.exp(w.float())).contiguous()

#                             y = torch.empty((B, T, C), device=w.device, dtype=r.dtype, memory_format=torch.contiguous_format)
#                             if r.dtype == torch.bfloat16:
#                                 rwkv6.forward_bf16(B, T, C, H, state, r, k, v, eew, u, y)
#                             elif r.dtype == torch.float16:
#                                 rwkv6.forward_fp16(B, T, C, H, state, r, k, v, eew, u, y)
#                             elif r.dtype == torch.float32:
#                                 rwkv6.forward_fp32(B, T, C, H, state, r, k, v, eew, u, y)
#                             return y, state
#                 self.RWKV_6 = RWKV_6
        
#             gc.collect()
#             if 'cuda' in args.strategy_string:
#                 torch.cuda.empty_cache()

#     def RUN_RWKV_5(self, B, T, C, H, state, r, k, v, w, u):
#         return self.RWKV_5.apply(B, T, C, H, state, r, k, v, w, u)

#     def RUN_RWKV_6(self, B, T, C, H, state, r, k, v, w, u):
#         return self.RWKV_6.apply(B, T, C, H, state, r, k, v, w, u)

#     ########################################################################################################

#     @MyFunction
#     def ffn_one(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         kx = xx * k_mix + sx * (1 - k_mix)
#         rx = xx * r_mix + sx * (1 - r_mix)

#         r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
#         vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
#         out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
#         return x + out, xx

#     @MyFunction
#     def ffn_seq(self, x, sx, ln_w, ln_b, k_mix, r_mix, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
#         kx = xx * k_mix + sx * (1 - k_mix)
#         rx = xx * r_mix + sx * (1 - r_mix)

#         r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
#         vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
#         out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
#         return x + out, xx[-1,:]

#     @MyFunction
#     def ffn_one_v6(self, x, sx, ln_w, ln_b, k_maa, r_maa, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         sx = sx - xx
#         kx = xx + sx * k_maa
#         rx = xx + sx * r_maa

#         r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
#         vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
#         out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
#         return x + out, xx

#     @MyFunction
#     def ffn_seq_v6(self, x, sx, ln_w, ln_b, k_maa, r_maa, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
#         sx = sx - xx
#         kx = xx + sx * k_maa
#         rx = xx + sx * r_maa

#         r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
#         vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
#         out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
#         return x + out, xx[-1,:]

#     ########################################################################################################

#     @MyFunction
#     def att_one(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         kx = xx * k_mix + sx * (1 - k_mix)
#         vx = xx * v_mix + sx * (1 - v_mix)
#         rx = xx * r_mix + sx * (1 - r_mix)

#         r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
#         k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
#         v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

#         ww = t_first + k
#         p = torch.maximum(pp, ww)
#         e1 = torch.exp(pp - p)
#         e2 = torch.exp(ww - p)
#         wkv = ((e1 * aa + e2 * v) / (e1 * bb + e2)).to(dtype=x.dtype)
#         ww = t_decay + pp
#         p = torch.maximum(ww, k)
#         e1 = torch.exp(ww - p)
#         e2 = torch.exp(k - p)

#         out = matmul(r * wkv, ow, omx, orx, omy, ory)
#         return x + out, xx, e1 * aa + e2 * v, e1 * bb + e2, p

#     @MyFunction
#     def att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
#         kx = xx * k_mix + sx * (1 - k_mix)
#         vx = xx * v_mix + sx * (1 - v_mix)
#         rx = xx * r_mix + sx * (1 - r_mix)

#         r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
#         k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
#         v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)

#         T = x.shape[0]
#         for t in range(T):
#             kk = k[t]
#             vv = v[t]
#             ww = t_first + kk
#             p = torch.maximum(pp, ww)
#             e1 = torch.exp(pp - p)
#             e2 = torch.exp(ww - p)
#             sx[t] = ((e1 * aa + e2 * vv) / (e1 * bb + e2)).to(dtype=x.dtype)
#             ww = t_decay + pp
#             p = torch.maximum(ww, kk)
#             e1 = torch.exp(ww - p)
#             e2 = torch.exp(kk - p)
#             aa = e1 * aa + e2 * vv
#             bb = e1 * bb + e2
#             pp = p
#         out = matmul(r * sx, ow, omx, orx, omy, ory)
#         return x + out, xx[-1,:], aa, bb, pp

#     ########################################################################################################

#     @MyFunction
#     def att_one_v5(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         kx = xx * k_mix + sx * (1 - k_mix)
#         vx = xx * v_mix + sx * (1 - v_mix)
#         rx = xx * r_mix + sx * (1 - r_mix)

#         H = t_decay.shape[0]
#         N = x.shape[-1] // H

#         r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
#         k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
#         v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
        
#         a = matmul(k, v)
#         out = r @ (t_first * a + s)
#         s = a + t_decay * s

#         out = out.flatten()
#         out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
#         out = out.to(dtype=x.dtype)
#         out = matmul(out, ow, omx, orx, omy, ory)

#         return x + out, xx, s

#     @MyFunction
#     def att_seq_v5(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
#         kx = xx * k_mix + sx * (1 - k_mix)
#         vx = xx * v_mix + sx * (1 - v_mix)
#         rx = xx * r_mix + sx * (1 - r_mix)

#         H = t_decay.shape[0]
#         N = x.shape[-1] // H
#         T = x.shape[0]

#         w = t_decay.reshape(-1, 1)
#         u = t_first.reshape(-1, 1)
#         ws = w.pow(T).reshape(H, 1, 1)
#         ind = torch.arange(T-1, -1, -1, device=w.device).unsqueeze(0).repeat(H, 1)
#         w = w.repeat(1, T).pow(ind)
#         wk = w.reshape(H, 1, T)
#         wb = wk.transpose(-2, -1).flip(1)
#         w = torch.cat([w[:, 1:], u], dim=1)
#         w = F.pad(w, (0, T))
#         w = torch.tile(w, [T])
#         w = w[:, :-T].reshape(-1, T, 2 * T - 1)
#         w = w[:, :, T-1:].reshape(H, T, T)

#         r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
#         k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
#         v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)

#         out = ((r @ k) * w) @ v + (r @ s) * wb
#         s = ws * s + (k * wk) @ v
        
#         out = out.transpose(0, 1).contiguous().reshape(T, H*N)
#         out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
#         out = out.to(dtype=x.dtype)
#         out = matmul(out, ow, omx, orx, omy, ory)

#         return x + out, xx[-1,:], s

#     ########################################################################################################

#     @MyFunction
#     def att_one_v5_1(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         kx = xx * k_mix + sx * (1 - k_mix)
#         vx = xx * v_mix + sx * (1 - v_mix)
#         rx = xx * r_mix + sx * (1 - r_mix)
#         gx = xx * g_mix + sx * (1 - g_mix)

#         H = t_decay.shape[0]
#         N = x.shape[-1] // H

#         r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
#         k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
#         v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
#         g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))
        
#         a = matmul(k, v)
#         out = r @ (t_first * a + s)
#         s = a + t_decay * s

#         out = out.flatten()
#         out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
#         out = out.to(dtype=x.dtype) * g
#         out = matmul(out, ow, omx, orx, omy, ory)

#         return x + out, xx, s

#     @MyFunction
#     def att_seq_v5_1(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
#         kx = xx * k_mix + sx * (1 - k_mix)
#         vx = xx * v_mix + sx * (1 - v_mix)
#         rx = xx * r_mix + sx * (1 - r_mix)
#         gx = xx * g_mix + sx * (1 - g_mix)

#         H = t_decay.shape[0]
#         N = x.shape[-1] // H
#         T = x.shape[0]

#         w = t_decay.reshape(-1, 1)
#         u = t_first.reshape(-1, 1)
#         ws = w.pow(T).reshape(H, 1, 1)
#         ind = torch.arange(T-1, -1, -1, device=w.device).unsqueeze(0).repeat(H, 1)
#         w = w.repeat(1, T).pow(ind)
#         wk = w.reshape(H, 1, T)
#         wb = wk.transpose(-2, -1).flip(1)
#         w = torch.cat([w[:, 1:], u], dim=1)
#         w = F.pad(w, (0, T))
#         w = torch.tile(w, [T])
#         w = w[:, :-T].reshape(-1, T, 2 * T - 1)
#         w = w[:, :, T-1:].reshape(H, T, T)

#         r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
#         k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
#         v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
#         g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

#         out = ((r @ k) * w) @ v + (r @ s) * wb
#         s = ws * s + (k * wk) @ v
        
#         out = out.transpose(0, 1).contiguous().reshape(T, H*N)
#         out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
#         out = out.to(dtype=x.dtype) * g
#         out = matmul(out, ow, omx, orx, omy, ory)

#         return x + out, xx[-1,:], s

#     ########################################################################################################

#     @MyFunction
#     def att_seq_v5_2(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
#         kx = xx * k_mix + sx * (1 - k_mix)
#         vx = xx * v_mix + sx * (1 - v_mix)
#         rx = xx * r_mix + sx * (1 - r_mix)
#         gx = xx * g_mix + sx * (1 - g_mix)

#         H = t_decay.shape[0]
#         N = x.shape[-1] // H
#         T = x.shape[0]

#         r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
#         k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
#         v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
#         g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

#         out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
#         for t in range(T):
#             rt = r[:,t:t+1,:]
#             kt = k[:,:,t:t+1]
#             vt = v[:,t:t+1,:]
#             at = matmul(kt, vt)
#             out[t] = (rt @ (t_first * at + s)).squeeze(1)
#             s = at + t_decay * s

#         out = out.reshape(T, H*N)
#         out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
#         out = out.to(dtype=x.dtype) * g
#         out = matmul(out, ow, omx, orx, omy, ory)

#         return x + out, xx[-1,:], s

#     ########################################################################################################

#     @MyFunction
#     def att_one_v6_0(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
        
#         sx = sx - xx
#         xxx = xx + sx * x_maa
#         xxx = torch.tanh(xxx @ tm_w1).view(5, 1, -1)
#         xxx = torch.bmm(xxx, tm_w2).view(5, -1)
#         mw, mk, mv, mr, mg = xxx.unbind(dim=0)

#         wx = xx + sx * (w_maa + mw)
#         kx = xx + sx * (k_maa + mk)
#         vx = xx + sx * (v_maa + mv)
#         rx = xx + sx * (r_maa + mr)
#         gx = xx + sx * (g_maa + mg)

#         H = t_decay.shape[0]
#         N = x.shape[-1] // H

#         r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(H, 1, N)
#         k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(H, N, 1)
#         v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(H, 1, N)
#         g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))
        
#         w = t_decay + (torch.tanh(wx @ td_w1) @ td_w2).float().view(H, N, 1)
#         w = torch.exp(-torch.exp(w.float()))

#         a = matmul(k, v)
#         out = r @ (t_first * a + s)
#         s = a + w * s

#         out = out.flatten()
#         out = F.group_norm(out.unsqueeze(0), num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5).squeeze(0)
#         out = out.to(dtype=x.dtype) * g
#         out = matmul(out, ow, omx, orx, omy, ory)

#         return x + out, xx, s

#     @MyFunction
#     def att_seq_v6_0(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
#         H = t_decay.shape[0]
#         N = x.shape[-1] // H
#         T = x.shape[0]

#         xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#         sx = torch.cat((sx.unsqueeze(0), xx[:-1,:])) - xx
#         xxx = xx + sx * x_maa
#         xxx = torch.tanh(xxx @ tm_w1).view(T, 5, -1).transpose(0, 1)
#         xxx = torch.bmm(xxx, tm_w2).view(5, T, -1)
#         mw, mk, mv, mr, mg = xxx.unbind(dim=0)

#         wx = xx + sx * (w_maa + mw)
#         kx = xx + sx * (k_maa + mk)
#         vx = xx + sx * (v_maa + mv)
#         rx = xx + sx * (r_maa + mr)
#         gx = xx + sx * (g_maa + mg)

#         r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
#         k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32).view(T, H, N).permute(1, 2, 0)
#         v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32).view(T, H, N).transpose(0, 1)
#         g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

#         w = t_decay.view(1, H, N, 1) + (torch.tanh(wx @ td_w1) @ td_w2).float().view(T, H, N, 1)
#         w = torch.exp(-torch.exp(w.float()))
#         out = torch.empty((T, H, N), dtype=r.dtype, device=r.device)
#         for t in range(T):
#             rt = r[:,t:t+1,:]
#             kt = k[:,:,t:t+1]
#             vt = v[:,t:t+1,:]
#             at = matmul(kt, vt)
#             out[t] = (rt @ (t_first * at + s)).squeeze(1)
#             s = at + w[t] * s

#         out = out.reshape(T, H*N)
#         out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
#         out = out.to(dtype=x.dtype) * g
#         out = matmul(out, ow, omx, orx, omy, ory)

#         return x + out, xx[-1,:], s

#     ########################################################################################################

#     if os.environ["RWKV_CUDA_ON"] == '1':
#         @MyFunction
#         def cuda_att_seq(self, x, sx, aa, bb, pp, ln_w, ln_b, k_mix, v_mix, r_mix, t_decay, t_first, kw, vw, rw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, omx, orx, omy, ory):
#             T, C = x.shape
#             xx = F.layer_norm(x, (C,), weight=ln_w, bias=ln_b)
#             sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
#             kx = xx * k_mix + sx * (1 - k_mix)
#             vx = xx * v_mix + sx * (1 - v_mix)
#             rx = xx * r_mix + sx * (1 - r_mix)

#             r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
#             k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
#             v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
#             y, aa, bb, pp = cuda_wkv(T, C, t_decay, t_first, k, v, aa, bb, pp)

#             out = matmul(r * y.to(x.dtype), ow, omx, orx, omy, ory)
#             return x + out, xx[-1,:], aa, bb, pp

#         @MyFunction
#         def v5_2_before(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
#             xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#             sx = torch.cat((sx.unsqueeze(0), xx[:-1,:]))
#             kx = xx * k_mix + sx * (1 - k_mix)
#             vx = xx * v_mix + sx * (1 - v_mix)
#             rx = xx * r_mix + sx * (1 - r_mix)
#             gx = xx * g_mix + sx * (1 - g_mix)

#             r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
#             k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
#             v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
#             g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

#             return r, k, v, g, xx[-1,:], s.transpose(-1,-2).contiguous()

#         @MyFunction
#         def v5_2_after(self, t_decay, out, s, x, xxx, g, lx_w, lx_b, ow, omx, orx, omy, ory):
#             H = t_decay.shape[0]
#             N = x.shape[-1] // H
#             T = x.shape[0]

#             s = s.transpose(-1,-2)
#             out = out.reshape(T, H*N)
#             out = F.group_norm(out, num_groups=H, weight=lx_w, bias=lx_b, eps = 64e-5)
#             out = out.to(dtype=x.dtype) * g
#             out = matmul(out, ow, omx, orx, omy, ory)

#             return x + out, xxx, s

#         def cuda_att_seq_v5_2(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
#             H = t_decay.shape[0]
#             N = x.shape[-1] // H
#             T = x.shape[0]

#             r, k, v, g, xxx, ss = self.v5_2_before(x, sx, s, ln_w, ln_b, lx_w, lx_b, k_mix, v_mix, r_mix, g_mix, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory)

#             out, s = self.RUN_RWKV_5(1, T, self.args.n_att, H, ss, r, k, v, w=t_decay, u=t_first)

#             return self.v5_2_after(t_decay, out, s, x, xxx, g, lx_w, lx_b, ow, omx, orx, omy, ory)

#         @MyFunction
#         def v6_0_before(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
#             H = t_decay.shape[0]
#             N = x.shape[-1] // H
#             T = x.shape[0]

#             xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
#             sx = torch.cat((sx.unsqueeze(0), xx[:-1,:])) - xx
#             xxx = xx + sx * x_maa
#             xxx = torch.tanh(xxx @ tm_w1).view(T, 5, -1).transpose(0, 1)
#             xxx = torch.bmm(xxx, tm_w2).view(5, T, -1)
#             mw, mk, mv, mr, mg = xxx.unbind(dim=0)

#             wx = xx + sx * (w_maa + mw)
#             kx = xx + sx * (k_maa + mk)
#             vx = xx + sx * (v_maa + mv)
#             rx = xx + sx * (r_maa + mr)
#             gx = xx + sx * (g_maa + mg)

#             r = matmul(rx, rw, rmx, rrx, rmy, rry, output_dtype=torch.float32)
#             k = matmul(kx, kw, kmx, krx, kmy, kry, output_dtype=torch.float32)
#             v = matmul(vx, vw, vmx, vrx, vmy, vry, output_dtype=torch.float32)
#             g = F.silu(matmul(gx, gw, gmx, grx, gmy, gry))

#             w = t_decay.view(1, H, N, 1) + (torch.tanh(wx @ td_w1) @ td_w2).float().view(T, H, N, 1)

#             return r, k, v, g, w, xx[-1,:], s.transpose(-1,-2).contiguous()

#         def cuda_att_seq_v6_0(self, x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory):
#             H = t_decay.shape[0]
#             N = x.shape[-1] // H
#             T = x.shape[0]

#             r, k, v, g, w, xxx, ss = self.v6_0_before(x, sx, s, ln_w, ln_b, lx_w, lx_b, x_maa, w_maa, k_maa, v_maa, r_maa, g_maa, tm_w1, tm_w2, td_w1, td_w2, t_decay, t_first, kw, vw, rw, gw, ow, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry, gmx, grx, gmy, gry, omx, orx, omy, ory)

#             out, s = self.RUN_RWKV_6(1, T, self.args.n_att, H, ss, r, k, v, w=w, u=t_first)
#             return self.v5_2_after(t_decay, out, s, x, xxx, g, lx_w, lx_b, ow, omx, orx, omy, ory)

#     ########################################################################################################

#     def forward(self, tokens, state, full_output=False):
#         with torch.no_grad():
#             w = self.w
#             args = self.args

#             if state == None:
#                 if self.version == 4:
#                     state = [None] * args.n_layer * 5
#                     for i in range(args.n_layer): # state: 0=att_xx 1=att_aa 2=att_bb 3=att_pp 4=ffn_xx
#                         dd = self.strategy[i]
#                         dev = dd.device
#                         atype = dd.atype
#                         state[i*5+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
#                         state[i*5+1] = torch.zeros(args.n_att, dtype=torch.float, requires_grad=False, device=dev).contiguous()
#                         state[i*5+2] = torch.zeros(args.n_att, dtype=torch.float, requires_grad=False, device=dev).contiguous()
#                         state[i*5+3] = torch.zeros(args.n_att, dtype=torch.float, requires_grad=False, device=dev).contiguous() - 1e30
#                         state[i*5+4] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
#                 elif int(self.version) in [5,6]:
#                     state = [None] * args.n_layer * 3
#                     for i in range(args.n_layer): # state: 0=att_xx 1=att_kv 2=ffn_xx
#                         dd = self.strategy[i]
#                         dev = dd.device
#                         atype = dd.atype
#                         state[i*3+0] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()
#                         if args.time_state:
#                             state[i*3+1] = w[f'blocks.{i}.att.time_state'].transpose(1,2).to(dtype=torch.float, device=dev).requires_grad_(False).contiguous()
#                         else:
#                             state[i*3+1] = torch.zeros((args.n_head, args.n_att//args.n_head, args.n_att//args.n_head), dtype=torch.float, requires_grad=False, device=dev).contiguous()
#                         state[i*3+2] = torch.zeros(args.n_embd, dtype=atype, requires_grad=False, device=dev).contiguous()

#             seq_mode = len(tokens) > 1

#             x = w['emb.weight'][tokens if seq_mode else tokens[0]]

#             for i in range(args.n_layer):
#                 bbb = f'blocks.{i}.'
#                 att = f'blocks.{i}.att.'
#                 ffn = f'blocks.{i}.ffn.'
#                 dd = self.strategy[i]
#                 dev = dd.device
#                 atype = dd.atype
#                 wtype = dd.wtype
#                 if seq_mode:
#                     cuda_applicable = os.environ["RWKV_CUDA_ON"] == '1' and 'cuda' in str(dev)
#                     if cuda_applicable:
#                         ATT = self.cuda_att_seq
#                     else:
#                         ATT = self.att_seq
#                     if self.version == 5:
#                         ATT = self.att_seq_v5
#                     elif self.version == 5.1:
#                         ATT = self.att_seq_v5_1
#                     elif self.version == 5.2:
#                         ATT = self.att_seq_v5_2
#                         if cuda_applicable:
#                             ATT = self.cuda_att_seq_v5_2
#                     elif self.version == 6.0:
#                         ATT = self.att_seq_v6_0
#                         if cuda_applicable:
#                             ATT = self.cuda_att_seq_v6_0
#                     FFN = self.ffn_seq
#                     if self.version >= 6.0:
#                         FFN = self.ffn_seq_v6
#                 else:
#                     ATT = self.att_one
#                     if self.version == 5:
#                         ATT = self.att_one_v5
#                     elif self.version == 5.1:
#                         ATT = self.att_one_v5_1
#                     elif self.version == 5.2:
#                         ATT = self.att_one_v5_1 # same as v5.1
#                     elif self.version == 6.0:
#                         ATT = self.att_one_v6_0
#                     FFN = self.ffn_one
#                     if self.version >= 6.0:
#                         FFN = self.ffn_one_v6

#                 x = x.to(dtype=atype, device=dev)

#                 kw = w[f'{att}key.weight']
#                 vw = w[f'{att}value.weight']
#                 rw = w[f'{att}receptance.weight']
#                 ow = w[f'{att}output.weight']
#                 if dd.stream:
#                     kw = kw.to(device=dev, non_blocking=True)
#                     vw = vw.to(device=dev, non_blocking=True)
#                     rw = rw.to(device=dev, non_blocking=True)
#                     ow = ow.to(device=dev, non_blocking=True)
#                 kmx = w[f'{att}key.weight_mx'] if wtype == torch.uint8 else x
#                 krx = w[f'{att}key.weight_rx'] if wtype == torch.uint8 else x
#                 kmy = w[f'{att}key.weight_my'] if wtype == torch.uint8 else x
#                 kry = w[f'{att}key.weight_ry'] if wtype == torch.uint8 else x
#                 vmx = w[f'{att}value.weight_mx'] if wtype == torch.uint8 else x
#                 vrx = w[f'{att}value.weight_rx'] if wtype == torch.uint8 else x
#                 vmy = w[f'{att}value.weight_my'] if wtype == torch.uint8 else x
#                 vry = w[f'{att}value.weight_ry'] if wtype == torch.uint8 else x
#                 rmx = w[f'{att}receptance.weight_mx'] if wtype == torch.uint8 else x
#                 rrx = w[f'{att}receptance.weight_rx'] if wtype == torch.uint8 else x
#                 rmy = w[f'{att}receptance.weight_my'] if wtype == torch.uint8 else x
#                 rry = w[f'{att}receptance.weight_ry'] if wtype == torch.uint8 else x
#                 omx = w[f'{att}output.weight_mx'] if wtype == torch.uint8 else x
#                 orx = w[f'{att}output.weight_rx'] if wtype == torch.uint8 else x
#                 omy = w[f'{att}output.weight_my'] if wtype == torch.uint8 else x
#                 ory = w[f'{att}output.weight_ry'] if wtype == torch.uint8 else x
#                 if self.version in [5.1, 5.2, 6.0]:
#                     gw = w[f'{att}gate.weight']
#                     if dd.stream:
#                         gw = gw.to(device=dev, non_blocking=True)
#                     gmx = w[f'{att}gate.weight_mx'] if wtype == torch.uint8 else x
#                     grx = w[f'{att}gate.weight_rx'] if wtype == torch.uint8 else x
#                     gmy = w[f'{att}gate.weight_my'] if wtype == torch.uint8 else x
#                     gry = w[f'{att}gate.weight_ry'] if wtype == torch.uint8 else x
#                 if self.version == 4:
#                     x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3] = ATT(
#                         x, state[i*5+0], state[i*5+1], state[i*5+2], state[i*5+3],
#                         w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
#                         w[f'{att}time_mix_k'], w[f'{att}time_mix_v'], w[f'{att}time_mix_r'],
#                         w[f'{att}time_decay'], w[f'{att}time_first'],
#                         kw, vw, rw, ow,
#                         kmx, krx, kmy, kry,
#                         vmx, vrx, vmy, vry,
#                         rmx, rrx, rmy, rry,
#                         omx, orx, omy, ory,
#                         )
#                 elif self.version == 5:
#                     x, state[i*3+0], state[i*3+1] = ATT(
#                         x, state[i*3+0], state[i*3+1],
#                         w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
#                         w[f'{att}ln_x.weight'], w[f'{att}ln_x.bias'],
#                         w[f'{att}time_mix_k'], w[f'{att}time_mix_v'], w[f'{att}time_mix_r'],
#                         w[f'{att}time_decay'], w[f'{att}time_first'],
#                         kw, vw, rw, ow,
#                         kmx, krx, kmy, kry,
#                         vmx, vrx, vmy, vry,
#                         rmx, rrx, rmy, rry,
#                         omx, orx, omy, ory,
#                         )
#                 elif self.version in [5.1, 5.2]:
#                     x, state[i*3+0], state[i*3+1] = ATT(
#                         x, state[i*3+0], state[i*3+1],
#                         w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
#                         w[f'{att}ln_x.weight'], w[f'{att}ln_x.bias'],
#                         w[f'{att}time_mix_k'], w[f'{att}time_mix_v'], w[f'{att}time_mix_r'], w[f'{att}time_mix_g'],
#                         w[f'{att}time_decay'], w[f'{att}time_first'],
#                         kw, vw, rw, gw, ow,
#                         kmx, krx, kmy, kry,
#                         vmx, vrx, vmy, vry,
#                         rmx, rrx, rmy, rry,
#                         gmx, grx, gmy, gry,
#                         omx, orx, omy, ory,
#                         )
#                 elif self.version == 6.0:
#                     x, state[i*3+0], state[i*3+1] = ATT(
#                         x, state[i*3+0], state[i*3+1],
#                         w[f'{bbb}ln1.weight'], w[f'{bbb}ln1.bias'],
#                         w[f'{att}ln_x.weight'], w[f'{att}ln_x.bias'],
#                         w[f'{att}time_maa_x'], w[f'{att}time_maa_w'], w[f'{att}time_maa_k'], w[f'{att}time_maa_v'], w[f'{att}time_maa_r'], w[f'{att}time_maa_g'],
#                         w[f'{att}time_maa_w1'], w[f'{att}time_maa_w2'], w[f'{att}time_decay_w1'], w[f'{att}time_decay_w2'],
#                         w[f'{att}time_decay'], w[f'{att}time_first'],
#                         kw, vw, rw, gw, ow,
#                         kmx, krx, kmy, kry,
#                         vmx, vrx, vmy, vry,
#                         rmx, rrx, rmy, rry,
#                         gmx, grx, gmy, gry,
#                         omx, orx, omy, ory,
#                         )
#                 if dd.stream:
#                     del kw, vw, rw, ow
#                     if self.version in [5.1, 5.2, 6.0]:
#                         del gw

#                 kw = w[f'{ffn}key.weight']
#                 vw = w[f'{ffn}value.weight']
#                 rw = w[f'{ffn}receptance.weight']
#                 if dd.stream:
#                     kw = kw.to(device=dev, non_blocking=True)
#                     vw = vw.to(device=dev, non_blocking=True)
#                     rw = rw.to(device=dev, non_blocking=True)
#                 kmx = w[f'{ffn}key.weight_mx'] if wtype == torch.uint8 else x
#                 krx = w[f'{ffn}key.weight_rx'] if wtype == torch.uint8 else x
#                 kmy = w[f'{ffn}key.weight_my'] if wtype == torch.uint8 else x
#                 kry = w[f'{ffn}key.weight_ry'] if wtype == torch.uint8 else x
#                 vmx = w[f'{ffn}value.weight_mx'] if wtype == torch.uint8 else x
#                 vrx = w[f'{ffn}value.weight_rx'] if wtype == torch.uint8 else x
#                 vmy = w[f'{ffn}value.weight_my'] if wtype == torch.uint8 else x
#                 vry = w[f'{ffn}value.weight_ry'] if wtype == torch.uint8 else x
#                 rmx = w[f'{ffn}receptance.weight_mx'] if wtype == torch.uint8 else x
#                 rrx = w[f'{ffn}receptance.weight_rx'] if wtype == torch.uint8 else x
#                 rmy = w[f'{ffn}receptance.weight_my'] if wtype == torch.uint8 else x
#                 rry = w[f'{ffn}receptance.weight_ry'] if wtype == torch.uint8 else x
#                 if self.version == 4:
#                     offset = i*5+4
#                 elif int(self.version) in [5,6]:
#                     offset = i*3+2
#                 if self.version < 6.0:
#                     x, state[offset] = FFN(
#                         x, state[offset],
#                         w[f'{bbb}ln2.weight'], w[f'{bbb}ln2.bias'],
#                         w[f'{ffn}time_mix_k'], w[f'{ffn}time_mix_r'],
#                         kw, vw, rw,
#                         kmx, krx, kmy, kry,
#                         vmx, vrx, vmy, vry,
#                         rmx, rrx, rmy, rry,                    
#                         )
#                 else:
#                     x, state[offset] = FFN(
#                         x, state[offset],
#                         w[f'{bbb}ln2.weight'], w[f'{bbb}ln2.bias'],
#                         w[f'{ffn}time_maa_k'], w[f'{ffn}time_maa_r'],
#                         kw, vw, rw,
#                         kmx, krx, kmy, kry,
#                         vmx, vrx, vmy, vry,
#                         rmx, rrx, rmy, rry,                    
#                         )
#                 if dd.stream:                
#                     del kw, vw, rw
                
#                 if self.RESCALE_LAYER > 0:
#                     if (i+1) % self.RESCALE_LAYER == 0:
#                         x = x / 2
            
#             dd = self.strategy[args.n_layer]
#             x = x[-1,:] if (seq_mode and (not full_output)) else x
#             x = x.to(dtype=dd.atype, device=dd.device)
            
#             x = F.layer_norm(x, (args.n_embd,), weight=w['ln_out.weight'], bias=w['ln_out.bias'])
#             if w['head.weight'].dtype != torch.uint8:
#                 x = x @ w['head.weight']
#             else:
#                 if seq_mode and full_output:
#                     x = mm8_seq(x, w['head.weight'], w['head.weight_mx'], w['head.weight_rx'], w['head.weight_my'], w['head.weight_ry'])
#                 else:
#                     x = mm8_one(x, w['head.weight'], w['head.weight_mx'], w['head.weight_rx'], w['head.weight_my'], w['head.weight_ry'])

#             return x.float(), state