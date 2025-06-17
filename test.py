########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging

logging.basicConfig(level=logging.INFO)

########################################################################################################
# RWKV Tokenizer (slow version)
########################################################################################################

class RWKV_TOKENIZER():
    table: list[list[list[bytes]]]
    good: list[set[int]]
    wlen: list[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> list[int]:
        src_len: int = len(src)
        tokens: list[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()


if __name__ == "__main__":
    from argparse import ArgumentParser

    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.utilities import rank_zero_info, rank_zero_only

    rank_zero_info("########## work in progress ##########")

    parser = ArgumentParser()

    parser.add_argument("--load_model", default="",
                        type=str)  # full path, with .pth
    # wandb project name. if "" then don't use wandb
    parser.add_argument("--wandb", default="", type=str)
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)

    parser.add_argument("--data_file", default="", type=str)
    parser.add_argument("--data_type", default="utf-8", type=str)
    # vocab_size = 0 means auto (for char-level LM and .txt data)
    parser.add_argument("--vocab_size", default=0, type=int)

    parser.add_argument("--ctx_len", default=1024, type=int)
    # a mini "epoch" has [epoch_steps] steps
    parser.add_argument("--epoch_steps", default=1000, type=int)
    # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument("--epoch_count", default=500, type=int)
    # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_begin", default=0, type=int)
    # save the model every [epoch_save] "epochs"
    parser.add_argument("--epoch_save", default=5, type=int)

    # micro batch size (batch size per GPU)
    parser.add_argument("--micro_bsz", default=12, type=int)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)

    # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    parser.add_argument("--lr_init", default=6e-4, type=float)
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1,
                        type=int)  # try 10 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)
    parser.add_argument("--adam_eps", default=1e-18, type=float)
    # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--grad_cp", default=0, type=int)
    parser.add_argument("--weight_decay", default=0, type=float)  # try 0.1
    # reduce it to 0.7 / 0.5 / 0.3 / 0.2 for problematic samples
    parser.add_argument("--grad_clip", default=1.0, type=float)

    parser.add_argument("--train_stage", default=0,
                        type=int)  # my special pile mode
    # deepspeed bucket size in MB. 200 seems enough
    parser.add_argument("--ds_bucket_mb", default=200, type=int)

    # can try larger values for larger models
    parser.add_argument("--head_size", default=64, type=int)
    parser.add_argument("--load_partial", default=0, type=int)
    parser.add_argument("--magic_prime", default=0, type=int)
    parser.add_argument("--my_testing", default='x070', type=str)
    parser.add_argument("--my_exit_tokens", default=0, type=int)

    ###################### new #######################
    # parser.add_argument("--num_hidden_layers", default=128, type=int)
    parser.add_argument("--num_hidden_groups", default=16, type=int)
    parser.add_argument("--inner_group_num", default=2, type=int)
    ##################################################

    ##################### training args #####################
    parser.add_argument("--mean_recurrence", default=10, type=int) # mean recurrence steps per sample
    parser.add_argument("--mean_backprop_depth", default=5, type=int) # how many blocks to backprop
    parser.add_argument("--sampling_scheme", default='bptt', type=str) # how to sample recurrence steps
    parser.add_argument("--lockstep_n", default=False, type=bool) # 
    parser.add_argument("--lockstep_k", default=False, type=bool) # 
    parser.add_argument("--rand_step", default=0.0, type=int) # for convenience to change seed
    
    parser.add_argument("--mcleish_throttle", default=False, type=bool) # loss norm with n_grads steps
    parser.add_argument("--elbayad_weighing", default=True, type=bool) # weighted loss
    parser.add_argument("--elbayad_exponent", default=0.5, type=float) # weighted loss, with what power should future steps be penalized
    #########################################################
    parser.add_argument("--injection_type", default="linear", type=str)

    #####################testing args##########################
    parser.add_argument("--adaptive_loop_enabled", default=True, type=bool)
    parser.add_argument("--min_repeat", default=1, type=int)
    parser.add_argument("--max_repeat", default=12, type=int)
    parser.add_argument("--repeat_prob", default=0.4, type=bool)
    # parser.add_argument("--injection_type", default=True, type=bool)
    parser.add_argument("--early_exit_enabled", default=True, type=bool)
    parser.add_argument("--confidence_threshold", default=0.3, type=float)
    parser.add_argument("--stability_threshold", default=1e-3, type=float)
    parser.add_argument("--stability_check_layers", default=3, type=int)
    parser.add_argument("--max_compute_steps", default=50, type=int)
    


    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    ########################################################################################################

    import datetime
    import math
    import os
    import sys
    import time
    import warnings

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    if "deepspeed" in args.strategy:
        import deepspeed
    from pytorch_lightning import seed_everything

    if args.random_seed >= 0:
        print(
            f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings(
        "ignore", ".*Consider increasing the value of the `num_workers` argument*")
    warnings.filterwarnings(
        "ignore", ".*The progress bar already tracks a metric with the*")
    # os.environ["WDS_SHOW_SEED"] = "1"
    torch._C._jit_set_autocast_mode(False)
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method
    MyStatic = torch.jit.script

    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    args.gradient_clip_val = args.grad_clip
    args.num_sanity_val_steps = 0
    args.check_val_every_n_epoch = int(1e20)
    args.log_every_n_steps = int(1e20)
    args.max_epochs = -1  # continue forever
    args.betas = (args.beta1, args.beta2)
    args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 *
                           32)  # default = 3.5x emb size

    # args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
    # if not os.path.exists(args.proj_dir):
    #     os.makedirs(args.proj_dir)

    # args.epoch_count = args.magic_prime // 40320
    # args.epoch_steps = 40320 // args.real_bsz
    # assert args.epoch_steps * args.real_bsz == 40320

    # if args.train_stage >= 2:  # find latest saved model
    #     list_p = []
    #     for p in os.listdir(args.proj_dir):
    #         if p.startswith("rwkv") and p.endswith(".pth"):
    #             p = ((p.split("-"))[1].split("."))[0]
    #             if p != "final":
    #                 if p == "init":
    #                     p = -1
    #                 else:
    #                     p = int(p)
    #                 list_p += [p]
    #     list_p.sort()
    #     max_p = list_p[-1]
    #     if len(list_p) > 1:
    #         args.my_pile_prev_p = list_p[-2]  # in case max_p is corrupted
    #     if max_p == -1:
    #         args.load_model = f"{args.proj_dir}/rwkv-init.pth"
    #     else:
    #         args.load_model = f"{args.proj_dir}/rwkv-{max_p}.pth"
    #         if args.warmup_steps < 0:
    #             args.warmup_steps = 10
    #     args.epoch_begin = max_p + 1

#     samples_per_epoch = args.epoch_steps * args.real_bsz
#     tokens_per_epoch = samples_per_epoch * args.ctx_len
#     try:
#         deepspeed_version = deepspeed.__version__
#     except:
#         deepspeed_version = None
#         pass
#     rank_zero_info(
#         f"""
# ############################################################################
# #
# # RWKV-7 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.micro_bsz}={args.real_bsz}, {args.strategy} {'with grad_cp' if args.grad_cp > 0 else ''}
# #
# # Data = {args.data_file} ({args.data_type}), ProjDir = {args.proj_dir}
# #
# # Epoch = {args.epoch_begin} to {args.epoch_begin + args.epoch_count - 1} (will continue afterwards), save every {args.epoch_save} epoch
# #
# # Each "epoch" = {args.epoch_steps} steps, {samples_per_epoch} samples, {tokens_per_epoch} tokens
# #
# # Model = {args.n_layer} n_layer, {args.n_embd} n_embd, {args.ctx_len} ctx_len
# #
# # Adam = lr {args.lr_init} to {args.lr_final}, warmup {args.warmup_steps} steps, beta {args.betas}, eps {args.adam_eps}
# #
# # Found torch {torch.__version__}, recommend latest torch
# # Found deepspeed {deepspeed_version}, recommend latest deepspeed
# # Found pytorch_lightning {pl.__version__}, recommend 1.9.5
# #
# ############################################################################
# """
#     )
#     rank_zero_info(str(vars(args)) + "\n")

#     assert args.data_type in ["binidx"]

#     if args.lr_final == 0 or args.lr_init == 0:
#         rank_zero_info(
#             "\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n")

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info(
                "\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n")
    if args.precision == "fp16":
        rank_zero_info(
            "\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n")

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"  # somehow incompatible

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################

    # from src.dataset import MyDataset
    # from src.trainer import generate_init_weight, train_callback

    # train_data = MyDataset(args)
    # args.vocab_size = train_data.vocab_size
    
    tokenizer = RWKV_TOKENIZER("rwkv_vocab_v20230424.txt")  

    from src.model import RWKV, RWKV_shared, RWKV_x070_infer, RWKV_x070_infer_v2
    from torch.nn import functional as F
    # model = RWKV_shared(args)
    MODEL_PATH = "/data/projects/RWKV-LM-V7-Depth-recur/out/L32-D2048-x070/rwkv-52.pth"
    # DTYPE = torch.bfloat16
    DTYPE = torch.half # better 
    NUM_TRIALS = 3
    LENGTH_PER_TRIAL = 100
    TEMPERATURE = 1.0
    TOP_P = 0.0

    from torch.utils.cpp_extension import load
    HEAD_SIZE = args.head_size

    model_params = torch.load(MODEL_PATH, map_location="cpu")

    with torch.no_grad():
        model = RWKV_x070_infer_v2(args)
        model.load_state_dict(model_params, strict=False) # ignore blocks.0.att.v0/v1/v2
        prompt = "DNA stands for deoxyribonucleic"
        input = tokenizer.encode(prompt)
        print(f'\nInput:\n{input}')
        init_out, init_state = model.forward(tokenizer.encode(prompt), None)
        probs = F.softmax(init_out.float(), dim=-1) # compute softmax in float (more accurate)
        print(f'\n{prompt}')
        # print(f'\nProbabilities:\n{probs}')
        print(f'\nProbabilities shape:\n{probs.shape}')

        _, indices = torch.topk(probs, 10) # print top-10 possibilities
        print(f'\nIndices:\n{indices}')
        print(f'\nIndices shape:\n{indices.shape}')

        for i in range(len(indices)):
            token_id = indices[i].item()
            token = tokenizer.decode([token_id])
            token_prob = probs[token_id].item()
            print(token, f'[probability {token_prob:.2%}]')
    