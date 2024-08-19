from tinyai.model import *
from tinyai.learner import *
from tinyai.hooks import *
from tinyai.init import *
from tinyai.speedup import *
from tinyai.hyperparam import *
from tinyai.distributed import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from functools import partial

import tiktoken
import os

from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

if ddp_enabled:
    init_process_group(backend="nccl")
    ddp_rank = dist.get_rank()
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = dist.get_world_size()
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = default_device

set_seed(1337)
torch.cuda.manual_seed(1337)


# transform the text into tokens
def to_tensor(f):
    def _f(*args, **kwargs):
        return torch.tensor(f(*args, **kwargs), dtype=torch.long)

    return _f


enc = tiktoken.get_encoding("gpt2")


@to_tensor
def get_tokens(input_file):
    with open(input_file) as f:
        text = f.read()
    tokens = enc.encode(text)
    return tokens


# create data loader
cwd = os.getcwd()
data_dir = f"{cwd}"
pattern = "input.txt"


total_batch_size = 2**19  # 524288, 0.5M
B = 8  # bach size (micro batch size)
T = 1024  # sequence length
assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "make sure total_batch_size is divisible by B * T * ddp_world_size"
accu_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {accu_steps}")
    print(
        f"=> rank: {ddp_rank}, local rank: {ddp_local_rank}, world size: {ddp_world_size}"
    )

tds = FSDataSet(
    data_dir,
    pattern=pattern,
    token_fn=get_tokens,
    T=T,
    num_proc=ddp_world_size,
    rank=ddp_rank,
)
dls = DataLoaders.from_dd([tds, None], batch_size=B, drop_last=True)

# create model
model = get_model().to(device)
model = DDP(model, device_ids=[ddp_local_rank])

# create learner
cbs = [
    DDPGradAccuTrainCB(accu_steps=accu_steps),
    FixedStepCallback(step_count=50),
    InitWeightsCB(),
    DDPCB(ddp_local_rank),
    DeviceCB(),
]

# record = GradAccuRecordCB(lr=_lr, grad_norm=_grad_norm)
schd = GradAccuScheduleCB(partial(CosineLR, warmup_steps=10, max_steps=50))


def fit(model, epochs=1, opt_func=get_optimizer, xtra_cbs=None, lr=3e-4):
    lrn = Learner(
        model, dls=dls, opt_func=opt_func, cbs=cbs + xtra_cbs if xtra_cbs else [], lr=lr
    )
    lrn.fit(epochs, valid=False)
    return lrn


# start training
fit(
    model,
    opt_func=get_optimizer,
    xtra_cbs=[schd, GradAccuLogCallback(), FixedStepCallback(step_count=5)],
    lr=6e-4,
)
