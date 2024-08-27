from tinyai.model import *
from tinyai.learner import *
from tinyai.hooks import *
from tinyai.init import *
from tinyai.speedup import *
from tinyai.hyperparam import *
from tinyai.distributed import *
from tinyai.fineweb import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from functools import partial
import numpy as np
import multiprocessing as mp
from datasets import load_dataset

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
    if master_process:
        print(f"rank: {ddp_rank}, local rank: {ddp_local_rank}, world size: {ddp_world_size}")
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


@to_tensor
def get_tokens(input_file):
    with open(input_file) as f:
        text = f.read()
    tokens = gpt2_encoder.encode(text)
    return tokens


os.environ["DATA_DIR"] = "./edu_fineweb10B"
os.environ["CACHE_DIR"] = "./cache"

# load huggingface dataset
data_dir = os.getenv("DATA_DIR", None) or "./edu_fineweb10B"
cache_dir = os.getenv("CACHE_DIR", None)  # default to ~/.cache/huggingface/datasets
remote_name = "sample-10BT"
shard_size = int(1e8)
fw = load_dataset(
    "HuggingFaceFW/fineweb-edu", name=remote_name, split="train", cache_dir=cache_dir
)

# convert to shards
nproc = max(1, mp.cpu_count() // 2)
try:
    with TokenShardWriter(shard_size=shard_size, data_dir=data_dir) as wrt:
        with mp.Pool(nproc) as pool:
            for tokens in pool.imap(to_tokens_np, fw, chunksize=16):
                wrt.write(tokens)
except DataDirNotEmptyError:
    if master_process:
        print("Data directory is not empty, skipping creating shards")
    pass

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

# create data loader
tds = FSDataSet(
    data_dir,
    pattern="train",
    token_fn=to_tensor(load_np_tokens),
    T=T,
    num_proc=ddp_world_size,
    rank=ddp_rank,
)
vds = FSDataSet(
    data_dir,
    pattern="val",
    token_fn=to_tensor(load_np_tokens),
    T=T,
    num_proc=ddp_world_size,
    rank=ddp_rank,
)
dls = DataLoaders.from_dd([tds, vds], batch_size=B, drop_last=True)
# tdl = dls.train
# x, y = next(iter(tdl))
# print(x.shape, y.shape)
# os._exit(0)


# create model
model = get_model().to(device)
model = DDP(model, device_ids=[ddp_local_rank])

# create learner
cbs = [
    DDPGradAccuTrainCB(accu_steps=accu_steps),
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
