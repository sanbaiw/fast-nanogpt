# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/pt4-initialization.ipynb.

# %% auto 0
__all__ = ['clean_ipython_hist', 'clean_tb', 'clean_mem', 'is_residual', 'init_weights', 'InitCallback']

# %% ../nbs/pt4-initialization.ipynb 1
import random, math, torch, numpy as np, matplotlib.pyplot as plt
from .learner import *
from .model import *
from .hooks import *
import fastcore.all as fc
from functools import partial

# %% ../nbs/pt4-initialization.ipynb 6
def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals(): return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc): user_ns.pop('_i'+repr(n),None)
    user_ns.update(dict(_i='',_ii='',_iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 =  ''

# %% ../nbs/pt4-initialization.ipynb 7
def clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'): delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'): delattr(sys, 'last_value')

# %% ../nbs/pt4-initialization.ipynb 8
def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()

# %% ../nbs/pt4-initialization.ipynb 14
def is_residual(m):
    return isinstance(m, ResidualLinear)

# %% ../nbs/pt4-initialization.ipynb 16
def init_weights(m):
    std=0.02
    if isinstance(m, nn.Linear):
        if is_residual(m):
            std *= (2 * GPTConfig.n_layer) ** -0.5
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=std)

# %% ../nbs/pt4-initialization.ipynb 28
class InitCallback(Callback):
    def before_fit(self, learn):
        model = learn.model
        model.apply(init_weights)
