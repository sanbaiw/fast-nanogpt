# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/pt2-learner.ipynb.

# %% auto 0
__all__ = ['default_device', 'DataSet', 'DataLoaders', 'CancelFitException', 'CancelBatchException', 'CancelEpochException',
           'Callback', 'run_cbs', 'with_cbs', 'Learner', 'to_device', 'DeviceCB', 'TrainCB', 'to_cpu', 'MetricsCB',
           'ProgressCB']

# %% ../nbs/pt2-learner.ipynb 1
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from typing import Mapping
from copy import copy

# %% ../nbs/pt2-learner.ipynb 2
class DataSet:
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

# %% ../nbs/pt2-learner.ipynb 6
class DataLoaders:
    def __init__(self, *dls):
        self.train, self.valid = dls[:2]
    
    @classmethod
    def from_dd(cls, datasets, batch_size, **kwargs):
        return cls(*[DataLoader(ds, batch_size=batch_size, **kwargs) for ds in datasets])

# %% ../nbs/pt2-learner.ipynb 11
from operator import attrgetter
from functools import partial

# %% ../nbs/pt2-learner.ipynb 12
class CancelFitException(Exception): pass
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass

# %% ../nbs/pt2-learner.ipynb 13
class Callback:
    order = 0


def run_cbs(cbs, method_nm, learn=None):
    for cb in sorted(cbs, key=attrgetter("order")):
        method = getattr(cb, method_nm, None)
        if method is not None:
            method(learn)

# %% ../nbs/pt2-learner.ipynb 14
class with_cbs:
    def __init__(self, nm):
        self.nm = nm

    def __call__(self, f):
        def _f(o, *args, **kwargs):
            try:
                o.callback(f"before_{self.nm}")
                f(o, *args, **kwargs)
                o.callback(f"after_{self.nm}")
            except globals()[f"Cancel{self.nm.title()}Exception"]:
                pass
            finally:
                o.callback(f"cleanup_{self.nm}")

        return _f


class Learner:
    def __init__(
        self,
        model,
        dls=(0,),
        lr=0.1,
        cbs=None,
        opt_func=optim.SGD,
    ):
        self.model = model
        self.dls = dls
        self.lr = lr
        self.cbs = cbs if cbs else []
        self.opt_func = opt_func

    @with_cbs("batch")
    def _one_batch(self):
        self.predict()
        self.callback("after_predict")
        # self.get_loss()
        # self.callback("after_loss")
        if self.training:
            self.backward()
            self.callback("after_backward")
            self.step()
            self.callback("after_step")
            self.zero_grad()

    @with_cbs("epoch")
    def _one_epoch(self):
        for self.iter, self.batch in enumerate(self.dl):
            self._one_batch()

    def one_epoch(self, training=True):
        self.model.train(training)
        self.dl = self.dls.train if training else self.dls.valid
        self._one_epoch()

    @with_cbs("fit")
    def _fit(self, train, valid):
        for self.epoch in self.epochs:
            if train:
                self.one_epoch(training=True)
            if valid:
                with torch.no_grad():
                    self.one_epoch(False)

    def fit(self, n_epochs=1, train=True, valid=True, cbs=None, lr=None):
        if cbs is None:
            cbs = []
        for cb in cbs:
            self.cbs.append(cb)
        try:
            self.n_epochs = n_epochs
            self.epochs = range(n_epochs)
            if lr is None:
                lr = self.lr
            if self.opt_func:
                self.opt = self.opt_func(self.model.parameters(), lr)
            self._fit(train, valid)
        finally:
            for cb in cbs:
                self.cbs.remove(cb)

    def __getattr__(self, name):
        if name in ("predict", "get_loss", "backward", "step", "zero_grad"):
            return partial(self.callback, name)
        raise AttributeError(name)

    def callback(self, method_nm):
        run_cbs(self.cbs, method_nm, self)

    @property
    def training(self):
        return self.model.training

# %% ../nbs/pt2-learner.ipynb 15
default_device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)


def to_device(x, device=default_device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, Mapping):
        return {k: v.to(device) for k, v in x.items()}
    return type(x)(to_device(o, device) for o in x)

# %% ../nbs/pt2-learner.ipynb 16
class DeviceCB(Callback):
    """Put model to device at the beginning of training, and put batch to device before each forward pass."""

    def __init__(self, device=default_device):
        self.device = device

    def before_fit(self, learn):
        if hasattr(learn.model, "to"):
            learn.model.to(self.device)

    def before_batch(self, learn):
        learn.batch = to_device(learn.batch, device=self.device)

# %% ../nbs/pt2-learner.ipynb 17
class TrainCB(Callback):

    def predict(self, learn):
        # import pdb; pdb.set_trace()
        learn.preds, learn.loss = learn.model(*learn.batch)
        # print("epoch", learn.epoch, "step", learn.iter, "loss", learn.loss.item())

    def backward(self, learn):
        learn.loss.backward()

    def step(self, learn):
        learn.opt.step()

    def zero_grad(self, learn):
        learn.opt.zero_grad()

# %% ../nbs/pt2-learner.ipynb 24
from torcheval.metrics import Mean

# %% ../nbs/pt2-learner.ipynb 25
def to_cpu(x):
    if isinstance(x, Mapping):
        return {k: to_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_cpu(o) for o in x]
    if isinstance(x, tuple):
        return tuple(to_cpu(list(x)))
    res = x.detach().cpu()
    return res.float() if res.dtype == torch.float16 else res


class MetricsCB(Callback):
    def __init__(self, *ms, **metrics):
        for o in ms:
            metrics[type(o).__name__] = o
        self.metrics = metrics
        self.all_metrics = copy(metrics)
        self.all_metrics["loss"] = self.loss = Mean()

    def _log(self, d):
        print(d)

    def before_fit(self, learn):
        learn.metrics = self

    def before_epoch(self, learn):
        [o.reset() for o in self.all_metrics.values()]

    def after_epoch(self, learn):
        log = {k: f"{v.compute():.3f}" for k, v in self.all_metrics.items()}
        # log["epoch"] = f"{learn.epoch}"
        log["epoch"] = learn.epoch
        log["train"] = "train" if learn.model.training else "eval"
        self._log(log)

    def after_batch(self, learn):
        x, y, *_ = to_cpu(learn.batch)
        for m in self.metrics.values():
            m.update(to_cpu(learn.preds), y)
        self.loss.update(to_cpu(learn.loss), weight=len(x))

# %% ../nbs/pt2-learner.ipynb 26
from fastprogress import progress_bar, master_bar
import fastcore.all as fc

# %% ../nbs/pt2-learner.ipynb 27
class ProgressCB(Callback):
    order = MetricsCB.order + 1

    def __init__(self, plot=False):
        self.plot = plot

    def before_fit(self, learn):
        learn.epochs = self.mbar = master_bar(learn.epochs)
        self.first = True
        if hasattr(learn, "metrics"):
            learn.metrics._log = self._log
        self.losses = []
        self.val_losses = []

    def _log(self, d):
        if self.first:
            self.mbar.write(list(d), table=True)
            self.first = False
        # import pdb; pdb.set_trace()
        self.mbar.write(list(d.values()), table=True)

    def before_epoch(self, learn):
        learn.dl = progress_bar(learn.dl, leave=False, parent=self.mbar)

    def after_batch(self, learn):
        learn.dl.comment = f"{learn.loss:.3f}"
        if self.plot and hasattr(learn, "metrics") and learn.training:
            self.losses.append(learn.loss.item())
            if self.val_losses:
                graphs = [
                    [fc.L.range(self.losses), self.losses],
                    [
                        fc.L.range(learn.epoch).map(
                            lambda x: (x + 1) * len(learn.dls.train)
                        ),
                        self.val_losses,
                    ],
                ]
            else:
                graphs = [[fc.L.range(self.losses), self.losses]]
            self.mbar.update_graph(graphs)

    def after_epoch(self, learn):
        if self.plot and hasattr(learn, "metrics"):
            if not learn.training:
                self.val_losses.append(learn.metrics.all_metrics["loss"].compute())
                graphs = [
                    [fc.L.range(self.losses), self.losses],
                    [
                        fc.L.range(learn.epoch + 1).map(
                            lambda x: (x + 1) * len(learn.dls.train)
                        ),
                        self.val_losses,
                    ],
                ]
            else:
                graphs = [[fc.L.range(self.losses), self.losses]]
            self.mbar.update_graph(graphs)
