import sys
import importlib
# Patch importlib.import_module to disable C-based upb module which fails on Python 3.14
original_import_module = importlib.import_module
def custom_import_module(name, package=None):
    if name == 'google._upb._message':
        raise ImportError("google._upb._message is disabled on Python 3.14")
    return original_import_module(name, package)
importlib.import_module = custom_import_module

# Patch argparse.ArgumentParser.add_argument to support custom help objects on Python 3.14
import argparse
original_add_argument = argparse.ArgumentParser.add_argument
def patched_add_argument(self, *args, **kwargs):
    help_val = kwargs.get('help')
    if help_val is not None and not isinstance(help_val, str):
        cls = help_val.__class__
        if not hasattr(cls, '__contains__'):
            cls.__contains__ = lambda self, key: False
    return original_add_argument(self, *args, **kwargs)
argparse.ArgumentParser.add_argument = patched_add_argument

# Patch torch.load to default to weights_only=False to support legacy checkpoints containing OmegaConf metadata in PyTorch 2.6+
import torch
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Patch torch.Tensor.cuda to fallback to CPU when CUDA is not available
if not torch.cuda.is_available():
    torch.Tensor.cuda = lambda self, *args, **kwargs: self

# Patch PyTorch Geometric's Data.keys to be a hybrid property/method to maintain compatibility with both older and newer codebases
import torch_geometric.data
class KeysWrapper(list):
    def __call__(self):
        return self
original_keys_method = torch_geometric.data.Data.keys
torch_geometric.data.Data.keys = property(lambda self: KeysWrapper(original_keys_method(self)))

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from torch_points3d.trainer import Trainer


@hydra.main(config_path="conf", config_name="eval")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
    if getattr(cfg, "pretty_print", False):
        print(OmegaConf.to_yaml(cfg))
    
    if "training" not in cfg:
        cfg.training = cfg
    cfg.training.shuffle = False
    cfg.shuffle = False
    if "debugging" not in cfg:
        cfg.debugging = {}
    trainer = Trainer(cfg)
    trainer.eval(stage_name = "test")
    #
    # # https://github.com/facebookresearch/hydra/issues/440
    GlobalHydra.get_state().clear()
    return 0


if __name__ == "__main__":
    main()
