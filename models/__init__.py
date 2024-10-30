import importlib
from os import path as osp
from glob import glob
from utils.registry import MODEL_REGISTRY

__all__ = ['build_model']

# automatically scan and import model modules for registry
# scan all the files under the 'models' folder and collect files ending with '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in glob(osp.join(model_folder, '*.py'))]
# import all the model modules
_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]


def build_model(model_cfg):
    model_name = model_cfg.pop('name')
    model_cfg = model_cfg.pop(model_name)
    model = MODEL_REGISTRY.get(model_name)(**model_cfg)
    return model_name, model