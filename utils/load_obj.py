from typing import *
import importlib


def load_obj(hydra_obj: str, **kwargs) -> Any:
    """
    Adapted from: https://github.com/Erlemar/pytorch_tempest/blob/master/src/utils/technical_utils.py
    """
    obj_path_list = hydra_obj._target_.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else ''
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    arguments_dict = {**hydra_obj, **kwargs}
    if '_target_' in arguments_dict:
        del arguments_dict['_target_']
    return getattr(module_obj, obj_name)(**arguments_dict)
