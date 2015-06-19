import numpy as np
import sys

"""
# Generic Utils: generic utility functions
"""


def get_from_module(identifier, module_params, module_name, instantiate=False):
    """
    Get a function handle from module
    Inputs:
        - identifier: str, the function name in a module
        - module_params: dict, all the function names in a module
        - module_name: str, the target module name
    """
    if type(identifier) == str:
        res = module_params.get(identifier)
        if not res:
            raise Exception('Invalid' + module_name + ':' + identifier)
        if instantiate:
            return res()
        else:
            return res
    else:
        return identifier
