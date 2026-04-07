
import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def get_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_all_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
):
    fwd_pre_hooks = [(model_base.model.language_model.layers[layer], get_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model.language_model.layers[layer].self_attn, get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model.language_model.layers[layer].mlp, get_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks


def get_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal vector

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_multi_direction_ablation_input_pre_hook(directions: List[Tensor], coeffs: List[float]):

    def hook_fn(module, input):
        nonlocal directions, coeffs
        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        for direction, coeff in zip(directions, coeffs):
            if coeff == 0:
                continue
            
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction = direction.to(activation)
            activation -= coeff * (activation @ direction).unsqueeze(-1) * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn


def get_multi_direction_ablation_output_hook(directions: List[Tensor], coeffs: List[float]):

    def hook_fn(module, input, output):
        nonlocal directions, coeffs
        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        for direction, coeff in zip(directions, coeffs):
            if coeff == 0:
                continue
                
            direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction = direction.to(activation)
            activation -= coeff * (activation @ direction).unsqueeze(-1) * direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation
    return hook_fn


def get_all_multi_direction_ablation_hooks(
    model_base,
    directions: List[Float[Tensor, 'd_model']],
    coeffs: List[float] = None,
):
    if coeffs is None:
        coeffs = [1.0] * len(directions)
    
    assert len(directions) == len(coeffs), "directions和coeffs长度必须一致"
    
    fwd_pre_hooks = [
        (model_base.model.language_model.layers[layer], 
         get_multi_direction_ablation_input_pre_hook(directions=directions, coeffs=coeffs)) 
        for layer in range(model_base.model.config.num_hidden_layers)                         # Qwen2.5-VL
        # for layer in range(model_base.model.config.text_config.num_hidden_layers)           # LLaVA-OneVision-1.5-8B  
    ]
    
    fwd_hooks = [
        (model_base.model.language_model.layers[layer].self_attn, 
         get_multi_direction_ablation_output_hook(directions=directions, coeffs=coeffs)) 
        for layer in range(model_base.model.config.num_hidden_layers)                         # Qwen2.5-VL
        # for layer in range(model_base.model.config.text_config.num_hidden_layers)           # LLaVA-OneVision-1.5-8B
    ]
    
    fwd_hooks += [
        (model_base.model.language_model.layers[layer].mlp, 
         get_multi_direction_ablation_output_hook(directions=directions, coeffs=coeffs)) 
        for layer in range(model_base.model.config.num_hidden_layers)                         # Qwen2.5-VL
        # for layer in range(model_base.model.config.text_config.num_hidden_layers)           # LLaVA-OneVision-1.5-8B
    ]

    return fwd_pre_hooks, fwd_hooks