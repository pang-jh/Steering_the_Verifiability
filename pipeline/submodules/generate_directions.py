import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks
from qwen_vl_utils import process_vision_info

QWEN25_VL_CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{instruction}<|im_end|>
<|im_start|>assistant
"""

LLAVA_ONEVISION_CHAT_TEMPLATE = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>{instruction}<|im_end|>
<|im_start|>assistant
"""

def tokenize_instructions_fn(processor, dataset):
    batch_texts, batch_imgs = [], []
    for data in dataset:
        img = data.get("image")
        txt = data.get("text")
        messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": txt},
            ],
        }]
        # print("messages:",messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        batch_texts.append(text)
        batch_imgs.append(image_inputs)
    # original_padding_side = processor.tokenizer.padding_side
    # processor.tokenizer.padding_side = 'right'
    inputs = processor(
        text=batch_texts,
        images=batch_imgs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs


def get_mean_activations_pre_hook(layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn

def get_mean_activations(model, processor, dataset, block_modules: List[torch.nn.Module], batch_size=1, positions=[-1]):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_samples = len(dataset)

    n_layers = model.config.num_hidden_layers                   # Qwen2.5-VL
    # n_layers = model.config.text_config.num_hidden_layers     # LLaVA-OneVision-1.5-8B
    
    d_model = model.config.hidden_size                   # Qwen2.5-VL   
    # d_model = model.config.text_config.hidden_size     # LLaVA-OneVision-1.5-8B

    mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    fwd_pre_hooks = [(block_modules[layer], get_mean_activations_pre_hook(layer=layer, cache=mean_activations, n_samples=n_samples, positions=positions)) for layer in range(n_layers)]

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size)):
            # inputs = tokenize_instructions_fn(processor, [image], [text])
            inputs = tokenize_instructions_fn(processor, dataset=dataset[i:i+batch_size])
            model_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to(model.device)
                else:
                    model_inputs[k] = v
                    
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
                model(**model_inputs)

    return mean_activations

def get_mean_diff(model, processor, hallucinated_dataset, nonhallucinated_dataset, batch_size, positions=[-1]):
    mean_activations_h = get_mean_activations(model, processor, hallucinated_dataset, model.language_model.layers, batch_size=batch_size, positions=positions)
    mean_activations_nh = get_mean_activations(model, processor, nonhallucinated_dataset, model.language_model.layers, batch_size=batch_size, positions=positions)

    mean_diff = mean_activations_h - mean_activations_nh

    return mean_diff

def generate_directions(model_base, processor, hallucinated_dataset, nonhallucinated_dataset, batch_size, artifact_dir, save_name="mean_diffs"):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    eoi_toks = processor.tokenizer.encode(QWEN25_VL_CHAT_TEMPLATE.split("{instruction}")[-1])           # Qwen2.5-VL
    print("eoi_toks:", QWEN25_VL_CHAT_TEMPLATE.split("{instruction}")[-1])                              # Qwen2.5-VL

    # eoi_toks = processor.tokenizer.encode(LLAVA_ONEVISION_CHAT_TEMPLATE.split("{instruction}")[-1])   # LLaVA-OneVision-1.5-8B
    # print("eoi_toks:", LLAVA_ONEVISION_CHAT_TEMPLATE.split("{instruction}")[-1])                      # LLaVA-OneVision-1.5-8B
    
    mean_diffs = get_mean_diff(model_base.model, processor, hallucinated_dataset, nonhallucinated_dataset, batch_size, positions=list(range(-len(eoi_toks), 0)))
    assert not mean_diffs.isnan().any(), f"mean_diffs contains NaN: {mean_diffs}"
    torch.save(mean_diffs, f"{artifact_dir}/{save_name}.pt")
    return mean_diffs