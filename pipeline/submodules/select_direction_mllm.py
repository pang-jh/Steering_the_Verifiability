import json
import torch
import math
import matplotlib.pyplot as plt
import os
import numpy as np

from typing import List, Optional
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks, get_activation_addition_input_pre_hook, get_direction_ablation_input_pre_hook, get_direction_ablation_output_hook
from pipeline.submodules.generate_directions import tokenize_instructions_fn 
def plot_scores(
    scores: Float[Tensor, 'n_pos n_layer'],
    baseline_score: Optional[float],
    token_labels: List[str],
    title: str,
    artifact_dir: str,
    artifact_name: str,
    ylabel: str = 'Score',
):

    n_pos, n_layer = scores.shape

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(-n_pos, 0):
        p_idx = i + n_pos
        score_data = scores[p_idx].cpu().numpy() if isinstance(scores, torch.Tensor) else scores[p_idx]
        ax.plot(
            list(range(n_layer)),
            score_data,
            label=f'pos {i}: {token_labels[p_idx] if p_idx < len(token_labels) else str(i)}',
            marker='o',
            markersize=2,
            linewidth=1
        )

    if baseline_score is not None:
        ax.axhline(y=baseline_score, color='black', linestyle='--', linewidth=2)
        ax.annotate(
            f'Baseline: {baseline_score:.4f}', 
            xy=(0.98, baseline_score), 
            xycoords=('axes fraction', 'data'),
            textcoords='offset points',
            xytext=(-5, 5),
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize=9,
            color='black'
        )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Layer', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(title='Position', loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{artifact_dir}/{artifact_name}.png", dpi=150)
    plt.close(fig)

def hallucination_score(
    logits: Float[Tensor, 'batch seq d_vocab_out'],
    gt_texts: List[str], 
    yes_token_ids: List[int],
    no_token_ids: List[int],
    unknown_token_ids: List[int],
    epsilon: Float = 1e-10,
):

    logits = logits.to(torch.float64)
    last_token_logits = logits[:, -1, :] 
    probs = torch.nn.functional.softmax(last_token_logits, dim=-1) # [batch, vocab]

    p_yes_concept = probs[:, yes_token_ids].sum(dim=-1)
    p_no_concept = probs[:, no_token_ids].sum(dim=-1)
    p_unknown_concept = probs[:, unknown_token_ids].sum(dim=-1)

    hr_scores = []
    acc_scores = []
    
    for i, gt_text in enumerate(gt_texts):
        gt_clean = gt_text.strip().lower()

        p_unk = p_unknown_concept[i]
        p_yes = p_yes_concept[i]
        p_no = p_no_concept[i]
        
        if "是" in gt_clean:
            p_gt = p_yes
            p_hallu = p_no
        elif "否" in gt_clean:
            p_gt = p_no
            p_hallu = p_yes

        s_hr = torch.log(p_hallu + epsilon) - torch.log(p_gt + p_unk + p_hallu + epsilon)

        s_acc = torch.log(p_gt + epsilon) - torch.log(p_gt + p_unk + p_hallu + epsilon)

        hr_scores.append(s_hr)
        acc_scores.append(s_acc)

    return torch.stack(hr_scores), torch.stack(acc_scores)


def hallucination_score_from_last_logits(
    last_token_logits: Float[Tensor, 'batch d_vocab_out'],
    gt_texts: List[str],
    yes_token_ids: List[int],
    no_token_ids: List[int],
    unknown_token_ids: List[int],
    epsilon: Float = 1e-10,
):
    logits = last_token_logits.to(torch.float64)
    probs = torch.nn.functional.softmax(logits, dim=-1)

    p_yes_concept = probs[:, yes_token_ids].sum(dim=-1)
    p_no_concept = probs[:, no_token_ids].sum(dim=-1)
    p_unknown_concept = probs[:, unknown_token_ids].sum(dim=-1)

    hr_scores = []
    acc_scores = []
    for i, gt_text in enumerate(gt_texts):
        gt_clean = gt_text.strip().lower()

        p_unk = p_unknown_concept[i]
        p_yes = p_yes_concept[i]
        p_no = p_no_concept[i]

        if "是" in gt_clean:
            p_gt = p_yes
            p_hallu = p_no
        elif "否" in gt_clean:
            p_gt = p_no
            p_hallu = p_yes

        s_hr = torch.log(p_hallu + epsilon) - torch.log(p_gt + p_unk + p_hallu + epsilon)
        s_acc = torch.log(p_gt + epsilon) - torch.log(p_gt + p_unk + p_hallu + epsilon)

        hr_scores.append(s_hr)
        acc_scores.append(s_acc)

    return torch.stack(hr_scores), torch.stack(acc_scores)


def build_tokenized_batches(dataset, processor, batch_size=1):

    cached_batches = []
    for i in range(0, len(dataset), batch_size):
        batch_data = dataset[i:i + batch_size]
        inputs = tokenize_instructions_fn(processor, batch_data)
        gt_texts = [data['gt'] for data in batch_data]
        cached_batches.append({"inputs": inputs, "gt_texts": gt_texts})
    return cached_batches


def get_evaluation_scores(model_base, dataset, processor, yes_token_ids, no_token_ids, unknown_token_ids, fwd_pre_hooks=[], fwd_hooks=[], batch_size=1, cached_batches=None):
    all_hr_scores = []
    all_acc_scores = []

    if cached_batches is not None:
        iterable_batches = cached_batches
    else:
        iterable_batches = build_tokenized_batches(dataset, processor, batch_size=batch_size)

    with torch.no_grad():
        for batch in iterable_batches:
            inputs = batch['inputs']
            gt_texts = batch['gt_texts']
            model_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to(model_base.device)
                else:
                    model_inputs[k] = v
                    
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                logits = model_base(**model_inputs).logits
            # print("yes_token_ids:", yes_token_ids)
            batch_hr, batch_acc = hallucination_score(
                logits, gt_texts, 
                yes_token_ids, no_token_ids, unknown_token_ids
            )
            all_hr_scores.append(batch_hr)
            all_acc_scores.append(batch_acc)

    return torch.cat(all_hr_scores), torch.cat(all_acc_scores)

def kl_div_fn(logits_a, logits_b, epsilon=1e-6):
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)
    return torch.mean(kl_divs, dim=-1)

def get_last_position_logits(model_base, dataset, processor, fwd_pre_hooks=[], fwd_hooks=[], batch_size=1, cached_batches=None):
    all_logits = []

    if cached_batches is not None:
        iterable_batches = cached_batches
    else:
        iterable_batches = build_tokenized_batches(dataset, processor, batch_size=batch_size)

    with torch.no_grad():
        for batch in iterable_batches:
            inputs = batch['inputs']
            model_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    model_inputs[k] = v.to(model_base.device)
                else:
                    model_inputs[k] = v
                    
            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                logits = model_base(**model_inputs).logits

            all_logits.append(logits[:, -1, :].detach().cpu())
    return torch.cat(all_logits, dim=0)

def select_and_save_direction(
    cfg,
    model_base,
    processor,
    hallucinated_dataset,    
    nonhallucinated_dataset, 
    candidate_directions: Float[Tensor, 'n_pos n_layer d_model'],
    kl_threshold=0.1,        
    nh_acc_degradation_threshold=0.1,
    prune_layer_percentage=0.1,
    batch_size=1,
    save_prefix=""
):
    artifact_dir = os.path.join(cfg.artifact_path(), 'select_direction')
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    n_pos, n_layer, d_model = candidate_directions.shape

    yes_token_ids = processor.tokenizer.encode("是", add_special_tokens=False)
    no_token_ids = processor.tokenizer.encode("否", add_special_tokens=False)
    unknown_token_ids = processor.tokenizer.encode("不确定", add_special_tokens=False)

    hallucinated_cached_batches = build_tokenized_batches(hallucinated_dataset, processor, batch_size=batch_size)
    nonhallucinated_cached_batches = build_tokenized_batches(nonhallucinated_dataset, processor, batch_size=batch_size)
    nonhallucinated_gt_texts = [d['gt'] for d in nonhallucinated_dataset]

    base_hr_h, base_acc_h = get_evaluation_scores(
        model_base,
        hallucinated_dataset,
        processor,
        yes_token_ids,
        no_token_ids,
        unknown_token_ids,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size,
        cached_batches=hallucinated_cached_batches,
    )
    baseline_hr_h_val = base_hr_h.mean().item()
    # print(f"baseline_hr_h_val: {baseline_hr_h_val}")
    baseline_acc_h_val = base_acc_h.mean().item()

    hr_h_matrix = torch.zeros((n_pos, n_layer)) 
    acc_h_matrix = torch.zeros((n_pos, n_layer))

    hr_nh_matrix = torch.zeros((n_pos, n_layer))
    acc_nh_matrix = torch.zeros((n_pos, n_layer))
    
    kl_scores_matrix = torch.zeros((n_pos, n_layer))

    print(f"Monitoring Token IDs: Yes={yes_token_ids}, No={no_token_ids}, Unk={unknown_token_ids}")
    
    print("Computing baseline logits for KL")
    baseline_logits_nh = get_last_position_logits(
        model_base, nonhallucinated_dataset, 
        processor, fwd_pre_hooks=[], fwd_hooks=[], 
        batch_size=batch_size,
        cached_batches=nonhallucinated_cached_batches,
    )

    base_hr_nh, base_acc_nh = hallucination_score_from_last_logits(
        baseline_logits_nh.to(model_base.model.device),
        nonhallucinated_gt_texts,
        yes_token_ids,
        no_token_ids,
        unknown_token_ids,
    )
    baseline_hr_nh_val = base_hr_nh.mean().item()
    baseline_acc_nh_val = base_acc_nh.mean().item()


    for p_idx, source_pos in enumerate(range(-n_pos, 0)): 
        for source_layer in tqdm(range(n_layer), desc=f"Layers"):
            
            direction = candidate_directions[p_idx, source_layer]
            
            fwd_pre_hooks = [(model_base.model.language_model.layers[layer], get_direction_ablation_input_pre_hook(direction)) for layer in range(model_base.model.config.num_hidden_layers)]                       # Qwen2.5-VL
            fwd_hooks = [(model_base.model.language_model.layers[layer].self_attn, get_direction_ablation_output_hook(direction)) for layer in range(model_base.model.config.num_hidden_layers)]                    # Qwen2.5-VL
            fwd_hooks += [(model_base.model.language_model.layers[layer].mlp, get_direction_ablation_output_hook(direction)) for layer in range(model_base.model.config.num_hidden_layers)]                         # Qwen2.5-VL

            # fwd_pre_hooks = [(model_base.model.language_model.layers[layer], get_direction_ablation_input_pre_hook(direction)) for layer in range(model_base.model.config.text_config.num_hidden_layers)]         # LLaVA-OneVision-1.5-8B
            # fwd_hooks = [(model_base.model.language_model.layers[layer].self_attn, get_direction_ablation_output_hook(direction)) for layer in range(model_base.model.config.text_config.num_hidden_layers)]      # LLaVA-OneVision-1.5-8B
            # fwd_hooks += [(model_base.model.language_model.layers[layer].mlp, get_direction_ablation_output_hook(direction)) for layer in range(model_base.model.config.text_config.num_hidden_layers)]           # LLaVA-OneVision-1.5-8B
            
            intervention_logits = get_last_position_logits(
                model_base, nonhallucinated_dataset,
                processor, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, 
                batch_size=batch_size,
                cached_batches=nonhallucinated_cached_batches,
            )

            kl_scores_matrix[p_idx, source_layer] = kl_div_fn(baseline_logits_nh.to(model_base.model.device), intervention_logits.to(model_base.model.device)).mean(dim=0).item()
            
            hr_h, acc_h = get_evaluation_scores(
                model_base,
                hallucinated_dataset,
                processor,
                yes_token_ids,
                no_token_ids,
                unknown_token_ids,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size,
                cached_batches=hallucinated_cached_batches,
            )
            hr_h_matrix[p_idx, source_layer] = hr_h.mean().item()
            acc_h_matrix[p_idx, source_layer] = acc_h.mean().item()

            hr_nh, acc_nh = hallucination_score_from_last_logits(
                intervention_logits.to(model_base.model.device),
                nonhallucinated_gt_texts,
                yes_token_ids,
                no_token_ids,
                unknown_token_ids,
            )
            hr_nh_matrix[p_idx, source_layer] = hr_nh.mean().item()
            acc_nh_matrix[p_idx, source_layer] = acc_nh.mean().item()
    
    token_labels = [f"pos_{i}" for i in range(-n_pos, 0)]
    plot_scores(
        scores=hr_h_matrix,
        baseline_score=baseline_hr_h_val,
        token_labels=token_labels,
        title=f'{save_prefix} Hallucination Rate Score on Hallucinated Data (Lower is Better)',
        artifact_dir=artifact_dir,
        artifact_name=f'{save_prefix}_hr_h_scores',
        ylabel='HR Score'
    )
    
    plot_scores(
        scores=hr_nh_matrix,
        baseline_score=baseline_hr_nh_val,
        token_labels=token_labels,
        title=f'{save_prefix} Hallucination Rate Score on Non-Hallucinated Data',
        artifact_dir=artifact_dir,
        artifact_name=f'{save_prefix}_hr_nh_scores',
        ylabel='HR Score'
    )

    plot_scores(
        scores=acc_h_matrix,
        baseline_score=baseline_acc_h_val,
        token_labels=token_labels,
        title=f'{save_prefix} Accuracy Score on Hallucinated Data',
        artifact_dir=artifact_dir,
        artifact_name=f'{save_prefix}_acc_h_scores',
        ylabel='ACC Score'
    )
    
    plot_scores(
        scores=acc_nh_matrix,
        baseline_score=baseline_acc_nh_val,
        token_labels=token_labels,
        title=f'{save_prefix} Accuracy Score on Non-Hallucinated Data',
        artifact_dir=artifact_dir,
        artifact_name=f'{save_prefix}_acc_nh_scores',
        ylabel='ACC Score'
    )
    
    plot_scores(
        scores=kl_scores_matrix,
        baseline_score=0.0,
        token_labels=token_labels,
        title=f'{save_prefix} KL Divergence Score on Non-Hallucinated Data',
        artifact_dir=artifact_dir,
        artifact_name=f'{save_prefix}_kl_scores',
        ylabel='KL Divergence'
    )

    filtered_candidates = []
    json_output_all_scores = []
    for p_idx, source_pos in enumerate(range(-n_pos, 0)):
        for layer in range(n_layer):
            val_hr_h = hr_h_matrix[p_idx, layer].item()
            val_hr_nh = hr_nh_matrix[p_idx, layer].item()
            val_acc_h = acc_h_matrix[p_idx, layer].item()
            val_acc_nh = acc_nh_matrix[p_idx, layer].item()
            val_kl = kl_scores_matrix[p_idx, layer].item()
            res = {
                "pos": source_pos,
                "layer": layer,
                "hr_h_score": val_hr_h,
                "hr_nh_score": val_hr_nh,
                "acc_h_score": val_acc_h,
                "acc_nh_score": val_acc_nh,
                "kl_score": val_kl
            }
            json_output_all_scores.append(res)

            if any(map(math.isnan, [val_hr_h, val_hr_nh, val_acc_h, val_acc_nh, val_kl])):
                continue
            if prune_layer_percentage is not None and layer >= int(n_layer * (1.0 - prune_layer_percentage)):
                continue
            if kl_threshold is not None and val_kl > kl_threshold:
                continue
            if nh_acc_degradation_threshold is not None and (baseline_acc_nh_val - val_acc_nh) > nh_acc_degradation_threshold:
                continue
            filtered_candidates.append(res)

    with open(f"{artifact_dir}/{save_prefix}_all_candidates.json", "w") as f:
        json.dump(json_output_all_scores, f, indent=4)
    
    if not filtered_candidates:
        print(f"[{save_prefix}] WARNING: No directions passed filter !!!")
        best_candidate = min(json_output_all_scores, key=lambda x: x['hr_h_score'])
    else:
        filtered_candidates.sort(key=lambda x: x['hr_h_score'])
        best_candidate = filtered_candidates[0]
        with open(f"{artifact_dir}/{save_prefix}_filtered_candidates.json", "w") as f:
            json.dump(filtered_candidates, f, indent=4)
        
    print(f"Selected Direction: Position {best_candidate['pos']}, Layer {best_candidate['layer']}")
    print(f"Metrics:")
    print(f" - HR H Score: {best_candidate['hr_h_score']:.4f} (Base: {baseline_hr_h_val:.4f})")
    print(f" - HR NH Score: {best_candidate['hr_nh_score']:.4f} (Base: {baseline_hr_nh_val:.4f})")
    print(f" - ACC H Score: {best_candidate['acc_h_score']:.4f} (Base: {baseline_acc_h_val:.4f})")
    print(f" - ACC NH Score: {best_candidate['acc_nh_score']:.4f} (Base: {baseline_acc_nh_val:.4f})")
    print(f" - KL Score:      {best_candidate['kl_score']:.4f}")

    pos = best_candidate['pos']
    layer = best_candidate['layer']

    select_direction = candidate_directions[pos + n_pos, layer]
    # print("pos + n_pos:", pos + n_pos)
    print("select_direction.size():", select_direction.size())
    with open(f'{artifact_dir}/{save_prefix}_direction_metadata.json', "w") as f:
        json.dump({"pos": pos, "layer": layer, "metrics": best_candidate}, f, indent=4)

    torch.save(select_direction, f'{artifact_dir}/{save_prefix}_direction.pt')
    
    return pos, layer, select_direction