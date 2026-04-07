import torch
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from pipeline.submodules.generate_directions import tokenize_instructions_fn
from pipeline.utils.hook_utils import add_hooks

def generate_completions_with_logits(
    model_base, processor, fwd_pre_hooks=[], fwd_hooks=[], 
    dataset=None, batch_size=1, max_new_tokens=32
):

    completions = []
    logits_metrics = None
    
    images = [data['image'] for data in dataset]
    texts = [data['text'] for data in dataset]
    origin_preds = [data['pred'] for data in dataset]
    gts = [data['gt'] for data in dataset]
    
    yes_token_ids = processor.tokenizer.encode("是", add_special_tokens=False)
    no_token_ids = processor.tokenizer.encode("否", add_special_tokens=False)
    unknown_token_ids = processor.tokenizer.encode("不确定", add_special_tokens=False)
    
    all_baseline_logits = []
    all_intervention_logits = []
    
    for i in tqdm(range(0, len(dataset), batch_size)):
        inputs = tokenize_instructions_fn(processor, dataset=dataset[i:i+batch_size])

        model_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.to(model_base.model.device)
            else:
                model_inputs[k] = v

        with torch.no_grad():
            baseline_outputs = model_base(**model_inputs)
            baseline_logits = baseline_outputs.logits[:, -1, :].cpu()  # [batch, vocab]
            all_baseline_logits.append(baseline_logits)
        
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            with torch.no_grad():
                intervention_outputs = model_base(**model_inputs)
                intervention_logits = intervention_outputs.logits[:, -1, :].cpu()
                all_intervention_logits.append(intervention_logits)
            
            generated_ids = model_base.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=False)
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        decoded_answers = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )
        
        for idx, answer in enumerate(decoded_answers):
            completions.append({
                'text': texts[i + idx],
                'image': images[i + idx],
                'gt': gts[i + idx],
                'origin_pred': origin_preds[i + idx],
                'response': answer
            })

    baseline_logits_all = torch.cat(all_baseline_logits, dim=0)  # [n_samples, vocab]
    intervention_logits_all = torch.cat(all_intervention_logits, dim=0)
    
    logits_metrics = compute_logits_metrics(
        baseline_logits=baseline_logits_all,
        intervention_logits=intervention_logits_all,
        gts=gts,
        yes_token_ids=yes_token_ids, no_token_ids=no_token_ids, unknown_token_ids=unknown_token_ids
    )

    return completions, logits_metrics


def compute_logits_metrics(
    baseline_logits: torch.Tensor,
    intervention_logits: torch.Tensor,
    gts: List[str],
    yes_token_ids: List[int],
    no_token_ids: List[int],
    unknown_token_ids: List[int],
    epsilon: float = 1e-10
):

    baseline_logits = baseline_logits.to(torch.float64)
    intervention_logits = intervention_logits.to(torch.float64)
    
    baseline_probs = torch.softmax(baseline_logits, dim=-1)
    intervention_probs = torch.softmax(intervention_logits, dim=-1)
    
    p_yes_base = baseline_probs[:, yes_token_ids].sum(dim=-1)
    p_no_base = baseline_probs[:, no_token_ids].sum(dim=-1)
    p_unk_base = baseline_probs[:, unknown_token_ids].sum(dim=-1)
    
    p_yes_inter = intervention_probs[:, yes_token_ids].sum(dim=-1)
    p_no_inter = intervention_probs[:, no_token_ids].sum(dim=-1)
    p_unk_inter = intervention_probs[:, unknown_token_ids].sum(dim=-1)
    
    metrics = {
        'per_sample': [],
        'aggregate': {}
    }
    
    for i, gt in enumerate(gts):
        gt_clean = gt.strip().lower()
        
        if "是" in gt_clean:
            gt_label = "是"
            p_gt_base = p_yes_base[i]
            p_gt_inter = p_yes_inter[i]
            p_hallu_base = p_no_base[i]
            p_hallu_inter = p_no_inter[i]
        elif "否" in gt_clean:
            gt_label = "否"
            p_gt_base = p_no_base[i]
            p_gt_inter = p_no_inter[i]
            p_hallu_base = p_yes_base[i]
            p_hallu_inter = p_yes_inter[i]
        
        # HR = P(hallu) / [P(gt) + P(hallu)]
        hr_base = (p_hallu_base / (p_yes_base[i] + p_no_base[i] + p_unk_base[i] + epsilon)).item()
        hr_inter = (p_hallu_inter / (p_yes_inter[i] + p_no_inter[i] + p_unk_inter[i] + epsilon)).item()
        
        # ACC = P(gt) / [P(yes) + P(no) + P(unk)]
        acc_base = (p_gt_base / (p_yes_base[i] + p_no_base[i] + p_unk_base[i] + epsilon)).item()
        acc_inter = (p_gt_inter / (p_yes_inter[i] + p_no_inter[i] + p_unk_inter[i] + epsilon)).item()
        
        unk_tendency_base = (p_unk_base[i] / (p_yes_base[i] + p_no_base[i] + p_unk_base[i] + epsilon)).item()
        unk_tendency_inter = (p_unk_inter[i] / (p_yes_inter[i] + p_no_inter[i] + p_unk_inter[i] + epsilon)).item()

        hr_increase_pct = (hr_inter - hr_base) / (hr_base + epsilon)
        acc_increase_pct = (acc_inter - acc_base) / (acc_base + epsilon)
        unk_tendency_change_pct = (unk_tendency_inter - unk_tendency_base) / (unk_tendency_base + epsilon)

        sample_metrics = {
            'gt': gt_label,
            'probs_baseline': {
                '是': p_yes_base[i].item(),
                '否': p_no_base[i].item(),
                '不确定': p_unk_base[i].item()
            },
            'probs_intervention': {
                '是': p_yes_inter[i].item(),
                '否': p_no_inter[i].item(),
                '不确定': p_unk_inter[i].item()
            },
            'hallucination_rate': {
                'baseline': hr_base,
                'intervention': hr_inter,
                'increase': hr_inter - hr_base,
                'increase_percentage': hr_increase_pct
            },
            'accuracy': {
                'baseline': acc_base,
                'intervention': acc_inter,
                'increase': acc_inter - acc_base,
                'increase_percentage': acc_increase_pct
            },
            'unknown_tendency': {
                'baseline': unk_tendency_base,
                'intervention': unk_tendency_inter,
                'change': unk_tendency_inter - unk_tendency_base,
                'change_percentage': unk_tendency_change_pct
            },
            'prob_change': {
                '是': (p_yes_inter[i] - p_yes_base[i]).item(),
                '否': (p_no_inter[i] - p_no_base[i]).item(),
                '不确定': (p_unk_inter[i] - p_unk_base[i]).item()
            }
        }
        
        metrics['per_sample'].append(sample_metrics)
    
    metrics['aggregate'] = {
        'mean_hr_increase': round(np.mean([m['hallucination_rate']['increase'] for m in metrics['per_sample']]), 4),
        'mean_acc_increase': round(np.mean([m['accuracy']['increase'] for m in metrics['per_sample']]), 4),
        'mean_unk_tendency_change': round(np.mean([m['unknown_tendency']['change'] for m in metrics['per_sample']]), 4),
        'mean_hr_increase_percentage': round(np.mean([m['hallucination_rate']['increase_percentage'] for m in metrics['per_sample']]), 4),
        'mean_acc_increase_percentage': round(np.mean([m['accuracy']['increase_percentage'] for m in metrics['per_sample']]), 4),
        'mean_unk_tendency_change_percentage': round(np.mean([m['unknown_tendency']['change_percentage'] for m in metrics['per_sample']]), 4),
        
        'samples_hr_increased': sum([1 for m in metrics['per_sample'] if m['hallucination_rate']['increase'] < 0]),
        'samples_acc_increased': sum([1 for m in metrics['per_sample'] if m['accuracy']['increase'] > 0]),
        'effective_rate_hr': round(sum([1 for m in metrics['per_sample'] if m['hallucination_rate']['increase'] < 0]) / len(metrics['per_sample']), 4),
        'effective_rate_acc': round(sum([1 for m in metrics['per_sample'] if m['accuracy']['increase'] > 0]) / len(metrics['per_sample']), 4),
        
        'mean_kl_divergence': round(compute_kl_divergence(baseline_probs, intervention_probs).mean().item(), 4),
        
        'baseline_mean_hr': round(np.mean([m['hallucination_rate']['baseline'] for m in metrics['per_sample']]), 4),
        'baseline_mean_acc': round(np.mean([m['accuracy']['baseline'] for m in metrics['per_sample']]), 4),
        'intervention_mean_hr': round(np.mean([m['hallucination_rate']['intervention'] for m in metrics['per_sample']]), 4),
        'intervention_mean_acc': round(np.mean([m['accuracy']['intervention'] for m in metrics['per_sample']]), 4),
        
        'n_samples': len(gts)
    }
    
    return metrics


def compute_kl_divergence(probs_base, probs_inter, epsilon=1e-10):
    return torch.sum(
        probs_base * (torch.log(probs_base + epsilon) - torch.log(probs_inter + epsilon)),
        dim=-1
    )