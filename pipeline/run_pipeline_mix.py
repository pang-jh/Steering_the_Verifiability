import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 

import torch
import json

from tqdm import tqdm
from dataset.load_dataset import load_dataset_split, load_dataset_test
from pipeline.utils.hook_utils import add_hooks
from pipeline.config import Config
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions, tokenize_instructions_fn
from pipeline.submodules.select_direction_mllm import select_and_save_direction
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from pipeline.eval_logits import generate_completions_with_logits
from pipeline.utils.hook_utils import get_all_multi_direction_ablation_hooks


def _normalize_direction(direction: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return direction / (direction.norm(dim=-1, keepdim=True) + eps)


def _build_mixed_direction(direction_obv: torch.Tensor, direction_elu: torch.Tensor, lam: float) -> torch.Tensor:
    # r_mix(lambda) = norm((1-lambda) * r_obv + lambda * r_elu)
    obv_hat = _normalize_direction(direction_obv)
    elu_hat = _normalize_direction(direction_elu)
    mixed = (1.0 - lam) * obv_hat + lam * elu_hat
    return _normalize_direction(mixed)

def _read_aggregate_metrics(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        agg = data.get("aggregate", {})
        return {
            "mean_hr_increase": agg.get("mean_hr_increase"),
            "mean_acc_increase": agg.get("mean_acc_increase"),
            "mean_unk_tendency_change": agg.get("mean_unk_tendency_change"),
            "mean_hr_increase_percentage": agg.get("mean_hr_increase_percentage"),
            "mean_acc_increase_percentage": agg.get("mean_acc_increase_percentage"),
            "mean_unk_tendency_change_percentage": agg.get("mean_unk_tendency_change_percentage"),
        }
    except FileNotFoundError:
        print(f"[warn] 未找到指标文件: {path}")
        return {}
    

def load_and_sample_datasets_train(data_h_train, data_nh_train):

    hallucinated_train = load_dataset_split(data_h_train)
    nonhallucinated_train = load_dataset_split(data_nh_train)

    return hallucinated_train, nonhallucinated_train


def generate_and_save_candidate_directions(cfg, model_base, processor, hallucinated_train, nonhallucinated_train, batch_size, save_name="mean_diffs"):
    save_dir = os.path.join(cfg.artifact_path(), 'generate_directions')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    mean_diffs = generate_directions(
        model_base,
        processor,
        hallucinated_train,
        nonhallucinated_train,
        batch_size,
        artifact_dir=save_dir,
        save_name=save_name
    )

    torch.save(mean_diffs, os.path.join(cfg.artifact_path(), f'generate_directions/{save_name}.pt'))

    return mean_diffs

def generate_completions(model_base, processor, fwd_pre_hooks=[], fwd_hooks=[], dataset=None, batch_size=1, max_new_tokens=32):
    completions = []
    images = [data['image'] for data in dataset]
    texts = [data['text'] for data in dataset]
    origin_preds = [data['pred'] for data in dataset]

    gts = [data['gt'] for data in dataset]
    for i in tqdm(range(0, len(dataset), batch_size)):
        inputs = tokenize_instructions_fn(processor, dataset=dataset[i:i+batch_size])

        model_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                model_inputs[k] = v.to(model_base.model.device)
            else:
                model_inputs[k] = v

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            outputs = model_base.generate( **model_inputs, max_new_tokens=max_new_tokens, do_sample = False)

        decoded_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
        decoded_answers = processor.batch_decode(
            decoded_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for idx, answer in enumerate(decoded_answers):

            completions.append({
                'text': texts[i + idx],
                'image': images[i + idx],
                'gt': gts[i + idx],
                'origin_pred': origin_preds[i + idx],
                'response': answer
            })

    return completions

def generate_and_save_completions_for_dataset(cfg, model_base, processor, fwd_pre_hooks, fwd_hooks, output_file, dataset, batch_size):

    completions, logits_metrics = generate_completions_with_logits(model_base, processor, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, dataset=dataset, batch_size=batch_size, max_new_tokens=cfg.max_new_tokens)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(completions, f, indent=4, ensure_ascii=False)

    output_root, _ = os.path.splitext(output_file)
    logits_output_file = f"{output_root}_logits.json"
    with open(logits_output_file, "w", encoding="utf-8") as f:
        json.dump(logits_metrics, f, indent=4, ensure_ascii=False)

def run_pipeline(model_path, train_obv, train_obv_nh, train_elu, train_elu_nh, val_obv, val_elu, val_nh, test_obv, test_elu, test_nh):
    # candidate_directions_obv = "pipeline/runs/qwen2.5-vl-7b-instruct/generate_directions/mean_diffs_obvious_balance.pt"
    # candidate_directions_elu = "pipeline/runs/qwen2.5-vl-7b-instruct/generate_directions/mean_diffs_elusive_balance.pt"
    # select_direction_obv = "pipeline/runs/qwen2.5-vl-7b-instruct/select_direction/obvious_balance_direction.pt"
    # select_direction_elu = "pipeline/runs/qwen2.5-vl-7b-instruct/select_direction/elusive_balance_direction.pt"

    kl_threshold = 0.1
    nh_acc_degradation_threshold = 0.1
    prune_layer_percentage = 0.1
    batch_size = 1
    model_alias = os.path.basename(model_path)
    cfg = Config(model_alias=model_alias, model_path=model_path)

    model_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto").eval()
    model_base.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_obv, train_obv_nh = load_and_sample_datasets_train(train_obv, train_obv_nh)
    train_elu, train_elu_nh = load_and_sample_datasets_train(train_elu, train_elu_nh)
    val_obv = load_dataset_split(val_obv)
    val_elu = load_dataset_split(val_elu)
    val_nh = load_dataset_split(val_nh)

    print("train_obv.size:",len(train_obv))
    print("train_obv_nh.size:",len(train_obv_nh))
    print("train_elu.size:",len(train_elu))
    print("train_elu_nh.size:",len(train_elu_nh))
    print("val_obv.size:",len(val_obv))
    print("val_elu.size:",len(val_elu))
    print("val_nh.size:",len(val_nh))

    print("train_obv:",train_obv[:1])

    candidate_directions_obv = generate_and_save_candidate_directions(cfg, model_base, processor, train_obv, train_obv_nh, batch_size, save_name="mean_diffs_obvious_balance")
    candidate_directions_elu = generate_and_save_candidate_directions(cfg, model_base, processor, train_elu, train_elu_nh, batch_size, save_name="mean_diffs_elusive_balance")
    # candidate_directions_obv = torch.load(candidate_directions_obv)
    # candidate_directions_elu = torch.load(candidate_directions_elu)

    pos_obv, layer_obv, direction_obv = select_and_save_direction(cfg, model_base, processor, val_obv, val_nh, candidate_directions_obv, kl_threshold, nh_acc_degradation_threshold, prune_layer_percentage, batch_size, save_prefix="obvious_balance")   # torch.Size([5, 36, 2048]) 
    pos_elu, layer_elu, direction_elu = select_and_save_direction(cfg, model_base, processor, val_elu, val_nh, candidate_directions_elu, kl_threshold, nh_acc_degradation_threshold, prune_layer_percentage, batch_size, save_prefix="elusive_balance")   # torch.Size([5, 36, 2048]) 
    # direction_obv = torch.load(select_direction_obv)
    # direction_elu = torch.load(select_direction_elu)

    test_obv = load_dataset_split(test_obv)
    test_elu = load_dataset_split(test_elu)
    test_nh = load_dataset_split(test_nh)


    # r_mix(lambda) = norm((1-lambda) * r_obv + lambda * r_elu)
    # x' = x - alpha * r_mix r_mix^T x
    alpha_strength = 1
    lambda_values = [i / 10 for i in range(0, 11, 1)]

    metrics_log_path = os.path.join(cfg.artifact_path(), "dual_ablation_metrics_mix_lambda.jsonl")

    for lam in lambda_values:
        mixed_direction = _build_mixed_direction(direction_obv, direction_elu, lam)
        lambda_tag = f"{lam:.2f}".replace(".", "p")

        dual_ablation_fwd_pre_hooks, dual_ablation_fwd_hooks = get_all_multi_direction_ablation_hooks(
            model_base,
            directions=[mixed_direction],
            coeffs=[alpha_strength]
        )

        output_file_obv = os.path.join(cfg.artifact_path(), f"test_obvious_lambda_{lambda_tag}.json")
        output_file_elu = os.path.join(cfg.artifact_path(), f"test_elusive_lambda_{lambda_tag}.json")
        output_file_nh = os.path.join(cfg.artifact_path(), f"test_nh_lambda_{lambda_tag}.json")

        print(f"[mix lambda={lam:.2f}] alpha={alpha_strength:.2f}")
        print(f"双探针干预*显式幻觉*测试集，输出到: {output_file_obv}")
        generate_and_save_completions_for_dataset(
            cfg, model_base, processor,
            dual_ablation_fwd_pre_hooks, dual_ablation_fwd_hooks,
            output_file_obv, test_obv, batch_size
        )

        print(f"双探针干预*隐式幻觉*测试集，输出到: {output_file_elu}")
        generate_and_save_completions_for_dataset(
            cfg, model_base, processor,
            dual_ablation_fwd_pre_hooks, dual_ablation_fwd_hooks,
            output_file_elu, test_elu, batch_size
        )

        print(f"双探针干预*无幻觉*测试集，输出到: {output_file_nh}")
        generate_and_save_completions_for_dataset(
            cfg, model_base, processor,
            dual_ablation_fwd_pre_hooks, dual_ablation_fwd_hooks,
            output_file_nh, test_nh, batch_size
        )

        obv_metrics_path = output_file_obv.replace(".json", "_logits.json")
        elu_metrics_path = output_file_elu.replace(".json", "_logits.json")
        nh_metrics_path = output_file_nh.replace(".json", "_logits.json")

        obv_metrics = _read_aggregate_metrics(obv_metrics_path)
        elu_metrics = _read_aggregate_metrics(elu_metrics_path)
        nh_metrics = _read_aggregate_metrics(nh_metrics_path)

        with open(metrics_log_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps({
                "alpha_strength": alpha_strength,
                "lambda": lam,
                "paths": {
                    "obvious_output": output_file_obv,
                    "elusive_output": output_file_elu,
                    "nh_output": output_file_nh,
                    "obvious_logits": obv_metrics_path,
                    "elusive_logits": elu_metrics_path,
                    "nh_logits": nh_metrics_path
                },
                "obvious": obv_metrics,
                "elusive": elu_metrics,
                "nh": nh_metrics
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    model_path = "/model/qwen2.5-vl-7b-instruct"

    train_obv = "dataset/all_data/7b/filter/split/train_obvious.jsonl"
    train_obv_nh = "dataset/all_data/7b/filter/split/train_nh.jsonl"
    train_elu = "dataset/all_data/7b/filter/split/train_elusive.jsonl"
    train_elu_nh = "dataset/all_data/7b/filter/split/train_nh.jsonl"

    val_obv = "dataset/all_data/7b/filter/split/val_obvious.jsonl"
    val_elu = "dataset/all_data/7b/filter/split/val_elusive.jsonl"
    val_nh = "dataset/all_data/7b/filter/split/val_nh.jsonl"

    test_obv = "dataset/all_data/7b/filter/split/test_obvious.jsonl"
    test_elu = "dataset/all_data/7b/filter/split/test_elusive.jsonl"
    test_nh = "dataset/all_data/7b/filter/split/test_nh.jsonl"


    run_pipeline(model_path, train_obv, train_obv_nh, train_elu, train_elu_nh, val_obv, val_elu, val_nh, test_obv, test_elu, test_nh)