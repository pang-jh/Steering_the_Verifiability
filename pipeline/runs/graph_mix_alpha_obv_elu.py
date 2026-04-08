import argparse
import json
import os
from typing import Any, Dict, List, Tuple
from matplotlib import font_manager
import matplotlib.pyplot as plt

font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if not os.path.exists(font_path):
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"

try:
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [prop.get_name(), "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
except Exception as e:
    print(f"Failed to load font: {e}")

plt.rcParams["axes.unicode_minus"] = False


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def choose_alpha_record(rows: List[Dict[str, Any]], target_alpha: float) -> Dict[str, Any]:
    return min(rows, key=lambda r: abs(float(r.get("lambda", 0.0)) - target_alpha))


def load_logits_and_outputs(logits_path: str, output_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    with open(logits_path, "r", encoding="utf-8") as f:
        logits_data = json.load(f)
    with open(output_path, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    per_sample = logits_data.get("per_sample", [])
    if len(per_sample) != len(output_data):
        n = min(len(per_sample), len(output_data))
        per_sample = per_sample[:n]
        output_data = output_data[:n]
    return per_sample, output_data


def topk_by_key(items: List[Dict[str, Any]], key: str, k: int, reverse: bool = True) -> List[Dict[str, Any]]:
    return sorted(items, key=lambda x: x.get(key, 0.0), reverse=reverse)[:k]


def build_case_candidates(rows: List[Dict[str, Any]], topk: int = 5) -> Dict[str, Any]:
    low = choose_alpha_record(rows, 0.1)
    mid = choose_alpha_record(rows, 0.5)
    high = choose_alpha_record(rows, 0.9)

    cases = {
        "low_alpha": {
            "alpha": low.get("alpha"),
            "obv_coeff": low.get("obv_coeff"),
            "elu_coeff": low.get("elu_coeff"),
            "samples": [],
        },
        "mid_alpha": {
            "alpha": mid.get("alpha"),
            "obv_coeff": mid.get("obv_coeff"),
            "elu_coeff": mid.get("elu_coeff"),
            "samples": [],
        },
        "high_alpha": {
            "alpha": high.get("alpha"),
            "obv_coeff": high.get("obv_coeff"),
            "elu_coeff": high.get("elu_coeff"),
            "samples": [],
        },
    }

    low_per, low_out = load_logits_and_outputs(
        safe_get(low, ["paths", "elusive_logits"]),
        safe_get(low, ["paths", "elusive_output"]),
    )
    low_items = []
    for i, m in enumerate(low_per):
        hr_i = safe_get(m, ["hallucination_rate", "intervention"], 0.0)
        low_items.append(
            {
                "index": i,
                "score": hr_i,
                "reason": "low_alpha_high_elusive_hr",
                "metrics": m,
                "sample": low_out[i],
            }
        )
    cases["low_alpha"]["samples"] = topk_by_key(low_items, "score", topk, reverse=True)

    mid_per, mid_out = load_logits_and_outputs(
        safe_get(mid, ["paths", "elusive_logits"]),
        safe_get(mid, ["paths", "elusive_output"]),
    )
    mid_items = []
    for i, m in enumerate(mid_per):
        hr_i = safe_get(m, ["hallucination_rate", "intervention"], 0.0)
        unk_change = abs(float(safe_get(m, ["unknown_tendency", "change"], 0.0)))
        balance_score = hr_i + 0.5 * unk_change
        mid_items.append(
            {
                "index": i,
                "score": balance_score,
                "reason": "mid_alpha_balanced_low_hr_and_moderate_uncertainty",
                "metrics": m,
                "sample": mid_out[i],
            }
        )
    cases["mid_alpha"]["samples"] = topk_by_key(mid_items, "score", topk, reverse=False)

    high_per, high_out = load_logits_and_outputs(
        safe_get(high, ["paths", "nh_logits"]),
        safe_get(high, ["paths", "nh_output"]),
    )
    high_items = []
    for i, m in enumerate(high_per):
        unk_i = safe_get(m, ["unknown_tendency", "intervention"], 0.0)
        high_items.append(
            {
                "index": i,
                "score": unk_i,
                "reason": "high_alpha_overcautious_high_nh_unknown",
                "metrics": m,
                "sample": high_out[i],
            }
        )
    cases["high_alpha"]["samples"] = topk_by_key(high_items, "score", topk, reverse=True)

    return cases


def plot_alpha_curve(
    rows: List[Dict[str, Any]],
    output_png: str,
    share_ylim: bool = False,
    ylim_min: float = None,
    ylim_max: float = None,
    font_scale: float = 1.0,
):
    rows = sorted(rows, key=lambda x: float(x.get("alpha", 0.0)))
    x_alpha = [float(r.get("lambda", 0.0)) for r in rows]

    y_obv_hr = [safe_get(r, ["obvious", "intervention_mean_hr"], None) for r in rows]
    y_elu_hr = [safe_get(r, ["elusive", "intervention_mean_hr"], None) for r in rows]

    x_obv, y_obv = [], []
    x_elu, y_elu = [], []
    for x, y in zip(x_alpha, y_obv_hr):
        if y is not None:
            x_obv.append(x)
            y_obv.append(float(y))
    for x, y in zip(x_alpha, y_elu_hr):
        if y is not None:
            x_elu.append(x)
            y_elu.append(float(y))

    base_font_size = 15
    title_fontsize = int((base_font_size) * font_scale)
    label_fontsize = int(base_font_size * font_scale)
    tick_fontsize = int((base_font_size) * font_scale)
    legend_fontsize = int(base_font_size * font_scale)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    line1, = ax1.plot(x_obv, y_obv, marker="o", linewidth=2, color="#C65F10", label="Obvious")
    ax1.set_xlabel("Elusive Weight λ", fontsize=label_fontsize)
    ax1.set_ylabel("Obvious Hallucination Rate", color="#C65F10", fontsize=label_fontsize)
    ax1.tick_params(axis="y", labelcolor="#C65F10", labelsize=tick_fontsize)
    ax1.tick_params(axis="x", labelsize=tick_fontsize)
    ax1.set_xlim(0.0, 1.0)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    line2, = ax2.plot(x_elu, y_elu, marker="s", linewidth=2, color="#00CED1", label="Elusive")
    ax2.set_ylabel("Elusive Hallucination Rate", color="#00CED1", fontsize=label_fontsize)
    ax2.tick_params(axis="y", labelcolor="#00CED1", labelsize=tick_fontsize)

    if ylim_min is not None and ylim_max is not None:
        ax1.set_ylim(ylim_min, ylim_max)
        ax2.set_ylim(ylim_min, ylim_max)
    elif share_ylim:
        all_y = y_obv + y_elu
        if all_y:
            y_min = min(all_y)
            y_max = max(all_y)
            if y_max == y_min:
                pad = 0.01
            else:
                pad = (y_max - y_min) * 0.08
            ax1.set_ylim(y_min - pad, y_max + pad)
            ax2.set_ylim(y_min - pad, y_max + pad)

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=False, fontsize=legend_fontsize)

    # plt.title("Qwen2.5-VL-3B-Instruct: Steer Verifiability by Mixing Obvious/Elusive Vectors", fontsize=title_fontsize)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot alpha-mix curve and export case candidates")
    artifact_dir = "pipeline/runs/qwen2.5-vl-7b-instruct"
    metrics_log = "dual_ablation_metrics_mix_lambda.jsonl"
    topk = 5
    ylim_min = 0.2
    ylim_max = 0.8
    font_scale = 1.1
    parser.add_argument(
        "--share_ylim",
        action="store_true",
        help="If set, force the two y-axes to use the same auto-computed y-range.",
    )
    args = parser.parse_args()


    metrics_log_path = os.path.join(artifact_dir, metrics_log)
    if not os.path.exists(metrics_log_path):
        raise FileNotFoundError(f"Metrics log not found: {metrics_log_path}")

    rows = load_jsonl(metrics_log_path)

    output_png = os.path.join(artifact_dir, "mix_alpha_verifiability_curve.png")
    output_cases = os.path.join(artifact_dir, "mix_alpha_case_candidates.json")

    plot_alpha_curve(
        rows,
        output_png,
        share_ylim=args.share_ylim,
        ylim_min=ylim_min,
        ylim_max=ylim_max,
        font_scale=font_scale,
    )
    cases = build_case_candidates(rows, topk=topk)

    with open(output_cases, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)

    print(f"Saved curve: {output_png}")
    print(f"Saved case candidates: {output_cases}")


if __name__ == "__main__":
    main()
