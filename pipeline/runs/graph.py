import json
import os
import urllib.request
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

model='qwen2.5-vl-3b-instruct'

OBVIOUS_COLOR = '#C65F10'
ELUSIVE_COLOR = '#00CED1'
OBVIOUS_CMAP = LinearSegmentedColormap.from_list('obvious_cmap', ['#F8E7DA', OBVIOUS_COLOR])
ELUSIVE_CMAP = LinearSegmentedColormap.from_list('elusive_cmap', ['#DDF7F7', ELUSIVE_COLOR])

obvious_path = f'/pipeline/runs/{model}/obvious.jsonl'
elusive_path = f'/pipeline/runs/{model}/elusive.jsonl'
save_dir = f'/pipeline/runs/{model}/'

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

if not os.path.exists(font_path):
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'

try:
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [prop.get_name(), 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']
except Exception as e:
    print(f"error: {e}")

plt.rcParams['axes.unicode_minus'] = False

with open(obvious_path, 'r') as f:
    obvious_data = [json.loads(line) for line in f]
with open(elusive_path, 'r') as f:
    elusive_data = [json.loads(line) for line in f]

metrics_config = [
    {'key': 'mean_hr_increase', 'label': 'HR increase'},
    {'key': 'mean_acc_increase', 'label': 'ACC Increase'},
    {'key': 'mean_unk_tendency_change', 'label': 'UNK Tendency Change'},
    {'key': 'mean_hr_increase_percentage', 'label': 'HR increase %'},
    {'key': 'mean_acc_increase_percentage', 'label': 'ACC Increase %'},
    {'key': 'mean_unk_tendency_change_percentage', 'label': 'UNK Tendency Change %'}
]

all_values_by_metric = {}
for metric_info in metrics_config:
    key = metric_info['key']
    obvious_vals = [d['obvious'][key] for d in obvious_data] + [d['obvious'][key] for d in elusive_data]
    elusive_vals = [d['elusive'][key] for d in obvious_data] + [d['elusive'][key] for d in elusive_data]
    all_values_by_metric[key] = obvious_vals + elusive_vals

y_ranges = {}
for metric_key, all_vals in all_values_by_metric.items():
    v_min = min(all_vals)
    v_max = max(all_vals)
    def round_to_nearest(val, step=0.025):
        if abs(v_max - v_min) < 0.1:
            step = 0.01
        elif abs(v_max - v_min) < 0.5:
            step = 0.025
        elif abs(v_max - v_min) < 2:
            step = 0.1
        else:
            step = 0.25
        return np.ceil(val / step) * step if val > 0 else np.floor(val / step) * step

    y_ranges[metric_key] = (
        round_to_nearest(v_min - 0.01 * (v_max - v_min)),
        round_to_nearest(v_max + 0.01 * (v_max - v_min))
    )

num_metrics = len(metrics_config)
num_cols = 2
num_rows = (num_metrics * 2 + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows))
if num_rows == 1 and num_cols == 1:
    axes = [[axes]]
elif num_rows == 1 or num_cols == 1:
    axes = axes.reshape(-1).tolist()
    axes = [axes[i:i+num_cols] for i in range(0, len(axes), num_cols)]
else:
    axes = axes.tolist() if isinstance(axes[0], (list, np.ndarray)) else [[axes]]

current_row, current_col = 0, 0

for metric_info in metrics_config:
    metric_key = metric_info['key']
    metric_label = metric_info['label']

    ax_a = axes[current_row][current_col]
    ax_a.plot([d['obv_coeff'] for d in obvious_data],
              [d['obvious'][metric_key] for d in obvious_data],
              'o-', label='OHI', markersize=6, color=OBVIOUS_COLOR)
    ax_a.plot([d['elu_coeff'] for d in elusive_data],
              [d['elusive'][metric_key] for d in elusive_data],
              's-', label='EHI', markersize=6, color=ELUSIVE_COLOR)
    ax_a.set_xlabel('Coefficient')
    ax_a.set_ylabel(f'{metric_label} (Obvious test)')
    ax_a.legend()
    ax_a.grid(True, linestyle='--', alpha=0.6)
    v_min, v_max = y_ranges[metric_key]
    ax_a.set_ylim(v_min, v_max)
    step_size = (v_max - v_min) / 10 if (v_max - v_min) > 0.1 else 0.025
    ticks = np.arange(v_min, v_max + step_size, step_size)
    ax_a.set_yticks(ticks[:11])

    current_col += 1
    if current_col >= num_cols:
        current_col = 0
        current_row += 1
    ax_b = axes[current_row][current_col]
    ax_b.plot([d['obv_coeff'] for d in obvious_data],
              [d['obvious'][metric_key] for d in obvious_data],
              'o-', label='OHI', markersize=6, color=OBVIOUS_COLOR)
    ax_b.plot([d['elu_coeff'] for d in elusive_data],
              [d['elusive'][metric_key] for d in elusive_data],
              's-', label='EHI', markersize=6, color=ELUSIVE_COLOR)
    ax_b.set_xlabel('Coefficient')
    ax_b.set_ylabel(f'{metric_label} (Elusive test)')
    ax_b.legend()
    ax_b.grid(True, linestyle='--', alpha=0.6)
    ax_b.set_ylim(v_min, v_max)
    ax_b.set_yticks(ticks[:11])

    current_col += 1
    if current_col >= num_cols:
        current_col = 0
        current_row += 1

total_subplots_needed = num_metrics * 2
total_available_subplots = num_rows * num_cols
for idx in range(total_subplots_needed, total_available_subplots):
    r = idx // num_cols
    c = idx % num_cols
    fig.delaxes(axes[r][c])

plt.tight_layout()
plt.savefig(f'{save_dir}intervention.png', dpi=300, bbox_inches='tight')

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager
import urllib.request

font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

if not os.path.exists(font_path):
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'

try:
    font_manager.fontManager.addfont(font_path)
    prop = font_manager.FontProperties(fname=font_path)
    
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = [prop.get_name(), 'Noto Sans CJK SC', 'WenQuanYi Micro Hei']
except Exception as e:
    print(f"error: {e}")

plt.rcParams['axes.unicode_minus'] = False


with open(obvious_path, 'r') as f:
    obvious_data = [json.loads(line) for line in f]
with open(elusive_path, 'r') as f:
    elusive_data = [json.loads(line) for line in f]

comparison_pairs = [
    {
        "primary": {"key": "mean_acc_increase", "label": "ACC Increase"},
        "percentage": {"key": "mean_acc_increase_percentage", "label": "ACC Increase %"}
    },
    {
        "primary": {"key": "mean_hr_increase", "label": "HR increase"},
        "percentage": {"key": "mean_hr_increase_percentage", "label": "HR increase %"}
    },
    {
        "primary": {"key": "mean_unk_tendency_change", "label": "UNK Tendency Change"},
        "percentage": {"key": "mean_unk_tendency_change_percentage", "label": "UNK Tendency Change %"}
    }
]

for pair in comparison_pairs:
    primary_metric = pair["primary"]
    percentage_metric = pair["percentage"]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    
    obv_primary_x = [d['obvious'][primary_metric['key']] for d in obvious_data]
    obv_primary_y = [d['elusive'][primary_metric['key']] for d in obvious_data]
    obv_coeffs = [d['obv_coeff'] for d in obvious_data]

    elu_primary_x = [d['obvious'][primary_metric['key']] for d in elusive_data]
    elu_primary_y = [d['elusive'][primary_metric['key']] for d in elusive_data]
    elu_coeffs = [d['elu_coeff'] for d in elusive_data]

    scatter1_p = ax1.scatter(obv_primary_x, obv_primary_y,
                             c=obv_coeffs, cmap=OBVIOUS_CMAP,
                             marker='o', s=100, alpha=0.7,
                             edgecolors=OBVIOUS_COLOR, linewidth=1.5,
                             label='OHI')
    scatter2_p = ax1.scatter(elu_primary_x, elu_primary_y,
                             c=elu_coeffs, cmap=ELUSIVE_CMAP,
                             marker='s', s=100, alpha=0.7,
                             edgecolors=ELUSIVE_COLOR, linewidth=1.5,
                             label='EHI')

    ax1.set_xlabel(f'{primary_metric["label"]} (Obvious test set)', fontsize=12)
    ax1.set_ylabel(f'{primary_metric["label"]} (Elusive test set)', fontsize=12)
    ax1.set_title(f'{primary_metric["label"]}: Obvious vs Elusive test set', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)

    all_x_vals = obv_primary_x + elu_primary_x
    all_y_vals = obv_primary_y + elu_primary_y
    lim_min = min(min(all_x_vals), min(all_y_vals))
    lim_max = max(max(all_x_vals), max(all_y_vals))
    ax1.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1, label='y=x')

    cbar1 = plt.colorbar(scatter1_p, ax=ax1, pad=0.02)
    cbar1.set_label('Obvious Coefficient (Darker Colors Indicate Stronger Interventions)', fontsize=10)

    ax2 = axes[1]

    obv_percentage_x = [d['obvious'][percentage_metric['key']] for d in obvious_data]
    obv_percentage_y = [d['elusive'][percentage_metric['key']] for d in obvious_data]

    elu_percentage_x = [d['obvious'][percentage_metric['key']] for d in elusive_data]
    elu_percentage_y = [d['elusive'][percentage_metric['key']] for d in elusive_data]

    scatter3_pct = ax2.scatter(obv_percentage_x, obv_percentage_y,
                               c=obv_coeffs, cmap=OBVIOUS_CMAP,
                               marker='o', s=100, alpha=0.7,
                               edgecolors=OBVIOUS_COLOR, linewidth=1.5,
                               label='OHI')
    scatter4_pct = ax2.scatter(elu_percentage_x, elu_percentage_y,
                               c=elu_coeffs, cmap=ELUSIVE_CMAP,
                               marker='s', s=100, alpha=0.7,
                               edgecolors=ELUSIVE_COLOR, linewidth=1.5,
                               label='EHI')

    ax2.set_xlabel(f'{percentage_metric["label"]} (Obvious test set)', fontsize=12)
    ax2.set_ylabel(f'{percentage_metric["label"]} (Elusive test set)', fontsize=12)
    ax2.set_title(f'{percentage_metric["label"]}: Obvious vs Elusive test set', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='best')
    ax2.grid(True, alpha=0.3)

    all_x_vals_pct = obv_percentage_x + elu_percentage_x
    all_y_vals_pct = obv_percentage_y + elu_percentage_y
    lim_min_pct = min(min(all_x_vals_pct), min(all_y_vals_pct))
    lim_max_pct = max(max(all_x_vals_pct), max(all_y_vals_pct))
    ax2.plot([lim_min_pct, lim_max_pct], [lim_min_pct, lim_max_pct], 'k--', alpha=0.5, linewidth=1, label='y=x')

    cbar2 = plt.colorbar(scatter3_pct, ax=ax2, pad=0.02)
    cbar2.set_label('Elusive Coefficient (Darker Colors Indicate Stronger Interventions)', fontsize=10)

    plt.tight_layout()
    filename = f'{save_dir}scatter_{primary_metric["key"]}_vs_{percentage_metric["key"]}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
