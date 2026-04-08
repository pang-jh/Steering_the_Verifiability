import json
import random
from collections import defaultdict

def load_grouped(path):
    groups = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            groups[item["image"]].append({
                "image": item["image"],
                "text": item["text"],
                "gt": item["gt"],
                "pred": item["pred"]
            })
    return groups

def write_jsonl(items, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def nh_label(item):
    pred = str(item.get("pred", ""))
    if item.get("gt") == "否" and (pred.startswith("否") or pred.startswith("不确定")):
        return "neg_or_uncertain"
    if item.get("gt") == "是" and pred.startswith("是"):
        return "yes_yes"
    return None

def count_nh_binary(items):
    neg_or_uncertain = sum(1 for x in items if nh_label(x) == "neg_or_uncertain")
    yes_yes = sum(1 for x in items if nh_label(x) == "yes_yes")
    return neg_or_uncertain, yes_yes

def collect_by_images(groups, images):
    items = []
    for img in images:
        if img in groups:
            items.extend(groups[img])
    return items

def collect_nh_balanced_by_images(groups, images, rng):
    neg_or_uncertain, yes_yes = [], []
    for img in images:
        for item in groups.get(img, []):
            lab = nh_label(item)
            if lab == "neg_or_uncertain":
                neg_or_uncertain.append(item)
            elif lab == "yes_yes":
                yes_yes.append(item)

    k = min(len(neg_or_uncertain), len(yes_yes))
    if k == 0:
        return []

    out = rng.sample(neg_or_uncertain, k) + rng.sample(yes_yes, k)
    rng.shuffle(out)
    return out

def main():
    obvious_path = "dataset/all_data/3b/filter/obvious_filter.jsonl"
    elusive_path = "dataset/all_data/3b/filter/elusive_filter.jsonl"
    nh_path = "dataset/all_data/3b/filter/nh_filter.jsonl"

    out_train_obvious = "dataset/all_data/3b/filter/split/train_obvious.jsonl"
    out_train_elusive = "dataset/all_data/3b/filter/split/train_elusive.jsonl"
    out_train_nh = "dataset/all_data/3b/filter/split/train_nh_full.jsonl"

    out_val_obvious = "dataset/all_data/3b/filter/split/val_obvious.jsonl"
    out_val_elusive = "dataset/all_data/3b/filter/split/val_elusive.jsonl"
    out_val_nh = "dataset/all_data/3b/filter/split/val_nh_full.jsonl"

    out_test_obvious = "dataset/all_data/3b/filter/split/test_obvious.jsonl"
    out_test_elusive = "dataset/all_data/3b/filter/split/test_elusive.jsonl"
    out_test_nh = "dataset/all_data/3b/filter/split/test_nh_full.jsonl"

    rng = random.Random(42)

    obvious_groups = load_grouped(obvious_path)
    elusive_groups = load_grouped(elusive_path)
    nh_groups = load_grouped(nh_path)

    train_ratio = 0.52
    val_ratio = 0.19
    test_ratio = 0.29
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Ratios must sum to 1"

    all_images = sorted(set(obvious_groups) | set(elusive_groups) | set(nh_groups))
    rng.shuffle(all_images)
    n = len(all_images)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    if n >= 3:
        train_end = max(1, min(train_end, n - 2))
        val_end = max(train_end + 1, min(val_end, n - 1))
    elif n == 2:
        train_end, val_end = 1, 1
    else:
        train_end, val_end = n, n

    train_images = set(all_images[:train_end])
    val_images = set(all_images[train_end:val_end])
    test_images = set(all_images[val_end:])

    train_obvious_items = collect_by_images(obvious_groups, sorted(train_images))
    train_elusive_items = collect_by_images(elusive_groups, sorted(train_images))
    train_nh_items = collect_nh_balanced_by_images(nh_groups, sorted(train_images), rng)

    write_jsonl(train_obvious_items, out_train_obvious)
    write_jsonl(train_elusive_items, out_train_elusive)
    write_jsonl(train_nh_items, out_train_nh)

    val_obvious_items = collect_by_images(obvious_groups, sorted(val_images))
    val_elusive_items = collect_by_images(elusive_groups, sorted(val_images))
    val_nh_items = collect_nh_balanced_by_images(nh_groups, sorted(val_images), rng)

    write_jsonl(val_obvious_items, out_val_obvious)
    write_jsonl(val_elusive_items, out_val_elusive)
    write_jsonl(val_nh_items, out_val_nh)

    test_obvious_items = collect_by_images(obvious_groups, sorted(test_images))
    test_elusive_items = collect_by_images(elusive_groups, sorted(test_images))
    test_nh_items = collect_nh_balanced_by_images(nh_groups, sorted(test_images), rng)

    write_jsonl(test_obvious_items, out_test_obvious)
    write_jsonl(test_elusive_items, out_test_elusive)
    write_jsonl(test_nh_items, out_test_nh)

    tr_neg, tr_yes_yes = count_nh_binary(train_nh_items)
    v_neg, v_yes_yes = count_nh_binary(val_nh_items)
    t_neg, t_yes_yes = count_nh_binary(test_nh_items)

    print(f"Images -> Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
    print(f"Train obvious: {len(train_obvious_items)}, Train elusive: {len(train_elusive_items)}, Train nh: {len(train_nh_items)}")
    print(f"Val obvious: {len(val_obvious_items)}, Val elusive: {len(val_elusive_items)}, Val nh: {len(val_nh_items)}")
    print(f"Test obvious: {len(test_obvious_items)}, Test elusive: {len(test_elusive_items)}, Test nh: {len(test_nh_items)}")

if __name__ == "__main__":
    main()