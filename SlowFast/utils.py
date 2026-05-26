import json
from pathlib import Path
from sklearn.model_selection import train_test_split


def create_splits(root_mp4='SOCCER/outputs/clips/mp4', out_json='SlowFast/splits.json',
                  test_size=0.2, val_size=0.1, random_state=42):
    """Create train/val/test splits and save as JSON list entries.

    Each entry is a dict: {'path': str(path), 'label': int, 'class_name': cls}
    """
    from pathlib import Path
    root = Path(root_mp4)
    classes = sorted([p.name for p in root.iterdir() if p.is_dir()])
    items = []
    for idx, cls in enumerate(classes):
        for p in sorted((root / cls).iterdir()):
            if p.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                items.append({'path': str(p), 'label': idx, 'class_name': cls})

    paths = [it['path'] for it in items]
    labels = [it['label'] for it in items]

    train_paths, test_paths, train_lbls, test_lbls = train_test_split(
        paths, labels, test_size=test_size, random_state=random_state, stratify=labels)

    # split train into train+val
    val_ratio = val_size / (1.0 - test_size)
    train_paths2, val_paths, train_lbls2, val_lbls = train_test_split(
        train_paths, train_lbls, test_size=val_ratio, random_state=random_state, stratify=train_lbls)

    def build_list(paths_list, labels_list):
        return [{'path': p, 'label': l, 'class_name': Path(p).parent.name} for p, l in zip(paths_list, labels_list)]

    splits = {
        'train': build_list(train_paths2, train_lbls2),
        'val': build_list(val_paths, val_lbls),
        'test': build_list(test_paths, test_lbls)
    }

    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f'Saved splits to {out}')
    return splits


def load_splits(path='SlowFast/splits.json'):
    p = Path(path)
    if not p.exists():
        return None
    import json
    with open(p, 'r') as f:
        return json.load(f)
