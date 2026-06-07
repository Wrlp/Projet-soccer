import argparse
import time
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from SlowFast.dataset import SoccerClipDataset
from SlowFast.model import SlowFastSimple
from SlowFast.utils import create_splits


def train_epoch(model, loader, criterion, optimizer, device, use_amp=False, log_every=50):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    for step, (slow, fast, label) in enumerate(loader, start=1):
        slow = slow.to(device)
        fast = fast.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(slow, fast)
            loss = criterion(logits, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * label.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()
        total += label.size(0)

        if log_every > 0 and step % log_every == 0:
            print(f'    train step {step}/{len(loader)} loss={loss.item():.4f}')

    return running_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for slow, fast, label in loader:
            slow = slow.to(device)
            fast = fast.to(device)
            label = label.to(device)
            logits = model(slow, fast)
            loss = criterion(logits, label)
            running_loss += loss.item() * label.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == label).sum().item()
            total += label.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--T_s', type=int, default=8)
    parser.add_argument('--alpha', type=int, default=8)
    parser.add_argument('--image-size', type=int, default=112)
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min-delta', type=float, default=1e-4)
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--use-npy', action='store_true')
    parser.add_argument('--root-mp4', type=str, default='SOCCER/outputs/clips/mp4')
    parser.add_argument('--max-per-folder', type=int, default=300)
    parser.add_argument('--splits', type=str, default='SlowFast/splits.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    print('Creating splits...')
    splits = create_splits(
        root_mp4=args.root_mp4,
        out_json=args.splits,
        max_per_folder=args.max_per_folder,
    )

    num_classes = len({e['class_name'] for e in splits['train']})

    train_ds = SoccerClipDataset(
        split_list=splits['train'], T_s=args.T_s, alpha=args.alpha,
        size=args.image_size, use_npy=args.use_npy
    )
    val_ds = SoccerClipDataset(
        split_list=splits['val'], T_s=args.T_s, alpha=args.alpha,
        size=args.image_size, use_npy=args.use_npy
    )
    test_ds = SoccerClipDataset(
        split_list=splits['test'], T_s=args.T_s, alpha=args.alpha,
        size=args.image_size, use_npy=args.use_npy
    )

    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin_memory
    )

    print('Datasets sizes: ', len(train_ds), len(val_ds), len(test_ds))

    model = SlowFastSimple(num_classes=num_classes, pretrained_backbones=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = device.type == 'cuda' and not args.no_amp
    print(f'AMP enabled: {use_amp}')

    best_val_acc = 0.0
    epochs_without_improvement = 0
    ckpt_dir = Path('SlowFast/checkpoints')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_amp=use_amp, log_every=args.log_every
        )
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        t1 = time.time()
        print(f'Epoch {epoch}/{args.epochs}  time={t1-t0:.1f}s')
        print(f'  Train loss={train_loss:.4f} acc={train_acc:.4f}')
        print(f'  Val   loss={val_loss:.4f} acc={val_acc:.4f}')

        # save best
        if val_acc > best_val_acc + args.min_delta:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       ckpt_dir / 'best.pth')
            print('  Saved best checkpoint')
        else:
            epochs_without_improvement += 1
            print(f'  No improvement for {epochs_without_improvement}/{args.patience} epoch(s)')

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f'  Early stopping triggered after {args.patience} epoch(s) without validation improvement')
            break

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                },
                ckpt_dir / f'epoch_{epoch:03d}.pth'
            )
            print(f'  Saved periodic checkpoint: epoch_{epoch:03d}.pth')

    # final test
    print('\nEvaluating on test set...')
    ckpt = ckpt_dir / 'best.pth'
    if ckpt.exists():
        data = torch.load(ckpt, map_location=device)
        model.load_state_dict(data['model_state'])
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f'Test loss={test_loss:.4f} acc={test_acc:.4f}')


if __name__ == '__main__':
    main()
