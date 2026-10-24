"""Train the original causal Transformer on validated CAD question/JSON pairs."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import TinyChatbot
from data_utils import build_vocabulary, encode, load_pairs, question_key, TOKENIZER_VERSION
from cad_schema import VERSION

ROOT = Path(__file__).resolve().parent

class QADataset(Dataset):
    def __init__(self, pairs, vocabulary, max_length):
        self.samples = []
        self.pad = vocabulary['<pad>']
        for question, answer in pairs:
            q = encode(question, vocabulary)
            tokens = [vocabulary['<bos>']] + q + [vocabulary['<sep>']] + encode(answer, vocabulary) + [vocabulary['<eos>']]
            if len(tokens) > max_length:
                raise ValueError('Example exceeds max_length; increase context rather than silently dropping data.')
            targets = tokens[1:]
            targets[:len(q) + 1] = [-100] * (len(q) + 1)
            self.samples.append((tokens[:-1], targets))
        if not self.samples:
            raise ValueError('Empty data partition.')
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]
    def collate(self, samples):
        length = max(len(x) for x, _ in samples)
        return (torch.tensor([x + [self.pad] * (length - len(x)) for x, _ in samples]),
                torch.tensor([y + [-100] * (length - len(y)) for _, y in samples]))

def split_pairs(pairs, seed=42):
    """Group identical JSON targets to prevent paraphrase/parameter leakage."""
    groups = {}
    for pair in pairs:
        groups.setdefault(pair[1], []).append(pair)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    if len(keys) < 10:
        raise ValueError('Need at least 10 distinct command labels for a three-way split.')
    a, b = int(len(keys) * .8), int(len(keys) * .9)
    return tuple([pair for key in part for pair in groups[key]] for part in (keys[:a], keys[a:b], keys[b:]))

def check_disjoint(parts):
    seen = set()
    for part in parts:
        keys = {question_key(q) for q, _ in part}
        if seen & keys:
            raise ValueError('Question leakage across supplied partitions.')
        seen.update(keys)
    labels = {}
    for part in parts:
        for q, a in part:
            k = question_key(q)
            if k in labels and labels[k] != a:
                raise ValueError('Conflicting labels across partitions.')
            labels[k] = a

def validation_loss(model, loader, device):
    model.eval()
    loss_sum, count = 0., 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction='sum').item()
            count += (y != -100).sum().item()
    return loss_sum / count

def fingerprint(pairs):
    return hashlib.sha256(json.dumps(sorted(pairs)).encode()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', nargs='+', default=[str(ROOT / 'dataset_1.csv'), str(ROOT / 'dataset_2.csv'), str(ROOT / 'dataset_3.csv')])
    p.add_argument('--validation', help='Optional explicit held-out CSV; requires --test.')
    p.add_argument('--test', help='Optional explicit held-out CSV; requires --validation.')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--max-length', type=int, default=384)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default=str(ROOT / 'cad_model.pt'))
    p.add_argument('--split-output', default=str(ROOT / 'cad_splits.json'))
    args = p.parse_args()
    try:
        if Path(args.output).resolve().name == 'model.pt':
            raise ValueError('Original model.pt is protected; choose cad_model.pt.')
        if args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0 or args.learning_rate <= 0:
            raise ValueError('Training parameters must be positive.')
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        pairs = load_pairs(args.data)
        if bool(args.validation) != bool(args.test):
            raise ValueError('Supply both --validation and --test.')
        parts = (pairs, load_pairs([args.validation]), load_pairs([args.test])) if args.validation else split_pairs(pairs, args.seed)
        check_disjoint(parts)
        vocabulary = build_vocabulary()
        training = QADataset(parts[0], vocabulary, args.max_length)
        validation = QADataset(parts[1], vocabulary, args.max_length)
        QADataset(parts[2], vocabulary, args.max_length)
    except ValueError as exc:
        p.exit(2, str(exc) + '\n')
    # No files are created until the entire input and all partitions pass validation.
    split_payload = dict(zip(('train', 'validation', 'test'), parts))
    with open(args.split_output, 'w', encoding='utf-8') as handle:
        json.dump(split_payload, handle, indent=2)
    train_loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, collate_fn=training.collate)
    val_loader = DataLoader(validation, batch_size=args.batch_size, collate_fn=validation.collate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyChatbot(len(vocabulary), max_length=args.max_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best, stale, history = float('inf'), 0, []
    print('Device: %s | train/validation/test: %s' % (device, [len(x) for x in parts]))
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, count = 0., 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            n = (y != -100).sum().item()
            total += loss.item() * n
            count += n
        val = validation_loss(model, val_loader, device)
        history.append({'epoch': epoch, 'train_loss': total / count, 'validation_loss': val})
        print(history[-1])
        if val < best - .0001:
            best, stale = val, 0
            torch.save({'model_state': model.state_dict(), 'model_config': model.config,
                'vocabulary': vocabulary, 'tokenizer_version': TOKENIZER_VERSION,
                'schema_version': VERSION, 'seed': args.seed,
                'train_questions': [question_key(q) for q, _ in parts[0]],
                'validation_questions': [question_key(q) for q, _ in parts[1]],
                'split_fingerprints': [fingerprint(part) for part in parts]}, args.output)
        else:
            stale += 1
        if stale >= args.patience:
            break
    with open(Path(args.output).with_suffix('.loss.json'), 'w', encoding='utf-8') as handle:
        json.dump(history, handle, indent=2)
    print('Saved best CAD checkpoint: ' + args.output)

if __name__ == '__main__':
    main()
